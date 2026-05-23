#pragma once
// worker.h — worker thread with std::atomic<size_t> chunk allocator.
//
// Key improvements over Python:
//   - std::atomic<size_t>::fetch_add: single instruction, no lock, no shared Value
//   - Weights array lives in memory and is read directly — no memmap-to-disk step
//   - Per-thread JSONL file: zero contention between threads

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "encode.h"

namespace fs = std::filesystem;

// ---- Write one JSONL record to file -----------------------------------------
// Mirrors exactly the Python JSONL record format (including "distorsion" typo).
static inline void write_jsonl_record(
    FILE* f,
    int   thread_id,
    size_t start,
    size_t end,
    const std::vector<uint8_t>& mutated_u8,
    double distortion)
{
    // Convert uint8 back to int8 range (-128..127) for output
    // (Same as Python: mutated_int8 = mutated_u8 - 128)
    fprintf(f, "{\"p\":%d,\"start\":%zu,\"end\":%zu,\"count\":%zu,\"values\":[",
            thread_id, start, end, end - start + 1);
    bool first = true;
    for (uint8_t v : mutated_u8) {
        if (!first) fputc(',', f);
        first = false;
        fprintf(f, "%d", (int)(int8_t)(v - 128));  // cast to int8 for sign
    }
    fprintf(f, "],\"distorsion\":%.8g,\"status\":\"ok\"}\n", distortion);
    fflush(f);  // line-buffered write
}

// ---- Worker thread function --------------------------------------------------
void worker_fn(
    int                      thread_id,
    std::atomic<size_t>&     next_idx,       // shared atomic counter
    size_t                   N,              // total elements
    size_t                   chunk_size,     // elements per work unit
    const uint8_t*           weights,        // read-only shared array (uint8, N elements)
    const float*             sens,           // read-only sensitivity array (may be nullptr)
    const std::string&       out_dir,
    Approach                 approach,
    int                      message_parity_size,
    int                      message_size,
    const PMatrix&           P,
    int                      move_range = 4)
{
    // Open per-thread output file
    std::string out_path = out_dir + "/chunks_p" + std::to_string(thread_id) + ".jsonl";
    FILE* f = fopen(out_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "[worker %d] Cannot open: %s\n", thread_id, out_path.c_str());
        return;
    }

    size_t progress_interval = chunk_size * 1000;

    while (true) {
        // Atomically claim the next chunk — single fetch_add, no lock needed
        size_t start = next_idx.fetch_add(chunk_size, std::memory_order_relaxed);
        if (start >= N) break;
        size_t end = std::min(start + chunk_size, N) - 1;  // inclusive
        size_t count = end - start + 1;

        // Run ECC encoding on this slice (direct pointer into shared array)
        auto result = process_payload(
            weights + start, (int)count,
            sens ? (sens + start) : nullptr,
            approach,
            (int)chunk_size,
            message_parity_size,
            message_size,
            P,
            move_range);

        write_jsonl_record(f, thread_id, start, end, result.values, result.distortion);

        // Progress report from thread 0 (mirrors Python worker logging)
        if (thread_id == 0 && (start % progress_interval) == 0) {
            fprintf(stderr, "[progress] thread=0 idx=%zu\n", start);
            fflush(stderr);
        }
    }

    fclose(f);
}

// ---- Coverage validator -------------------------------------------------------
// Mirrors Python validate_coverage(): scans all chunks_p*.jsonl in log_dir
// and checks that [0, N) is fully and continuously covered by ok records.
inline bool validate_coverage(size_t N, const std::string& log_dir) {
    std::vector<std::pair<size_t, size_t>> intervals;

    for (const auto& entry : fs::directory_iterator(log_dir)) {
        std::string name = entry.path().filename().string();
        // Accept both chunks_p*.jsonl and chunks_gap*.jsonl
        if (name.find("chunks_") == std::string::npos) continue;
        if (name.rfind(".jsonl") == std::string::npos) continue;

        FILE* fh = fopen(entry.path().c_str(), "r");
        if (!fh) continue;

        char line[1 << 20];  // 1 MB buffer per line
        while (fgets(line, sizeof(line), fh)) {
            // Quick parse: find "status":"ok" and extract start/end
            if (!strstr(line, "\"ok\"")) continue;
            size_t s = 0, e = 0;
            if (sscanf(line, "%*[^\"start\"]\"start\":%zu", &s) < 1) {
                // Try alternative parsing
                const char* sp = strstr(line, "\"start\":");
                const char* ep = strstr(line, "\"end\":");
                if (!sp || !ep) continue;
                sscanf(sp, "\"start\":%zu", &s);
                sscanf(ep, "\"end\":%zu",   &e);
            } else {
                const char* ep = strstr(line, "\"end\":");
                if (!ep) continue;
                sscanf(ep, "\"end\":%zu", &e);
            }
            intervals.emplace_back(s, e);
        }
        fclose(fh);
    }

    if (intervals.empty() && N > 0) {
        fprintf(stderr, "[coverage] ERROR: No completed intervals in %s\n", log_dir.c_str());
        return false;
    }

    // Deduplicate and sort
    std::sort(intervals.begin(), intervals.end());
    intervals.erase(std::unique(intervals.begin(), intervals.end()), intervals.end());

    size_t cur = 0;
    for (auto [s, e] : intervals) {
        if (s != cur) {
            fprintf(stderr, "[coverage] ERROR: Gap at index %zu (next starts at %zu)\n", cur, s);
            return false;
        }
        cur = e + 1;
    }
    if (cur != N) {
        fprintf(stderr, "[coverage] ERROR: Did not reach N=%zu; last covered=%zu\n", N, cur - 1);
        return false;
    }
    fprintf(stdout, "[coverage] OK: [0, %zu) covered by %zu chunks.\n", N, intervals.size());
    return true;
}

// ---- Run all workers for one layer -------------------------------------------
inline void run_layer_workers(
    size_t           N,
    size_t           chunk_size,
    const uint8_t*   weights,     // shared array of uint8 values (length N)
    const float*     sens,        // sensitivity array or nullptr
    const std::string& out_dir,
    Approach         approach,
    int              message_parity_size,
    int              message_size,
    const PMatrix&   P,
    int              num_workers,
    int              move_range = 4)
{
    // Create output directory
    fs::create_directories(out_dir);

    // Remove any pre-existing chunk files
    for (const auto& entry : fs::directory_iterator(out_dir)) {
        std::string name = entry.path().filename().string();
        if (name.find("chunks_p") == 0 && name.rfind(".jsonl") != std::string::npos)
            fs::remove(entry.path());
    }

    // Shared atomic counter — replaces Python's mp.Value + mp.Lock
    std::atomic<size_t> next_idx{0};

    // Spawn worker threads — weights array shared natively (no memmap-to-disk!)
    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (int p = 0; p < num_workers; p++) {
        threads.emplace_back(worker_fn,
            p, std::ref(next_idx),
            N, chunk_size,
            weights, sens,
            out_dir,
            approach,
            message_parity_size, message_size,
            std::cref(P),
            move_range);
    }
    for (auto& t : threads) t.join();

    validate_coverage(N, out_dir);
}
