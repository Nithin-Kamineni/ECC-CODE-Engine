// ecc_embed.cpp — C++ parallel ECC embedding pipeline.
//
// Equivalent to ecc_embed.py but implements only search3, greedy, and no approaches.
// Parallelisation is cleaner:
//   - std::atomic<size_t>::fetch_add replaces Python's multiprocessing Lock + shared Value
//   - std::thread shares the weights array natively (no memmap-to-disk step)
//   - Per-thread JSONL output is identical to the Python version

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "npy_reader.h"
#include "bch.h"
#include "encode.h"
#include "worker.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---- Simple CLI argument parsing ----
struct Args {
    std::string dataset;
    std::string arch;
    int         quant_bits      = 8;
    int         t_value         = 2;
    std::string approach        = "search3";
    int         codeword        = 63;
    int         workers         = 24;
    std::string patterns_dir;
    std::string chunks_dir;
    std::string sensitivity_dir;
    int         move_range      = 4;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --dataset DS --arch ARCH --quant-bits N --t-value T\n"
        "           --approach (search3|greedy|no) --codeword (63|127|255)\n"
        "           --workers W --patterns-dir PATH --chunks-dir PATH\n"
        "           [--sensitivity-dir PATH] [--move-range R]\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", key.c_str()); exit(1); }
            return argv[++i];
        };
        if      (key == "--dataset")         args.dataset        = next();
        else if (key == "--arch")            args.arch           = next();
        else if (key == "--quant-bits")      args.quant_bits     = std::stoi(next());
        else if (key == "--t-value")         args.t_value        = std::stoi(next());
        else if (key == "--approach")        args.approach       = next();
        else if (key == "--codeword")        args.codeword       = std::stoi(next());
        else if (key == "--workers")         args.workers        = std::stoi(next());
        else if (key == "--patterns-dir")    args.patterns_dir   = next();
        else if (key == "--chunks-dir")      args.chunks_dir     = next();
        else if (key == "--sensitivity-dir") args.sensitivity_dir= next();
        else if (key == "--move-range")      args.move_range     = std::stoi(next());
        else if (key == "--help") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown argument: %s\n", key.c_str()); print_usage(argv[0]); exit(1); }
    }
    if (args.dataset.empty() || args.arch.empty() || args.patterns_dir.empty() || args.chunks_dir.empty()) {
        fprintf(stderr, "Missing required arguments\n");
        print_usage(argv[0]);
        exit(1);
    }
    return args;
}

// ---- Sanitize layer name (matches Python _sanitize) ----
static std::string sanitize(const std::string& s) {
    std::string r = s;
    for (char& c : r) if (c == '.' || c == '/' || c == ' ') c = '_';
    return r;
}

// ---- Convert dataset name to lowercase path component ----
static std::string ds_lower(const std::string& ds) {
    std::string r = ds;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

// ---- Load sensitivity array for a layer (from sens.npy in patterns dir) ----
// Returns empty vector if not found.
static std::vector<float> load_sensitivity(
    const std::string& sens_npy_path,
    const std::vector<int64_t>* perm  // permutation (from perm_file) or nullptr
) {
    if (!fs::exists(sens_npy_path)) return {};

    std::vector<float> sens;
    try {
        sens = npy_load_float32(sens_npy_path);
    } catch (...) {
        return {};
    }

    if (perm && !perm->empty()) {
        // Apply permutation: sens_permuted[i] = sens_original[perm[i]]
        std::vector<float> s2(perm->size());
        for (size_t i = 0; i < perm->size(); i++) {
            size_t idx = (size_t)(*perm)[i];
            s2[i] = (idx < sens.size()) ? sens[idx] : 0.0f;
        }
        return s2;
    }
    return sens;
}

// ---- Process one layer ----
static void process_layer(
    const std::string& layer_name,
    const json&        entry,
    const Args&        args,
    int                chunk_size,
    int                message_parity_size,
    int                message_size,
    const PMatrix&     P,
    Approach           approach,
    const std::string& bit_label,
    const std::string& m_tag)
{
    // weights_perm_file from manifest
    std::string weights_file;
    if (entry.contains("weights_perm_file") && !entry["weights_perm_file"].is_null())
        weights_file = entry["weights_perm_file"].get<std::string>();

    if (weights_file.empty() || !fs::exists(weights_file)) {
        fprintf(stdout, "  [skip] %s: weights_perm_file missing (%s)\n",
                layer_name.c_str(), weights_file.c_str());
        return;
    }

    // Load int8 weights and shift to uint8
    std::vector<int8_t> w_int8;
    try {
        w_int8 = npy_load_int8(weights_file);
    } catch (const std::exception& e) {
        fprintf(stdout, "  [skip] %s: load failed: %s\n", layer_name.c_str(), e.what());
        return;
    }
    size_t N = w_int8.size();

    // Shift int8 (-128..127) → uint8 (0..255)
    std::vector<uint8_t> w_u8(N);
    for (size_t i = 0; i < N; i++)
        w_u8[i] = (uint8_t)((int)w_int8[i] + 128);

    // Load sensitivity array (optional, for search3/greedy)
    std::vector<float>  sens_vec;
    const float*        sens_ptr = nullptr;
    if (approach != Approach::NO) {
        // Try to load sens.npy from patterns dir
        std::string layer_safe = sanitize(layer_name);
        std::string sens_path  = args.patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                                 args.arch + "/PTQ/" + bit_label + "/" + layer_safe + "_sens.npy";
        // Also load perm file if available (to map sensitivity to permuted order)
        std::string perm_path;
        if (entry.contains("perm_file") && !entry["perm_file"].is_null())
            perm_path = entry["perm_file"].get<std::string>();

        std::vector<int64_t> perm_vec;
        if (!perm_path.empty() && fs::exists(perm_path)) {
            try { perm_vec = npy_load_int64(perm_path); } catch (...) {}
        }

        sens_vec = load_sensitivity(sens_path, perm_vec.empty() ? nullptr : &perm_vec);
        if (!sens_vec.empty()) {
            if (sens_vec.size() < N) sens_vec.resize(N, 0.0f);
            sens_ptr = sens_vec.data();
        }
    }

    // Output directory for this layer's chunks
    std::string layer_safe = sanitize(layer_name);
    std::string out_dir = args.chunks_dir + "/" + ds_lower(args.dataset) + "/" +
                          args.arch + "/PTQ/" + bit_label + "/" +
                          m_tag + "/" + args.approach + "/" + layer_safe;

    fprintf(stdout, "  [embed] %s  N=%zu  out=%s\n",
            layer_name.c_str(), N, out_dir.c_str());
    fflush(stdout);

    // Spawn worker threads (weights array shared in memory — no disk temp file)
    run_layer_workers(
        N, (size_t)chunk_size,
        w_u8.data(), sens_ptr,
        out_dir,
        approach,
        message_parity_size, message_size,
        P,
        args.workers,
        args.move_range);
}

// ---- main --------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    Approach approach = parse_approach(args.approach);

    // BCH parameters
    int message_size        = bch_message_size(args.codeword, args.t_value);
    int message_parity_size = args.codeword;
    int chunk_size = (approach == Approach::NO || approach == Approach::SEARCH3 || approach == Approach::GREEDY)
                   ? message_parity_size  // n (full codeword, not shortened)
                   : message_size;        // parfix uses k

    // Bit label and manifest path
    std::string bit_label = std::to_string(args.quant_bits) + "-bit";
    std::string m_tag     = "M" + std::to_string(args.codeword) + "_t" + std::to_string(args.t_value);

    std::string manifest_path = args.patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                                args.arch + "/PTQ/" + bit_label + "/pattern_manifest.json";

    if (!fs::exists(manifest_path)) {
        fprintf(stdout, "[skip] No manifest: %s\n", manifest_path.c_str());
        return 0;
    }

    // Load manifest
    json manifest;
    {
        std::ifstream f(manifest_path);
        if (!f) { fprintf(stderr, "Cannot open manifest: %s\n", manifest_path.c_str()); return 1; }
        f >> manifest;
    }

    fprintf(stdout, "[ecc_embed_cpp] dataset=%s arch=%s bits=%s t=%d approach=%s "
            "codeword=%d chunk_size=%d msg_size=%d workers=%d\n",
            args.dataset.c_str(), args.arch.c_str(), bit_label.c_str(),
            args.t_value, args.approach.c_str(), args.codeword,
            chunk_size, message_size, args.workers);
    fprintf(stdout, "[ecc_embed_cpp] manifest=%s (%zu layers)\n",
            manifest_path.c_str(), manifest.size());
    fflush(stdout);

    // Build BCH parity matrix once, shared by all workers
    PMatrix P;
    if (approach != Approach::NO) {
        fprintf(stdout, "[bch] Computing BCH(%d, %d, t=%d) parity matrix ...\n",
                args.codeword, message_size, args.t_value);
        fflush(stdout);
        auto P_raw = bch_parity_matrix(args.codeword, message_size, args.t_value);
        P = PMatrix(P_raw);
        fprintf(stdout, "[bch] Parity matrix built: %d × %d\n", P.k, P.r);
        fflush(stdout);
    }

    // Process each layer
    for (const auto& [layer_name, entry] : manifest.items()) {
        process_layer(layer_name, entry, args, chunk_size, message_parity_size,
                      message_size, P, approach, bit_label, m_tag);
    }

    fprintf(stdout, "[ecc_embed_cpp] Done.\n");
    return 0;
}
