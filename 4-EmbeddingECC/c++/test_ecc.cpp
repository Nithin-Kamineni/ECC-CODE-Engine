// test_ecc.cpp — minimal C++ test binary for one-chunk ECC comparison.
//
// Reads a small .npy int8 input (N values), applies ECC embedding using the
// Python-exported parity matrix, and writes the result as JSON so the Python
// comparison script can verify correctness.
//
// Build:  make test_ecc
// Usage:
//   ./test_ecc --input vals.npy --parity-matrix P.npy --n 63 --t 2 \
//              --approach search3 --output result.json [--verbose]

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bch.h"
#include "encode.h"
#include "npy_reader.h"

namespace fs = std::filesystem;

// NANDT_TO_K table (matches Python and bch.h)
static int lookup_k(int n, int t) {
    static const struct { int n, t, k; } table[] = {
        {63,1,57},{63,2,51},{63,3,45},{63,4,39},
        {63,5,36},{63,6,30},{63,7,24},{63,8,18},
        {127,1,120},{127,2,113},{127,3,106},{127,4,99},
        {127,5,92},{127,6,85},{127,7,78},{127,8,71},
    };
    for (auto& e : table) if (e.n == n && e.t == t) return e.k;
    throw std::invalid_argument("Unsupported (n, t) for lookup_k");
}

int main(int argc, char** argv) {
    std::string input_path;
    std::string parity_path;
    std::string output_path;
    std::string approach_str = "search3";
    int n_codeword = 63;
    int t_value    = 2;
    int move_range = 4;
    bool verbose   = false;

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", key.c_str()); exit(1); }
            return argv[++i];
        };
        if      (key == "--input")          input_path  = next();
        else if (key == "--parity-matrix")  parity_path = next();
        else if (key == "--output")         output_path = next();
        else if (key == "--approach")       approach_str= next();
        else if (key == "--n")              n_codeword  = std::stoi(next());
        else if (key == "--t")              t_value     = std::stoi(next());
        else if (key == "--move-range")     move_range  = std::stoi(next());
        else if (key == "--verbose")        verbose = true;
        else { fprintf(stderr, "Unknown arg: %s\n", key.c_str()); return 1; }
    }

    if (input_path.empty() || parity_path.empty() || output_path.empty()) {
        fprintf(stderr, "Usage: %s --input vals.npy --parity-matrix P.npy "
                "--n 63 --t 2 --approach search3 --output result.json\n", argv[0]);
        return 1;
    }

    int k = lookup_k(n_codeword, t_value);
    Approach approach = parse_approach(approach_str);

    // Load int8 input values
    std::vector<int8_t> i8_vals = npy_load_int8(input_path);
    size_t N = i8_vals.size();

    // Shift int8 → uint8
    std::vector<uint8_t> u8_vals(N);
    for (size_t i = 0; i < N; i++) u8_vals[i] = (uint8_t)((int)i8_vals[i] + 128);

    if (verbose) {
        fprintf(stderr, "[C++ test] Loaded %zu int8 values from %s\n", N, input_path.c_str());
        fprintf(stderr, "[C++ test] BCH(n=%d, k=%d, t=%d) approach=%s\n",
                n_codeword, k, t_value, approach_str.c_str());
        fprintf(stderr, "[C++ test] Input uint8: ");
        for (size_t i = 0; i < std::min(N, (size_t)20); i++)
            fprintf(stderr, "%d ", (int)u8_vals[i]);
        if (N > 20) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    // Load parity matrix
    PMatrix P = pmatrix_from_npy(parity_path);
    if (verbose)
        fprintf(stderr, "[C++ test] Parity matrix: %d × %d\n", P.k, P.r);

    if (P.k != k || P.r != n_codeword - k) {
        fprintf(stderr, "[C++ test] ERROR: P matrix shape (%d×%d) does not match "
                "BCH(n=%d,k=%d) expected (%d×%d)\n",
                P.k, P.r, n_codeword, k, k, n_codeword - k);
        return 1;
    }

    // Run encoding
    PayloadResult res = process_payload(
        u8_vals.data(), (int)N,
        nullptr,          // no sensitivity weights for this test
        approach,
        n_codeword,       // chunk_size = n (full codeword)
        n_codeword,       // message_parity_size
        k,                // message_size
        P,
        move_range
    );

    if (verbose) {
        fprintf(stderr, "[C++ test] Output uint8 (first 20): ");
        for (size_t i = 0; i < std::min(res.values.size(), (size_t)20); i++)
            fprintf(stderr, "%d ", (int)res.values[i]);
        if (res.values.size() > 20) fprintf(stderr, "...");
        fprintf(stderr, "\n");
        fprintf(stderr, "[C++ test] Distortion: %g\n", res.distortion);
    }

    // Compute L1 vs input
    double l1 = 0.0;
    for (size_t i = 0; i < N; i++)
        l1 += std::abs((int)res.values[i] - (int)u8_vals[i]);

    // Write JSON output
    std::ofstream out(output_path);
    if (!out) { fprintf(stderr, "Cannot open output: %s\n", output_path.c_str()); return 1; }

    out << "{\"values\":[";
    for (size_t i = 0; i < res.values.size(); i++) {
        if (i) out << ',';
        out << (int)res.values[i];
    }
    out << "],\"distortion\":" << res.distortion;
    out << ",\"l1\":" << l1;
    out << ",\"n\":" << n_codeword << ",\"k\":" << k << ",\"t\":" << t_value;
    out << ",\"approach\":\"" << approach_str << "\"}\n";
    out.close();

    fprintf(stdout, "[C++ test] Done. Output: %s\n", output_path.c_str());
    return 0;
}
