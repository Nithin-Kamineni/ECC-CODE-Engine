// ecc_embed.cpp — C++ parallel ECC embedding pipeline.
//
// Equivalent to ecc_embed.py but implements only search3, greedy, and no approaches.
// Parallelisation is cleaner:
//   - std::atomic<size_t>::fetch_add replaces Python's multiprocessing Lock + shared Value
//   - std::thread shares the weights array natively (no memmap-to-disk step)
//   - Per-thread JSONL output is identical to the Python version
//
// Top-K pattern flow (mirrors ecc_embed.py's two-phase search3 flow):
//   --approach greedy --top-patterns-dir D --best-patterns-dir B
//       Loads top_patterns_manifest.json (top-K candidate patterns per layer,
//       written by 3-PatternFinder).  For each layer with >1 candidate, scores
//       every candidate via score_layer_workers() (greedy total distortion —
//       sum of per-chunk result.distortion, no chunk files written) and picks
//       the argmin.  Writes best_pattern_selection.json.
//   --approach search3 --top-patterns-dir D --best-patterns-dir B
//       Reads best_pattern_selection.json, resolves each layer's winning
//       candidate from top_patterns_manifest.json, and runs the normal
//       search3 embedding (writes chunks_p*.jsonl) using that pattern.
//
// If --top-patterns-dir is not supplied, behaves exactly as before: reads
// pattern_manifest.json (rank-0 pattern only).

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    std::string parity_matrix;  // optional: path to Python-exported P matrix .npy
    bool        no_sensitivity  = false;  // --no-sensitivity: skip loading, use sens=1.0 for all
    float       sens_norm_min   = 0.5f;   // --sens-norm-min: floor of [min,1.0] normalization
    std::string qmode           = "PTQ";  // PTQ or QAT — selects patterns/{qmode}/ subdirectory
    std::string top_patterns_dir;         // top_patterns_manifest.json root (top-K flow)
    std::string best_patterns_dir;        // best_pattern_selection.json root (top-K flow)
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --dataset DS --arch ARCH --quant-bits N --t-value T\n"
        "           --approach (search3|greedy|no) --codeword (63|127|255)\n"
        "           --workers W --patterns-dir PATH --chunks-dir PATH\n"
        "           [--sensitivity-dir PATH] [--move-range R]\n"
        "           [--parity-matrix PATH]  # .npy exported by export_parity_matrix.py\n"
        "           [--no-sensitivity]      # disable sensitivity: all bucket weights = 1.0\n"
        "           [--sens-norm-min F]     # sensitivity normalization floor [0.0,1.0] (default 0.5)\n"
        "           [--qmode (PTQ|QAT)]     # selects patterns/{qmode}/ subdirectory (default PTQ)\n"
        "           [--top-patterns-dir PATH]  # top_patterns_manifest.json root (top-K flow)\n"
        "           [--best-patterns-dir PATH] # best_pattern_selection.json root (top-K flow)\n",
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
        else if (key == "--parity-matrix")   args.parity_matrix  = next();
        else if (key == "--no-sensitivity")  args.no_sensitivity = true;
        else if (key == "--sens-norm-min")   args.sens_norm_min  = std::stof(next());
        else if (key == "--qmode")           args.qmode          = next();
        else if (key == "--top-patterns-dir")  args.top_patterns_dir  = next();
        else if (key == "--best-patterns-dir") args.best_patterns_dir = next();
        else if (key == "--help") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown argument: %s\n", key.c_str()); print_usage(argv[0]); exit(1); }
    }
    if (args.dataset.empty() || args.arch.empty() || args.patterns_dir.empty() || args.chunks_dir.empty()) {
        fprintf(stderr, "Missing required arguments\n");
        print_usage(argv[0]);
        exit(1);
    }
    if (args.qmode != "PTQ" && args.qmode != "QAT") {
        fprintf(stderr, "--qmode must be PTQ or QAT (got '%s')\n", args.qmode.c_str());
        exit(1);
    }
    if (args.sens_norm_min < 0.0f || args.sens_norm_min > 1.0f) {
        fprintf(stderr, "--sens-norm-min must be in [0.0, 1.0] (got %.3f)\n", args.sens_norm_min);
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
// min_norm: floor of [min_norm, 1.0] normalization range (matches SENS_NORM_MIN in env.sh).
// NOTE: only called for the raw binary _sens.npy fallback path.  The primary
// _sens_py.npy path (pre-normalized by export_sensitivity.py) is loaded directly
// in process_layer() without going through this function.
static std::vector<float> load_sensitivity(
    const std::string& sens_npy_path,
    const std::vector<int64_t>* perm,  // permutation (from perm_file) or nullptr
    float min_norm = 0.5f
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
        sens = std::move(s2);
    }

    // Normalize to [min_norm, 1.0] — mirrors ecc_embed.py _load_layer_sensitivity():
    //   sens_arr = min_norm + (1 - min_norm) * (sens_arr / max_val)
    // Called only for the raw binary _sens.npy fallback (binary {0,1} indicators),
    // so min_norm=0.5 gives {0.5, 1.0} which matches the original behavior.
    float max_val = *std::max_element(sens.begin(), sens.end());
    if (max_val > 0.0f) {
        for (auto& v : sens) v = min_norm + (1.0f - min_norm) * (v / max_val);
    } else {
        std::fill(sens.begin(), sens.end(), 1.0f);  // all-zero fallback → uniform weight
    }
    return sens;
}

// ---- Load permutation-independent dense sensitivity array -------------------
// Indexed by ORIGINAL flat_idx, normalized to [0.5, 1.0].  Re-permuting it for
// any candidate pattern's perm_file (sens_vec[k] = dense[perm[k]]) is bit-exact
// to running _load_layer_sensitivity() with that pattern's perm_arr, since
// normalization divides by max(sens_dict.values()), which is permutation-
// independent.
//
// Priority 1: {layer}_sens_dense_py.npy (export_sensitivity.py) — already
//             dense + normalized.
// Priority 2: {layer}_sens.npy (prepare_patterns.py) — dense (indexed by
//             flat_idx), NOT normalized.  Normalized here with the same
//             formula; mathematically equivalent to Priority 1.
// Returns empty vector if neither file exists (caller treats as sens=1.0).
static std::vector<float> load_dense_sensitivity(
    const std::string& layer_name,
    const Args&        args,
    const std::string& bit_label)
{
    std::string layer_safe = sanitize(layer_name);
    std::string base_path = args.patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                             args.arch + "/" + args.qmode + "/" + bit_label + "/" + layer_safe;

    std::string dense_path = base_path + "_sens_dense_py.npy";
    if (fs::exists(dense_path)) {
        try {
            auto dense = npy_load_float32(dense_path);
            fprintf(stdout, "  [sens] %s: dense sensitivity (%zu values) from %s\n",
                    layer_name.c_str(), dense.size(), dense_path.c_str());
            return dense;
        } catch (...) {}
    }

    std::string sens_path = base_path + "_sens.npy";
    if (fs::exists(sens_path)) {
        try {
            auto dense = npy_load_float32(sens_path);
            float max_val = dense.empty() ? 0.0f : *std::max_element(dense.begin(), dense.end());
            if (max_val > 0.0f) {
                for (auto& v : dense)
                    v = args.sens_norm_min + (1.0f - args.sens_norm_min) * (v / max_val);
            } else {
                std::fill(dense.begin(), dense.end(), 1.0f);
            }
            fprintf(stdout,
                "  [sens] %s: WARNING — %s_sens_dense_py.npy not found; using "
                "_sens.npy (dense, normalized in C++).  Run export_sensitivity.py "
                "for exact Python match.\n", layer_name.c_str(), layer_safe.c_str());
            return dense;
        } catch (...) {}
    }

    return {};
}

// ---- Re-permute a dense sensitivity array for one candidate pattern ----------
// sens_vec[k] = dense[perm_vec[k]]
// min_norm is used as the out-of-bounds fallback (consistent with normalization floor).
static std::vector<float> remap_sensitivity(
    const std::vector<float>&   dense,
    const std::vector<int64_t>& perm_vec,
    float                       min_norm = 0.5f)
{
    std::vector<float> sens_vec(perm_vec.size());
    for (size_t k = 0; k < perm_vec.size(); k++) {
        size_t idx = (size_t)perm_vec[k];
        sens_vec[k] = (idx < dense.size()) ? dense[idx] : min_norm;
    }
    return sens_vec;
}

// ---- Load weights_perm_file as uint8 (shifted from int8) --------------------
static bool load_weights_u8(const std::string& weights_file, std::vector<uint8_t>& out) {
    if (weights_file.empty() || !fs::exists(weights_file)) return false;
    std::vector<int8_t> w_int8;
    try {
        w_int8 = npy_load_int8(weights_file);
    } catch (...) {
        return false;
    }
    out.resize(w_int8.size());
    for (size_t i = 0; i < w_int8.size(); i++)
        out[i] = (uint8_t)((int)w_int8[i] + 128);
    return true;
}

// ---- Embed one layer: load weights, run workers, write chunks_p*.jsonl ------
static void embed_layer(
    const std::string& layer_name,
    const std::string& weights_file,
    const float*        sens_ptr,
    const std::string&  out_dir,
    const Args&         args,
    int                 chunk_size,
    int                 message_parity_size,
    int                 message_size,
    const PMatrix&      P,
    Approach            approach)
{
    std::vector<uint8_t> w_u8;
    if (!load_weights_u8(weights_file, w_u8)) {
        fprintf(stdout, "  [skip] %s: weights_perm_file missing/unreadable (%s)\n",
                layer_name.c_str(), weights_file.c_str());
        return;
    }
    size_t N = w_u8.size();

    fprintf(stdout, "  [embed] %s  N=%zu  out=%s\n", layer_name.c_str(), N, out_dir.c_str());
    fflush(stdout);

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

// ---- Process one layer (legacy single-pattern flow: pattern_manifest.json) --
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

    // Load sensitivity array (optional, for search3/greedy)
    // Skipped entirely when --no-sensitivity is set (EMBED_SENSITIVITY=false in env.sh):
    // sens_ptr stays nullptr → build_bucket_meta sets meta.sens=1.0f for all buckets.
    std::vector<float>  sens_vec;
    const float*        sens_ptr = nullptr;
    if (approach != Approach::NO && !args.no_sensitivity) {
        std::string layer_safe = sanitize(layer_name);
        std::string base_path  = args.patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                                 args.arch + "/" + args.qmode + "/" + bit_label + "/" + layer_safe;

        // ── Priority 1: _sens_py.npy (exported by export_sensitivity.py) ────────
        // Already permuted (rank 0) and normalized to [0.5, 1.0] by Python's
        // _load_layer_sensitivity() — uses continuous Taylor scores, exact match
        // with the Python runner.  Load directly with no further processing.
        std::string sens_py_path = base_path + "_sens_py.npy";
        if (fs::exists(sens_py_path)) {
            try {
                sens_vec = npy_load_float32(sens_py_path);
                float s_min = *std::min_element(sens_vec.begin(), sens_vec.end());
                float s_max = *std::max_element(sens_vec.begin(), sens_vec.end());
                fprintf(stdout,
                    "  [sens] %s: Python sensitivity (%zu values, range=[%.4f, %.4f])\n",
                    layer_name.c_str(), sens_vec.size(), s_min, s_max);
                fflush(stdout);
            } catch (...) {
                sens_vec.clear();
            }
        }

        // ── Priority 2: _sens.npy fallback (indicator {0,1} + perm + normalize) ─
        // Uses binary indicators from prepare_patterns.py — results in {0.5, 1.0}
        // after normalization (not continuous).  Used when export_sensitivity.py
        // has not been run yet.
        if (sens_vec.empty()) {
            std::string sens_path = base_path + "_sens.npy";
            std::string perm_path;
            if (entry.contains("perm_file") && !entry["perm_file"].is_null())
                perm_path = entry["perm_file"].get<std::string>();

            std::vector<int64_t> perm_vec;
            if (!perm_path.empty() && fs::exists(perm_path)) {
                try { perm_vec = npy_load_int64(perm_path); } catch (...) {}
            }

            sens_vec = load_sensitivity(sens_path, perm_vec.empty() ? nullptr : &perm_vec,
                                         args.sens_norm_min);
            if (!sens_vec.empty()) {
                fprintf(stdout,
                    "  [sens] %s: WARNING — _sens_py.npy not found; using indicator "
                    "_sens.npy (binary {0.5,1.0}).  Run export_sensitivity.py for "
                    "exact Python match.\n", layer_name.c_str());
                fflush(stdout);
            }
        }

        if (!sens_vec.empty()) sens_ptr = sens_vec.data();
    } else if (args.no_sensitivity) {
        fprintf(stdout, "  [sens] %s: sensitivity disabled (--no-sensitivity) → all weights = 1.0\n",
                layer_name.c_str());
        fflush(stdout);
    }

    // Output directory for this layer's chunks
    std::string layer_safe = sanitize(layer_name);
    std::string out_dir = args.chunks_dir + "/" + ds_lower(args.dataset) + "/" +
                          args.arch + "/" + args.qmode + "/" + bit_label + "/" +
                          m_tag + "/" + args.approach + "/" + layer_safe;

    embed_layer(layer_name, weights_file, sens_ptr, out_dir, args,
                 chunk_size, message_parity_size, message_size, P, approach);
}

// ---- Process one layer using a specific top-K candidate pattern -------------
// Used by the search3-with-best-pattern flow.  `dense_sens` is the
// permutation-independent dense sensitivity array (may be empty).
static void process_layer_topk(
    const std::string& layer_name,
    const json&        rank_entry,
    const std::vector<float>& dense_sens,
    const Args&        args,
    int                chunk_size,
    int                message_parity_size,
    int                message_size,
    const PMatrix&     P,
    Approach           approach,
    const std::string& bit_label,
    const std::string& m_tag)
{
    std::string weights_file;
    if (rank_entry.contains("weights_perm_file") && !rank_entry["weights_perm_file"].is_null())
        weights_file = rank_entry["weights_perm_file"].get<std::string>();

    std::vector<float> sens_vec;
    const float*       sens_ptr = nullptr;
    if (approach != Approach::NO && !args.no_sensitivity && !dense_sens.empty()) {
        std::string perm_path;
        if (rank_entry.contains("perm_file") && !rank_entry["perm_file"].is_null())
            perm_path = rank_entry["perm_file"].get<std::string>();

        std::vector<int64_t> perm_vec;
        if (!perm_path.empty() && fs::exists(perm_path)) {
            try { perm_vec = npy_load_int64(perm_path); } catch (...) {}
        }
        if (!perm_vec.empty()) {
            sens_vec = remap_sensitivity(dense_sens, perm_vec, args.sens_norm_min);
            sens_ptr = sens_vec.data();
        }
    } else if (args.no_sensitivity) {
        fprintf(stdout, "  [sens] %s: sensitivity disabled (--no-sensitivity) → all weights = 1.0\n",
                layer_name.c_str());
    }

    std::string layer_safe = sanitize(layer_name);
    std::string out_dir = args.chunks_dir + "/" + ds_lower(args.dataset) + "/" +
                          args.arch + "/" + args.qmode + "/" + bit_label + "/" +
                          m_tag + "/" + args.approach + "/" + layer_safe;

    embed_layer(layer_name, weights_file, sens_ptr, out_dir, args,
                 chunk_size, message_parity_size, message_size, P, approach);
}

// ---- Phase 1: greedy selection across top-K candidate patterns --------------
// For each layer with >1 candidate pattern, score every candidate via
// score_layer_workers() (sum of per-chunk result.distortion — the C++ analog
// of summing the "distorsion" field across chunks_p*.jsonl) and pick the
// argmin.  Writes best_pattern_selection.json.
static void run_greedy_selection(
    const json&        top_manifest,
    const Args&        args,
    int                chunk_size,
    int                message_parity_size,
    int                message_size,
    const PMatrix&     P,
    const std::string& bit_label,
    const std::string& m_tag)
{
    json selection;
    selection["dataset"]  = ds_lower(args.dataset);
    selection["arch"]     = args.arch;
    selection["qmode"]    = args.qmode;
    selection["bits"]     = args.quant_bits;
    selection["codeword"] = args.codeword;
    selection["t_value"]  = args.t_value;
    selection["layers"]   = json::object();

    for (const auto& [layer_name, entry] : top_manifest.items()) {
        if (!entry.contains("top_patterns") || entry["top_patterns"].empty()) {
            fprintf(stdout, "  [skip] %s: no top_patterns entries\n", layer_name.c_str());
            continue;
        }
        const auto& top_patterns = entry["top_patterns"];

        if (top_patterns.size() == 1) {
            const auto& rank0 = top_patterns[0];
            json sel;
            sel["best_rank"]              = rank0.value("rank", 0);
            sel["best_param"]             = rank0.contains("param") ? rank0.at("param") : json(nullptr);
            sel["best_total_distortion"]  = nullptr;
            sel["all_total_distortion"]   = json::array();
            sel["skipped"]                = true;
            selection["layers"][layer_name] = sel;
            continue;
        }

        std::vector<float> dense_sens;
        if (!args.no_sensitivity) {
            dense_sens = load_dense_sensitivity(layer_name, args, bit_label);
        }

        std::vector<double> all_distortion;
        all_distortion.reserve(top_patterns.size());

        for (const auto& rank_entry : top_patterns) {
            std::string weights_file;
            if (rank_entry.contains("weights_perm_file") && !rank_entry["weights_perm_file"].is_null())
                weights_file = rank_entry["weights_perm_file"].get<std::string>();

            std::vector<uint8_t> w_u8;
            if (!load_weights_u8(weights_file, w_u8)) {
                fprintf(stdout, "  [skip] %s rank%d: weights_perm_file missing/unreadable (%s)\n",
                        layer_name.c_str(), rank_entry.value("rank", -1), weights_file.c_str());
                all_distortion.push_back(std::numeric_limits<double>::max());
                continue;
            }

            std::vector<float> sens_vec;
            const float*       sens_ptr = nullptr;
            if (!dense_sens.empty()) {
                std::string perm_path;
                if (rank_entry.contains("perm_file") && !rank_entry["perm_file"].is_null())
                    perm_path = rank_entry["perm_file"].get<std::string>();

                std::vector<int64_t> perm_vec;
                if (!perm_path.empty() && fs::exists(perm_path)) {
                    try { perm_vec = npy_load_int64(perm_path); } catch (...) {}
                }
                if (!perm_vec.empty()) {
                    sens_vec = remap_sensitivity(dense_sens, perm_vec, args.sens_norm_min);
                    sens_ptr = sens_vec.data();
                }
            }

            double total = score_layer_workers(
                w_u8.size(), (size_t)chunk_size,
                w_u8.data(), sens_ptr,
                Approach::GREEDY,
                message_parity_size, message_size,
                P, args.workers, args.move_range);

            all_distortion.push_back(total);
        }

        size_t best_idx = 0;
        for (size_t i = 1; i < all_distortion.size(); i++) {
            if (all_distortion[i] < all_distortion[best_idx]) best_idx = i;
        }
        const auto& best = top_patterns[best_idx];

        json sel;
        sel["best_rank"]             = best.value("rank", (int)best_idx);
        sel["best_param"]            = best.contains("param") ? best.at("param") : json(nullptr);
        sel["best_total_distortion"] = all_distortion[best_idx];
        sel["all_total_distortion"]  = all_distortion;
        sel["skipped"]               = false;
        selection["layers"][layer_name] = sel;

        fprintf(stdout, "  [greedy-select] %s: best_rank=%d (param=%s)  total_distortion=%.4f\n",
                layer_name.c_str(), (int)sel["best_rank"],
                best.contains("param") ? best.at("param").dump().c_str() : "null",
                all_distortion[best_idx]);
        fflush(stdout);
    }

    std::string out_dir = args.best_patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                          args.arch + "/" + args.qmode + "/" + bit_label + "/" + m_tag;
    fs::create_directories(out_dir);
    std::string out_path = out_dir + "/best_pattern_selection.json";
    {
        std::ofstream f(out_path);
        f << selection.dump(2);
    }
    fprintf(stdout, "[ecc_embed_cpp] [greedy-select] saved %s  (%zu layers)\n",
            out_path.c_str(), selection["layers"].size());
}

// ---- Phase 2: search3 embedding using each layer's winning pattern ----------
static void run_search3_with_best_pattern(
    const json&        top_manifest,
    const Args&        args,
    int                chunk_size,
    int                message_parity_size,
    int                message_size,
    const PMatrix&     P,
    Approach           approach,
    const std::string& bit_label,
    const std::string& m_tag)
{
    std::string sel_path = args.best_patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                           args.arch + "/" + args.qmode + "/" + bit_label + "/" + m_tag +
                           "/best_pattern_selection.json";
    if (!fs::exists(sel_path)) {
        fprintf(stderr,
            "[ecc_embed_cpp] ERROR: best_pattern_selection.json not found: %s\n"
            "Run --approach greedy with --top-patterns-dir/--best-patterns-dir first.\n",
            sel_path.c_str());
        exit(1);
    }
    json selection;
    {
        std::ifstream f(sel_path);
        f >> selection;
    }

    for (const auto& [layer_name, entry] : top_manifest.items()) {
        if (!entry.contains("top_patterns") || entry["top_patterns"].empty()) {
            fprintf(stdout, "  [skip] %s: no top_patterns entries\n", layer_name.c_str());
            continue;
        }
        const auto& top_patterns = entry["top_patterns"];

        int best_rank = top_patterns[0].value("rank", 0);
        if (selection.contains("layers") && selection["layers"].contains(layer_name)) {
            best_rank = selection["layers"][layer_name].value("best_rank", best_rank);
        }

        const json* rank_entry = &top_patterns[0];
        for (const auto& p : top_patterns) {
            if (p.value("rank", -1) == best_rank) { rank_entry = &p; break; }
        }

        fprintf(stdout, "  [search3] %s: using best_rank=%d (param=%s)\n",
                layer_name.c_str(), best_rank,
                rank_entry->contains("param") ? rank_entry->at("param").dump().c_str() : "null");
        fflush(stdout);

        std::vector<float> dense_sens;
        if (!args.no_sensitivity) {
            dense_sens = load_dense_sensitivity(layer_name, args, bit_label);
        }

        process_layer_topk(layer_name, *rank_entry, dense_sens, args, chunk_size,
                            message_parity_size, message_size, P, approach, bit_label, m_tag);
    }
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

    // Build BCH parity matrix once, shared by all workers.
    //
    // Prefer --parity-matrix file (exported by export_parity_matrix.py using
    // galois.BCH) so that C++ uses the SAME matrix as Python.  Fall back to
    // the C++ scratch implementation only when the file is not supplied — that
    // path produces a different (incompatible) matrix and should not be used
    // for production runs where Python accuracy is the reference.
    PMatrix P;
    if (approach != Approach::NO) {
        if (!args.parity_matrix.empty()) {
            if (!std::filesystem::exists(args.parity_matrix)) {
                fprintf(stderr, "[bch] ERROR: --parity-matrix file not found: %s\n",
                        args.parity_matrix.c_str());
                return 1;
            }
            fprintf(stdout, "[bch] Loading parity matrix from %s ...\n",
                    args.parity_matrix.c_str());
            fflush(stdout);
            P = pmatrix_from_npy(args.parity_matrix);
            fprintf(stdout, "[bch] Parity matrix loaded: %d × %d\n", P.k, P.r);
            fflush(stdout);
        } else {
            fprintf(stdout,
                "[bch] WARNING: --parity-matrix not supplied; computing BCH(%d, %d, t=%d) "
                "parity matrix from scratch.  This uses a different primitive polynomial "
                "than Python's galois.BCH and will produce lower accuracy than the Python "
                "reference.  Pass --parity-matrix (run export_parity_matrix.py first) for "
                "bit-exact Python compatibility.\n",
                args.codeword, message_size, args.t_value);
            fflush(stdout);
            auto P_raw = bch_parity_matrix(args.codeword, message_size, args.t_value);
            P = PMatrix(P_raw);
            fprintf(stdout, "[bch] Parity matrix built (scratch): %d × %d\n", P.k, P.r);
            fflush(stdout);
        }
    }

    // ---- Top-K pattern flow: greedy selection (Phase 1) / search3 with best pattern (Phase 2) ----
    if (!args.top_patterns_dir.empty() &&
        (approach == Approach::GREEDY || approach == Approach::SEARCH3)) {

        std::string top_manifest_path = args.top_patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                                        args.arch + "/" + args.qmode + "/" + bit_label +
                                        "/top_patterns_manifest.json";
        if (!fs::exists(top_manifest_path)) {
            fprintf(stdout, "[skip] No top-patterns manifest found: %s\n", top_manifest_path.c_str());
            return 0;
        }
        if (args.best_patterns_dir.empty()) {
            fprintf(stderr, "--top-patterns-dir requires --best-patterns-dir\n");
            return 1;
        }

        json top_manifest;
        {
            std::ifstream f(top_manifest_path);
            if (!f) { fprintf(stderr, "Cannot open manifest: %s\n", top_manifest_path.c_str()); return 1; }
            f >> top_manifest;
        }

        fprintf(stdout, "[ecc_embed_cpp] dataset=%s arch=%s qmode=%s bits=%s t=%d approach=%s "
                "codeword=%d chunk_size=%d msg_size=%d workers=%d\n",
                args.dataset.c_str(), args.arch.c_str(), args.qmode.c_str(), bit_label.c_str(),
                args.t_value, args.approach.c_str(), args.codeword,
                chunk_size, message_size, args.workers);
        fprintf(stdout, "[ecc_embed_cpp] top-patterns manifest: %s  (%zu layers)\n",
                top_manifest_path.c_str(), top_manifest.size());
        fflush(stdout);

        if (approach == Approach::GREEDY) {
            run_greedy_selection(top_manifest, args, chunk_size, message_parity_size,
                                  message_size, P, bit_label, m_tag);
        } else {
            run_search3_with_best_pattern(top_manifest, args, chunk_size, message_parity_size,
                                           message_size, P, approach, bit_label, m_tag);
        }

        fprintf(stdout, "[ecc_embed_cpp] Done. All layers processed.\n");
        return 0;
    }

    // ---- Legacy single-pattern flow: pattern_manifest.json (rank 0) ----
    std::string manifest_path = args.patterns_dir + "/" + ds_lower(args.dataset) + "/" +
                                args.arch + "/" + args.qmode + "/" + bit_label + "/pattern_manifest.json";

    if (!fs::exists(manifest_path)) {
        fprintf(stdout, "[skip] No manifest: %s\n", manifest_path.c_str());
        return 0;
    }

    json manifest;
    {
        std::ifstream f(manifest_path);
        if (!f) { fprintf(stderr, "Cannot open manifest: %s\n", manifest_path.c_str()); return 1; }
        f >> manifest;
    }

    fprintf(stdout, "[ecc_embed_cpp] dataset=%s arch=%s qmode=%s bits=%s t=%d approach=%s "
            "codeword=%d chunk_size=%d msg_size=%d workers=%d\n",
            args.dataset.c_str(), args.arch.c_str(), args.qmode.c_str(), bit_label.c_str(),
            args.t_value, args.approach.c_str(), args.codeword,
            chunk_size, message_size, args.workers);
    fprintf(stdout, "[ecc_embed_cpp] manifest=%s (%zu layers)\n",
            manifest_path.c_str(), manifest.size());
    fflush(stdout);

    // Process each layer
    for (const auto& [layer_name, entry] : manifest.items()) {
        process_layer(layer_name, entry, args, chunk_size, message_parity_size,
                      message_size, P, approach, bit_label, m_tag);
    }

    fprintf(stdout, "[ecc_embed_cpp] Done.\n");
    return 0;
}
