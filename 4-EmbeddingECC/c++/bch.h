#pragma once
// bch.h — BCH parity matrix computation from scratch using GF(2^m) arithmetic.
// No external galois library needed.
//
// Reference: Lin & Costello "Error Control Coding", 2nd ed., Chapters 6-7.
//
// Supports n = 63 (GF(2^6)), 127 (GF(2^7)), 255 (GF(2^8)).
// Returns the parity submatrix P of the systematic generator G = [I_k | P].
//   - P has dimensions k × (n-k)
//   - GF(2) elements packed in uint8_t

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// GF(2^m) field arithmetic
// ============================================================
struct GFField {
    int m;           // field exponent
    int n;           // n = 2^m - 1
    uint64_t prim;   // primitive polynomial (bit i = coeff of x^i)
    std::vector<uint64_t> exp_table;  // exp_table[i] = alpha^i  (i = 0..2n-1)
    std::vector<int>      log_table;  // log_table[a] = log_alpha(a) (a = 1..n)

    GFField(int m_, uint64_t prim_poly) : m(m_), n((1 << m_) - 1), prim(prim_poly),
        exp_table(2 * (1 << m_) + 1), log_table(1 << m_, -1)
    {
        uint64_t a = 1;
        for (int i = 0; i <= n; i++) {
            exp_table[i] = a;
            log_table[a] = i % n;
            // multiply by alpha: shift left, reduce by primitive poly if overflow
            a <<= 1;
            if (a & (1ULL << m)) a ^= prim;
        }
        // wrap-around entries for convenience
        for (int i = n + 1; i < 2 * n + 2; i++)
            exp_table[i] = exp_table[i - n];
    }

    // Multiply two GF(2^m) elements
    uint64_t mul(uint64_t a, uint64_t b) const {
        if (a == 0 || b == 0) return 0;
        return exp_table[(log_table[a] + log_table[b]) % n];
    }
};

// ============================================================
// GF(2)[x] polynomial arithmetic (bit-packed: bit i = coeff of x^i)
// ============================================================

// Degree of polynomial (position of highest set bit), -1 for 0
static inline int poly_deg(uint64_t p) {
    return p == 0 ? -1 : (63 - __builtin_clzll(p));
}

// Multiply two GF(2)[x] polynomials (carry-less multiply)
static inline uint64_t poly_mul_gf2(uint64_t a, uint64_t b) {
    uint64_t result = 0;
    while (b) {
        if (b & 1) result ^= a;
        a <<= 1;
        b >>= 1;
    }
    return result;
}

// a(x) mod b(x) in GF(2)[x]
static inline uint64_t poly_mod_gf2(uint64_t a, uint64_t b) {
    int db = poly_deg(b);
    while (true) {
        if (a == 0) return 0;
        int da = poly_deg(a);
        if (da < db) return a;
        a ^= (b << (da - db));
    }
}

// ============================================================
// Minimal polynomial of alpha^i over GF(2)
// ============================================================
// Returns min poly as bit-packed GF(2)[x] polynomial.
// The cyclotomic coset C_i = {i, 2i, 4i, ...} mod n.
// min_poly(alpha^i) = product_{j in C_i} (x - alpha^j) = product (x + alpha^j) [GF(2)]
//
// We track intermediate polynomial with GF(2^m) coefficients, then confirm they are in GF(2).
static uint64_t minimal_poly(const GFField& F, int i) {
    // Compute cyclotomic coset
    std::vector<int> coset;
    {
        std::set<int> seen;
        int j = i % F.n;
        while (seen.find(j) == seen.end()) {
            seen.insert(j);
            coset.push_back(j);
            j = (2 * j) % F.n;
        }
    }

    // Build min poly by multiplying (x + alpha^j) for each j in coset.
    // Poly coefficients in GF(2^m), stored as vector (index = degree).
    std::vector<uint64_t> p = {1};  // start with 1
    for (int j : coset) {
        uint64_t alpha_j = F.exp_table[j];
        // Multiply p by (x + alpha_j): new_p[k] = p[k-1] XOR alpha_j * p[k]
        std::vector<uint64_t> q(p.size() + 1, 0);
        for (size_t k = 0; k < p.size(); k++) {
            q[k + 1] ^= p[k];                   // x * p(x)
            q[k]     ^= F.mul(alpha_j, p[k]);   // alpha_j * p(x)
        }
        p = q;
    }

    // Convert to GF(2)[x]: each coefficient must be 0 or 1
    uint64_t result = 0;
    for (size_t k = 0; k < p.size(); k++) {
        assert(p[k] == 0 || p[k] == 1);
        result |= (p[k] << k);
    }
    return result;
}

// ============================================================
// Compute BCH generator polynomial g(x)
// g(x) = LCM of minimal polynomials of alpha^1, alpha^2, ..., alpha^(2t)
// ============================================================
static uint64_t bch_generator_poly(const GFField& F, int t) {
    uint64_t g = 1;
    std::set<int> included;

    for (int i = 1; i <= 2 * t; i++) {
        if (included.count(i % F.n)) continue;

        // Compute cyclotomic coset of i and mark all members
        std::set<int> coset;
        {
            int j = i % F.n;
            while (coset.find(j) == coset.end()) {
                coset.insert(j);
                j = (2 * j) % F.n;
            }
        }
        bool any_needed = false;
        for (int j : coset) if (j >= 1 && j <= 2 * t && !included.count(j)) { any_needed = true; break; }
        if (!any_needed) { for (int j : coset) included.insert(j); continue; }

        uint64_t mp = minimal_poly(F, i % F.n);
        // Multiply g by mp (in GF(2)[x])
        g = poly_mul_gf2(g, mp);
        for (int j : coset) included.insert(j);
    }
    return g;
}

// ============================================================
// Parity submatrix P of systematic BCH generator G = [I_k | P]
//
// Algorithm:
//   1. Build k rows: x^i * g(x) for i = 0..k-1  (non-systematic generator)
//   2. Gaussian elimination over GF(2) to get identity in columns 0..k-1
//   3. Extract the parity columns k..n-1 as P
// ============================================================
inline std::vector<std::vector<uint8_t>> bch_parity_matrix(int n, int k, int t) {
    // Map n to (m, primitive_polynomial)
    int m;
    uint64_t prim;
    if      (n == 63)  { m = 6; prim = 0x43; }   // x^6 + x + 1
    else if (n == 127) { m = 7; prim = 0x89; }   // x^7 + x^3 + 1
    else if (n == 255) { m = 8; prim = 0x11D; }  // x^8 + x^4 + x^3 + x^2 + 1
    else throw std::invalid_argument("Unsupported BCH n (must be 63, 127, or 255)");

    GFField F(m, prim);
    uint64_t g = bch_generator_poly(F, t);

    // Verify g has degree n-k
    if (poly_deg(g) != n - k)
        throw std::runtime_error("BCH generator polynomial has unexpected degree");

    // Build k rows: row[i] = g(x) * x^i
    // For n=63, each row fits in a 63-bit integer (uint64_t).
    // For n=127, we need 127 bits → use two uint64_t words.
    // For simplicity, use vector<uint64_t> with ceil(n/64) words.
    int words = (n + 63) / 64;
    auto make_row = [&](uint64_t low, int shift) -> std::vector<uint64_t> {
        std::vector<uint64_t> v(words, 0);
        int w = shift / 64, b = shift % 64;
        if (b == 0) {
            v[w] = low;
        } else {
            v[w]     |= (low << b);
            if (w + 1 < words) v[w + 1] = (low >> (64 - b));
        }
        return v;
    };

    auto xor_row = [&](std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        for (int w = 0; w < words; w++) a[w] ^= b[w];
    };
    auto get_bit = [&](const std::vector<uint64_t>& v, int col) -> int {
        return (v[col / 64] >> (col % 64)) & 1;
    };

    std::vector<std::vector<uint64_t>> rows(k);
    for (int i = 0; i < k; i++)
        rows[i] = make_row(g, i);

    // Gaussian elimination: pivot at column c for row c
    for (int c = 0; c < k; c++) {
        // Find pivot row
        int pivot = -1;
        for (int r = c; r < k; r++) {
            if (get_bit(rows[r], c)) { pivot = r; break; }
        }
        if (pivot < 0) throw std::runtime_error("BCH matrix is singular at col " + std::to_string(c));
        std::swap(rows[c], rows[pivot]);
        // Eliminate column c from all other rows
        for (int r = 0; r < k; r++) {
            if (r != c && get_bit(rows[r], c))
                xor_row(rows[r], rows[c]);
        }
    }

    // Extract P: columns k..n-1 of each row
    int r = n - k;
    std::vector<std::vector<uint8_t>> P(k, std::vector<uint8_t>(r, 0));
    for (int i = 0; i < k; i++)
        for (int j = 0; j < r; j++)
            P[i][j] = static_cast<uint8_t>(get_bit(rows[i], k + j));

    return P;
}

// ============================================================
// Packed BCH parity matrix for fast GF(2) multiply
// Packs each column of P into a bitset for __builtin_parityll
// ============================================================
struct PMatrix {
    int k, r;
    // col_words[j] = bit-packed column j of P
    // word i bit b = P[i*64+b][j]
    std::vector<std::vector<uint64_t>> col_words;
    int words_per_col;

    PMatrix() : k(0), r(0), words_per_col(0) {}

    explicit PMatrix(const std::vector<std::vector<uint8_t>>& P_raw) {
        k = (int)P_raw.size();
        r = k > 0 ? (int)P_raw[0].size() : 0;
        words_per_col = (k + 63) / 64;
        col_words.assign(r, std::vector<uint64_t>(words_per_col, 0));
        for (int i = 0; i < k; i++)
            for (int j = 0; j < r; j++)
                if (P_raw[i][j])
                    col_words[j][i / 64] |= (1ULL << (i % 64));
    }

    // Compute parity = m @ P  mod 2
    // m_bits: k bits packed into vector of uint64_t words
    std::vector<uint8_t> compute_parity(const std::vector<uint64_t>& m_bits) const {
        std::vector<uint8_t> parity(r, 0);
        for (int j = 0; j < r; j++) {
            int p = 0;
            for (int w = 0; w < words_per_col; w++)
                p ^= __builtin_parityll(m_bits[w] & col_words[j][w]);
            parity[j] = p & 1;
        }
        return parity;
    }
};

// Build the NANDT_TO_K lookup (mirrors Python NANDT_TO_K dict)
inline int bch_message_size(int n, int t) {
    // Comprehensive table for n=63, 127, 255
    static const int table63[]  = {0, 57, 51, 45, 39, 36, 30, 24, 18};
    static const int table127[] = {0,120,113,106, 99, 92, 85, 78, 71, 71, 64, 57, 50, 50};
    static const int table255[] = {0,  0,  0,  0,223,  0,  0,  0,191,187,179,171,163,155,
                                      147,139,131,  0,131,  0,  0,  0,  0,  0,  0, 91};
    if (n == 63  && t >= 1 && t <= 8)  return table63[t];
    if (n == 127 && t >= 1 && t <= 13) return table127[t];
    if (n == 255 && t < (int)(sizeof(table255)/sizeof(int))) return table255[t];
    throw std::invalid_argument("Unsupported (n,t) for BCH code");
}
