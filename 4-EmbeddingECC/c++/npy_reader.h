#pragma once
// npy_reader.h — minimal numpy .npy file reader for int8, int64, float32 arrays.
// Format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyInfo {
    std::string dtype;          // e.g. "<i1", "<i8", "<f4"
    bool        c_order;        // true = C (row-major), false = Fortran
    std::vector<size_t> shape;
    size_t      num_elements;
    size_t      data_offset;    // byte offset to start of data in file
};

inline NpyInfo npy_read_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    // Magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);

    uint8_t major = f.get(), minor = f.get();
    (void)minor;

    // Header length (2 bytes for v1, 4 bytes for v2)
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16; f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    size_t data_offset = 6 + 2 + (major == 1 ? 2u : 4u) + hlen;

    // Read header dict string
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);

    // Parse dtype
    NpyInfo info;
    info.data_offset = data_offset;

    auto extract = [&](const std::string& key) -> std::string {
        auto pos = hdr.find(key);
        if (pos == std::string::npos) throw std::runtime_error("Key not found: " + key);
        pos = hdr.find('\'', pos + key.size());
        auto end = hdr.find('\'', pos + 1);
        return hdr.substr(pos + 1, end - pos - 1);
    };

    info.dtype  = extract("'descr':");
    {
        auto fp = hdr.find("'fortran_order':");
        if (fp == std::string::npos)
            throw std::runtime_error("'fortran_order' not found in npy header: " + path);
        size_t vp = fp + 16;  // length of "'fortran_order':"
        while (vp < hdr.size() && (hdr[vp] == ' ' || hdr[vp] == '\t')) ++vp;
        // fortran_order: True → NOT C-order; fortran_order: False → C-order
        info.c_order = !(hdr.size() >= vp + 4 && hdr.compare(vp, 4, "True") == 0);
    }

    // Parse shape tuple
    auto sp = hdr.find("'shape':");
    if (sp == std::string::npos) throw std::runtime_error("'shape' not found");
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp);
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    std::istringstream ss(shape_str);
    std::string tok;
    info.num_elements = 1;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty() || tok.find_first_not_of(" \t\r\n0123456789") != std::string::npos)
            continue;
        size_t dim = std::stoull(tok);
        if (dim > 0) { info.shape.push_back(dim); info.num_elements *= dim; }
    }

    return info;
}

// ---- Typed loaders ----

inline std::vector<int8_t> npy_load_int8(const std::string& path) {
    NpyInfo info = npy_read_header(path);
    std::ifstream f(path, std::ios::binary);
    f.seekg(info.data_offset);
    std::vector<int8_t> buf(info.num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), info.num_elements);
    return buf;
}

inline std::vector<uint8_t> npy_load_uint8(const std::string& path) {
    NpyInfo info = npy_read_header(path);
    std::ifstream f(path, std::ios::binary);
    f.seekg(info.data_offset);
    std::vector<uint8_t> buf(info.num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), info.num_elements);
    return buf;
}

inline std::vector<int64_t> npy_load_int64(const std::string& path) {
    NpyInfo info = npy_read_header(path);
    std::ifstream f(path, std::ios::binary);
    f.seekg(info.data_offset);
    std::vector<int64_t> buf(info.num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), info.num_elements * 8);
    return buf;
}

inline std::vector<float> npy_load_float32(const std::string& path) {
    NpyInfo info = npy_read_header(path);
    std::ifstream f(path, std::ios::binary);
    f.seekg(info.data_offset);
    std::vector<float> buf(info.num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), info.num_elements * 4);
    return buf;
}

// Generic loader — returns raw bytes and dtype string for caller to interpret
inline std::pair<std::vector<uint8_t>, NpyInfo> npy_load_raw(const std::string& path) {
    NpyInfo info = npy_read_header(path);
    std::ifstream f(path, std::ios::binary);
    f.seekg(info.data_offset);
    // Determine item size
    size_t item_bytes = 1;
    if (info.dtype.find("i8") != std::string::npos || info.dtype.find("u8") != std::string::npos
        || info.dtype.find("f8") != std::string::npos) item_bytes = 8;
    else if (info.dtype.find("i4") != std::string::npos || info.dtype.find("u4") != std::string::npos
             || info.dtype.find("f4") != std::string::npos) item_bytes = 4;
    else if (info.dtype.find("i2") != std::string::npos || info.dtype.find("u2") != std::string::npos
             || info.dtype.find("f2") != std::string::npos) item_bytes = 2;
    size_t total_bytes = info.num_elements * item_bytes;
    std::vector<uint8_t> buf(total_bytes);
    f.read(reinterpret_cast<char*>(buf.data()), total_bytes);
    return {buf, info};
}
