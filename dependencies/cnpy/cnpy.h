#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace cnpy {

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<double> data;
    bool fortran_order;

    template<typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(data.data());
    }
};

inline void parse_npy_header(std::ifstream& fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    // Check magic string: exactly "\x93NUMPY"
    char magic[6];
    fp.read(magic, sizeof(magic));
    
    if (!fp.good() || magic[0] != '\x93' ||
        magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' ||
        magic[5] != 'Y') {
        throw std::runtime_error("parse_npy_header: Invalid magic string");
    }

    // Read version numbers
    uint8_t major, minor;
    fp.read(reinterpret_cast<char*>(&major), sizeof(uint8_t));
    fp.read(reinterpret_cast<char*>(&minor), sizeof(uint8_t));
    
    if (major != 1 || minor != 0) {
        throw std::runtime_error("parse_npy_header: Only version 1.0 is supported");
    }

    // Read header length
    uint16_t header_len;
    fp.read(reinterpret_cast<char*>(&header_len), sizeof(uint16_t));
    
    // Read the header
    std::string header(header_len, ' ');
    fp.read(&header[0], header_len);

    // // Debug output
    // std::cout << "NPY Header length: " << header_len << std::endl;
    // std::cout << "NPY Header: " << header << std::endl;

    // Find the start and end of the descr field
    size_t descr_start = header.find("'descr'");
    if (descr_start == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Could not find 'descr' field");
    }
    
    // Find the first quote after 'descr': 
    size_t descr_val_start = header.find("'", descr_start + 7); // Skip past 'descr':
    if (descr_val_start == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Could not find start of descr value");
    }
    descr_val_start++; // Move past the quote
    
    // Find the closing quote
    size_t descr_val_end = header.find("'", descr_val_start);
    if (descr_val_end == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Could not find end of descr value");
    }
    
    // Extract the descriptor
    std::string descr = header.substr(descr_val_start, descr_val_end - descr_val_start);
    std::cout << "Data type descriptor: '" << descr << "'" << std::endl;
    
    // For '<f8' format (little-endian double)
    if (descr == "<f8") {
        word_size = 8;  // double precision (8 bytes)
    } else {
        throw std::runtime_error("parse_npy_header: Only <f8 (little-endian double) format is supported");
    }

    // Parse fortran order
    size_t order_start = header.find("'fortran_order'");
    if (order_start == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Could not find 'fortran_order' field");
    }
    size_t order_val_start = header.find(":", order_start) + 1;
    while (order_val_start < header.length() && std::isspace(header[order_val_start])) order_val_start++;
    fortran_order = (header.substr(order_val_start, 4) == "True");

    // Parse shape
    size_t shape_start = header.find("'shape'");
    if (shape_start == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Could not find 'shape' field");
    }
    
    size_t shape_val_start = header.find("(", shape_start) + 1;
    size_t shape_val_end = header.find(")", shape_val_start);
    if (shape_val_start == std::string::npos || shape_val_end == std::string::npos) {
        throw std::runtime_error("parse_npy_header: Invalid shape format");
    }
    
    std::string shape_str = header.substr(shape_val_start, shape_val_end - shape_val_start);
    shape.clear();
    
    // Split shape string by commas
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Remove whitespace
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            shape.push_back(std::stoull(item));
        }
    }
}

inline NpyArray npy_load(const std::string& fname) {
    std::ifstream fp(fname, std::ios::binary);
    if(!fp) {
        throw std::runtime_error("npy_load: Unable to open file " + fname);
    }

    NpyArray arr;
    size_t word_size;
    parse_npy_header(fp, word_size, arr.shape, arr.fortran_order);

    if (arr.shape.size() != 2 || arr.shape[0] != 3 || arr.shape[1] != 3) {
        throw std::runtime_error("npy_load: Expected 3x3 matrix");
    }

    size_t total_size = arr.shape[0] * arr.shape[1];
    arr.data.resize(total_size);

    // Read the data directly as doubles
    fp.read(reinterpret_cast<char*>(arr.data.data()), total_size * sizeof(double));

    if (!fp.good()) {
        throw std::runtime_error("npy_load: Error reading data");
    }

    // Print the loaded matrix for debugging
    std::cout << "Successfully loaded matrix:\n";
    for (size_t i = 0; i < arr.shape[0]; i++) {
        for (size_t j = 0; j < arr.shape[1]; j++) {
            std::cout << arr.data[i * arr.shape[1] + j] << " ";
        }
        std::cout << "\n";
    }

    return arr;
}

} // namespace cnpy 