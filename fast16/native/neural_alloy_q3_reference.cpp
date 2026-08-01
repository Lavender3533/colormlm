#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> MAGIC = {'C', 'L', 'M', 'A', 'L', 'Y', '0', '1'};
constexpr std::uint32_t VERSION = 1;
constexpr std::uint32_t BITS = 3;
constexpr std::uint32_t GROUP_SIZE = 64;
constexpr std::uint32_t BLOCK_BYTES = 26;
constexpr std::uint64_t HEADER_BYTES = 72;
constexpr std::uint64_t DATA_ALIGNMENT = 4096;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
T read_scalar(std::istream & input) {
    T value {};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated Neural Alloy header");
    }
    return value;
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16;
    std::int32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | (static_cast<std::uint32_t>(exponent + 112) << 23) | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct Container {
    std::ifstream input;
    nlohmann::json manifest;
    std::uint64_t data_start = 0;

    explicit Container(const std::filesystem::path & path) : input(path, std::ios::binary) {
        if (!input) {
            throw std::runtime_error("cannot open Neural Alloy container");
        }

        std::array<char, 8> magic {};
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (magic != MAGIC) {
            throw std::runtime_error("invalid Neural Alloy magic");
        }
        const auto version = read_scalar<std::uint32_t>(input);
        const auto bits = read_scalar<std::uint32_t>(input);
        const auto group_size = read_scalar<std::uint32_t>(input);
        const auto block_bytes = read_scalar<std::uint32_t>(input);
        const auto tensor_count = read_scalar<std::uint64_t>(input);
        const auto manifest_bytes = read_scalar<std::uint64_t>(input);
        std::array<std::uint8_t, 32> manifest_hash {};
        input.read(reinterpret_cast<char *>(manifest_hash.data()), static_cast<std::streamsize>(manifest_hash.size()));

        if (version != VERSION || bits != BITS || group_size != GROUP_SIZE || block_bytes != BLOCK_BYTES) {
            throw std::runtime_error("unsupported Neural Alloy codec");
        }
        std::string json_text(static_cast<std::size_t>(manifest_bytes), '\0');
        input.read(json_text.data(), static_cast<std::streamsize>(json_text.size()));
        manifest = nlohmann::json::parse(json_text);
        if (manifest.at("tensors").size() != tensor_count) {
            throw std::runtime_error("manifest tensor count mismatch");
        }
        data_start = align_up(HEADER_BYTES + manifest_bytes, DATA_ALIGNMENT);
    }

    const nlohmann::json & tensor(const std::string & name) const {
        const auto & tensors = manifest.at("tensors");
        const auto found = std::find_if(tensors.begin(), tensors.end(), [&name](const nlohmann::json & item) {
            return item.at("name").get<std::string>() == name;
        });
        if (found == tensors.end()) {
            throw std::runtime_error("tensor not found in Neural Alloy container: " + name);
        }
        return *found;
    }

    std::vector<std::uint8_t> read_blocks(const nlohmann::json & tensor_info) {
        const auto offset = tensor_info.at("offset").get<std::uint64_t>();
        const auto group_count = tensor_info.at("group_count").get<std::uint64_t>();
        std::vector<std::uint8_t> blocks(static_cast<std::size_t>(group_count * BLOCK_BYTES));
        input.clear();
        input.seekg(static_cast<std::streamoff>(data_start + offset));
        input.read(reinterpret_cast<char *>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
        if (!input) {
            throw std::runtime_error("truncated q3 tensor data");
        }
        return blocks;
    }
};

std::int8_t decode_code(const std::uint8_t * block, std::uint32_t index) {
    const std::uint32_t chunk = index / 8;
    const std::uint32_t within = index % 8;
    const std::uint8_t * packed = block + 2 + chunk * 3;
    const std::uint32_t word = static_cast<std::uint32_t>(packed[0]) |
        (static_cast<std::uint32_t>(packed[1]) << 8) |
        (static_cast<std::uint32_t>(packed[2]) << 16);
    return static_cast<std::int8_t>((word >> (within * BITS)) & 7u) - 3;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: neural-alloy-q3-reference <container.nal> <2d-tensor-name>\n";
            return 2;
        }

        Container container(argv[1]);
        const auto & info = container.tensor(argv[2]);
        const auto shape = info.at("shape").get<std::vector<std::uint64_t>>();
        if (shape.size() != 2) {
            throw std::runtime_error("reference matvec requires a two-dimensional tensor");
        }
        const std::uint64_t input_size = shape[0];
        const std::uint64_t output_size = shape[1];
        const std::uint64_t logical_count = info.at("logical_count").get<std::uint64_t>();
        if (logical_count != input_size * output_size) {
            throw std::runtime_error("invalid matrix logical count");
        }

        const auto blocks = container.read_blocks(info);
        std::vector<float> input(static_cast<std::size_t>(input_size));
        std::vector<float> output(static_cast<std::size_t>(output_size), 0.0f);
        for (std::uint64_t i = 0; i < input_size; ++i) {
            input[static_cast<std::size_t>(i)] = std::sin(static_cast<float>(i + 1) * 0.013f);
        }

        std::uint64_t reserved_codes = 0;
        for (std::uint64_t linear = 0; linear < logical_count; ++linear) {
            const std::uint64_t group = linear / GROUP_SIZE;
            const std::uint32_t within = static_cast<std::uint32_t>(linear % GROUP_SIZE);
            const std::uint8_t * block = blocks.data() + group * BLOCK_BYTES;
            const std::uint16_t half = static_cast<std::uint16_t>(block[0]) |
                (static_cast<std::uint16_t>(block[1]) << 8);
            const auto q = decode_code(block, within);
            if (q == 4) {
                ++reserved_codes;
            }
            const std::uint64_t row = linear / input_size;
            const std::uint64_t column = linear % input_size;
            output[static_cast<std::size_t>(row)] += half_to_float(half) * static_cast<float>(q) * input[static_cast<std::size_t>(column)];
        }

        double l2 = 0.0;
        for (float value : output) {
            l2 += static_cast<double>(value) * value;
        }
        std::cout << "tensor=" << argv[2] << '\n';
        std::cout << "shape=" << input_size << 'x' << output_size << '\n';
        std::cout << "groups=" << info.at("group_count").get<std::uint64_t>() << '\n';
        std::cout << "reserved_codes=" << reserved_codes << '\n';
        std::cout << "output_l2=" << std::sqrt(l2) << '\n';
        std::cout << "output_first=" << output.front() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
