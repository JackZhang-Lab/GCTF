#include "mrc.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mrc {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "MRC mode 2 requires IEEE 754 binary32");

std::uint32_t load_word(const std::uint8_t* bytes, bool little) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value = (value << 8) | bytes[little ? 3 - i : i];
    }
    return value;
}

void store_word(std::uint8_t* bytes, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

std::int32_t header_int(const Header& header, std::size_t offset) {
    const auto word = load_word(header.bytes.data() + offset, header.little_endian());
    // Widen before interpreting the signed value to avoid an out-of-range cast.
    const auto signed_word = word <= 0x7fffffffU ? static_cast<std::int64_t>(word)
                                                  : static_cast<std::int64_t>(word) - 0x100000000LL;
    return static_cast<std::int32_t>(signed_word);
}

void store_float(std::uint8_t* bytes, float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    store_word(bytes, word);
}

std::size_t checked_multiply(std::size_t left, std::size_t right) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error("MRC size overflows size_t");
    }
    return left * right;
}

std::size_t voxel_count(std::int32_t nx, std::int32_t ny, std::int32_t nz) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("MRC dimensions must be positive");
    }
    return checked_multiply(checked_multiply(nx, ny), nz);
}

std::size_t bytes_per_voxel(Mode mode) {
    switch (mode) {
        case Mode::Int8: return 1;
        case Mode::Int16:
        case Mode::UInt16:
        case Mode::Float16: return 2;
        case Mode::Float32:
        case Mode::ComplexInt16: return 4;
        case Mode::ComplexFloat32: return 8;
    }
    throw std::runtime_error("Unsupported MRC mode");
}

struct Layout {
    std::size_t extended_size;
    std::size_t data_size;
};

Layout layout(const Header& header) {
    if (std::memcmp(header.bytes.data() + 208, "MAP ", 4) != 0) {
        throw std::runtime_error("Missing MRC MAP signature");
    }
    const auto extended_size = header.extended_header_size();
    if (extended_size < 0) {
        throw std::runtime_error("Negative MRC extended header size");
    }
    const auto count = voxel_count(header.nx(), header.ny(), header.nz());
    return {static_cast<std::size_t>(extended_size),
            checked_multiply(count, bytes_per_voxel(header.mode()))};
}

std::streamsize stream_size(std::size_t size) {
    if (size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("MRC block exceeds stream size limit");
    }
    return static_cast<std::streamsize>(size);
}

void read_block(std::ifstream& file, std::uint8_t* data, std::size_t size) {
    if (size && !file.read(reinterpret_cast<char*>(data), stream_size(size))) {
        throw std::runtime_error("Could not read complete MRC block");
    }
}

void write_block(std::ofstream& file, const std::uint8_t* data, std::size_t size) {
    if (size && !file.write(reinterpret_cast<const char*>(data), stream_size(size))) {
        throw std::runtime_error("Could not write complete MRC block");
    }
}

}  // namespace

constexpr std::size_t Header::size;

bool Header::little_endian() const {
    if (bytes[212] == 0x44 && (bytes[213] == 0x44 || bytes[213] == 0x41)) {
        return true;
    }
    if (bytes[212] == 0x11 && bytes[213] == 0x11) {
        return false;
    }
    throw std::runtime_error("Unrecognized MRC machine stamp");
}

std::int32_t Header::nx() const { return header_int(*this, 0); }
std::int32_t Header::ny() const { return header_int(*this, 4); }
std::int32_t Header::nz() const { return header_int(*this, 8); }
Mode Header::mode() const { return static_cast<Mode>(header_int(*this, 12)); }
std::int32_t Header::extended_header_size() const { return header_int(*this, 92); }

Volume read(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open MRC for reading: " + path);
    }

    const auto end = file.tellg();
    if (end < static_cast<std::streamoff>(Header::size)) {
        throw std::runtime_error("Truncated MRC header: " + path);
    }
    file.seekg(0);
    Volume volume;
    read_block(file, volume.header.bytes.data(), Header::size);
    const auto sizes = layout(volume.header);
    const auto remaining = static_cast<std::uintmax_t>(
        end - static_cast<std::streamoff>(Header::size));
    if (sizes.extended_size > remaining ||
        sizes.data_size > remaining - sizes.extended_size) {
        throw std::runtime_error("Truncated MRC extended header or voxel data: " + path);
    }

    volume.extended_header.resize(sizes.extended_size);
    volume.data.resize(sizes.data_size);
    read_block(file, volume.extended_header.data(), volume.extended_header.size());
    read_block(file, volume.data.data(), volume.data.size());
    return volume;
}

void write(const std::string& path, const Volume& volume) {
    const auto sizes = layout(volume.header);
    if (volume.extended_header.size() != sizes.extended_size ||
        volume.data.size() != sizes.data_size) {
        throw std::runtime_error("MRC block size does not match header");
    }
    stream_size(sizes.extended_size);
    stream_size(sizes.data_size);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Could not open MRC for writing: " + path);
    }
    write_block(file, volume.header.bytes.data(), Header::size);
    write_block(file, volume.extended_header.data(), sizes.extended_size);
    write_block(file, volume.data.data(), sizes.data_size);
    file.close();
    if (!file) {
        throw std::runtime_error("Could not finish writing MRC: " + path);
    }
}

Volume make_float32(std::int32_t nx, std::int32_t ny, std::int32_t nz,
                    const std::vector<float>& voxels, float voxel_size) {
    const auto count = voxel_count(nx, ny, nz);
    if (voxels.size() != count) {
        throw std::runtime_error("MRC voxel count does not match dimensions");
    }
    if (!std::isfinite(voxel_size) || voxel_size <= 0 ||
        static_cast<double>(voxel_size) * std::max({nx, ny, nz}) >
            std::numeric_limits<float>::max()) {
        throw std::runtime_error("Invalid MRC voxel size");
    }

    Volume volume;
    auto* header = volume.header.bytes.data();
    const std::int32_t dimensions[] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        store_word(header + 4 * axis, dimensions[axis]);  // NX, NY, NZ
        store_word(header + 28 + 4 * axis, dimensions[axis]);  // MX, MY, MZ
        store_float(header + 40 + 4 * axis, voxel_size * dimensions[axis]);  // CELLA
        store_float(header + 52 + 4 * axis, 90.0f);  // CELLB
        store_word(header + 64 + 4 * axis, axis + 1);  // MAPC, MAPR, MAPS
    }
    store_word(header + 12, static_cast<std::uint32_t>(Mode::Float32));
    store_word(header + 88, nz > 1 ? 1 : 0);  // ISPG
    store_word(header + 108, 20141);  // NVERSION
    std::memcpy(header + 208, "MAP ", 4);
    header[212] = 0x44;
    header[213] = 0x44;

    volume.data.resize(checked_multiply(count, sizeof(float)));
    double mean = 0.0;
    double sum_squares = 0.0;
    float minimum = voxels.front();
    float maximum = voxels.front();
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = voxels[i];
        store_float(volume.data.data() + sizeof(float) * i, value);
        if (std::isfinite(value)) {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            const double delta = value - mean;
            mean += delta / static_cast<double>(i + 1);
            sum_squares += delta * (value - mean);
        } else {
            finite = false;
        }
    }
    // MRC2014 sentinel values mark unavailable statistics.
    store_float(header + 76, finite ? minimum : 0.0f);
    store_float(header + 80, finite ? maximum : -1.0f);
    store_float(header + 84, finite ? static_cast<float>(mean) : -2.0f);
    store_float(header + 216, finite ? static_cast<float>(std::sqrt(sum_squares / count)) : -1.0f);
    return volume;
}

std::vector<float> float32_voxels(const Volume& volume) {
    const auto sizes = layout(volume.header);
    if (volume.header.mode() != Mode::Float32 || volume.data.size() != sizes.data_size) {
        throw std::runtime_error("Expected complete MRC mode-2 data");
    }

    const bool little = volume.header.little_endian();
    std::vector<float> voxels(sizes.data_size / sizeof(float));
    for (std::size_t i = 0; i < voxels.size(); ++i) {
        const auto word = load_word(volume.data.data() + sizeof(float) * i, little);
        std::memcpy(&voxels[i], &word, sizeof(word));
    }
    return voxels;
}

}  // namespace mrc
