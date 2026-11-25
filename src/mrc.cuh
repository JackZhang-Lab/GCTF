#ifndef GCTF_MRC_CUH
#define GCTF_MRC_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mrc {

enum class Mode : std::int32_t {
    Int8 = 0,
    Int16 = 1,
    Float32 = 2,
    ComplexInt16 = 3,
    ComplexFloat32 = 4,
    UInt16 = 6,
    Float16 = 12
};

// MRC2014 disk header. Keeping the original bytes preserves metadata on rewrite.
struct Header {
    static constexpr std::size_t size = 1024;
    std::array<std::uint8_t, size> bytes{};

    bool little_endian() const;
    std::int32_t nx() const;
    std::int32_t ny() const;
    std::int32_t nz() const;
    Mode mode() const;
    std::int32_t extended_header_size() const;
};

struct Volume {
    Header header;
    std::vector<std::uint8_t> extended_header;
    // Raw voxels in file byte order, with columns changing fastest.
    std::vector<std::uint8_t> data;
};

// Host-side I/O. Throws std::runtime_error for malformed files or I/O errors.
Volume read(const std::string& path);
void write(const std::string& path, const Volume& volume);

// Construct a little-endian mode-2 volume in conventional X/Y/Z axis order.
Volume make_float32(std::int32_t nx, std::int32_t ny, std::int32_t nz,
                    const std::vector<float>& voxels, float voxel_size = 1.0f);
std::vector<float> float32_voxels(const Volume& volume);

}  // namespace mrc

#endif
