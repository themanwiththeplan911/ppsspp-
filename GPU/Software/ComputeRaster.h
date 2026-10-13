// Copyright (c) 2012- PPSSPP Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

// Official git repository and contact information can be found at
// https://github.com/hrydgard/ppsspp and http://www.ppsspp.org/.

#pragma once

// The software renderer's rasterization on the GPU, in compute shaders (like paraLLEl-GS does it for the PS2).
// The CPU still transforms, clips and bins (TransformUnit, BinManager). Each primitive's setup (the same fixed
// point edges and planes Rasterizer.cpp computes) is packed into words here, and a compute shader draws it a
// pixel per invocation, with all of the GE's pixel pipeline, into a copy of PSP memory on the GPU.
//
// Layout of what's packed: everything is 32-bit words, and the shader (ComputeRasterVulkan.cpp) gets the
// offsets below as #defines, so this is the one place they're declared.

#include <cstdint>
#include <vector>

class StringWriter;
struct VertexData;
struct BinCoords;
namespace Rasterizer {
struct RasterizerState;
}

namespace SoftCompute {

// Where things are in the GPU's memory buffer, in bytes. VRAM is first, so a VRAM address masked with
// 0x001FFFFF is its offset; main RAM follows it.
constexpr uint32_t MEM_VRAM_OFFSET = 0;
constexpr uint32_t MEM_VRAM_SIZE = 0x00200000;
constexpr uint32_t MEM_RAM_OFFSET = MEM_VRAM_SIZE;

enum class PrimType : uint32_t {
	TRIANGLE = 0,
	RECT = 1,
	SPRITE = 2,
	CLEAR_RECT = 3,
	FRAGMENTS = 4,
};

// A primitive's header: the type and flags, its state, and the pixels it can touch (drawing coordinates,
// inclusive, within the scissor). The setup words for its type follow it.
constexpr int PRIM_TYPE_FLAGS = 0;  // Type in bits 0-3, PrimFlag bits above.
constexpr int PRIM_STATE = 1;
constexpr int PRIM_XY1 = 2;  // x | y << 16
constexpr int PRIM_XY2 = 3;
constexpr int PRIM_HEADER_WORDS = 4;

enum PrimFlag : uint32_t {
	PRIMF_FLATZ = 1 << 4,
	PRIMF_FLAT_C0 = 1 << 5,
	PRIMF_FLAT_C1 = 1 << 6,
	PRIMF_NOFOG = 1 << 7,
	PRIMF_UV = 1 << 8,
	// The edge functions fit in 32 bits over the primitive's pixels.
	PRIMF_EDGES32 = 1 << 9,
	// All the planes' sums fit in 32 bits over the primitive's pixels (and spans around them).
	PRIMF_PLANES32 = 1 << 10,
	// The mip level's q is picked per span of four pixels (Rasterizer's LodQFromPlanes).
	PRIMF_LODQ = 1 << 11,
	// The q plane is anchored on the right, so spans are walked right to left.
	PRIMF_Q_RTL = 1 << 12,
	// Through mode texture coordinates, which are scaled by the texture size.
	PRIMF_THROUGH_UV = 1 << 13,
	// The level of detail uses the pixel's q (LOD_BASE - log2(q)), rather than LOD_BASE alone.
	PRIMF_LOD_USES_Q = 1 << 14,
	// Sprite: textured.
	PRIMF_TEXTURED = 1 << 15,
};

// TRIANGLE setup, after the header.
constexpr int TRI_EDGES = 0;  // 3 x { a, b, c_lo, c_hi }: a x + b y + c >= 0 at pixel centers.
constexpr int TRI_SNAP = 12;  // Snap edge (-1 for none) in bits 0-7 (signed), snapLeft in bit 8.
constexpr int TRI_Z = 13;  // Flat z.
constexpr int TRI_C0 = 14;  // Flat colors.
constexpr int TRI_C1 = 15;
constexpr int TRI_LOD_BASE = 16;
constexpr int TRI_EXP_S = 17;
constexpr int TRI_EXP_T = 18;
constexpr int TRI_EXP_Q = 19;
// Then the planes it needs, in this order: depth (unless FLATZ), fog (unless NOFOG), 4 color0 (unless FLAT_C0),
// 3 color1 (unless FLAT_C1), q, s, t (if UV). Each is { base_lo, base_hi, kx_lo, kx_hi, ky_lo, ky_hi }.
constexpr int TRI_PLANES = 20;
constexpr int PLANE_WORDS = 6;

// RECT setup: drawn like DrawRectangle, with pixel centers at 16 k + 8 for the planes.
constexpr int RECT_Z = 0;
constexpr int RECT_FOG = 1;
constexpr int RECT_C0 = 2;
constexpr int RECT_C1 = 3;
constexpr int RECT_LOD_BASE = 4;
constexpr int RECT_EXP_S = 5;
constexpr int RECT_EXP_T = 6;
constexpr int RECT_EXP_Q = 7;
constexpr int RECT_PLANES = 8;  // q, s, t if UV.

// SPRITE setup: DrawSprite's 1:1 texel walk from the first pixel (the header's XY1).
constexpr int SPRITE_S = 0;
constexpr int SPRITE_T = 1;
constexpr int SPRITE_DS = 2;
constexpr int SPRITE_DT = 3;
constexpr int SPRITE_Z = 4;
constexpr int SPRITE_C0 = 5;
constexpr int SPRITE_WORDS = 6;

// CLEAR_RECT setup.
constexpr int CLEAR_Z = 0;
constexpr int CLEAR_COLOR = 1;
constexpr int CLEAR_WORDS = 2;

// FRAGMENTS: pixels worked out on the CPU (lines and points), drawn in order.
constexpr int FRAG_COUNT = 0;
constexpr int FRAG_FIRST = 1;
// Each fragment.
constexpr int FRAGW_XY = 0;  // x | y << 16
constexpr int FRAGW_Z = 1;
constexpr int FRAGW_FOG_FLAGS = 2;  // fog | textured << 8 | bilinear << 9 | level << 16 | frac << 24
constexpr int FRAGW_C0 = 3;
constexpr int FRAGW_C1 = 4;
constexpr int FRAGW_S = 5;  // float bits
constexpr int FRAGW_T = 6;
constexpr int FRAGW_EARLY_RANGE = 7;  // 1 if the depth range test fails (done before the depth test).
constexpr int FRAG_WORDS = 8;

// A state, as the shader reads it.
constexpr int ST_PIXEL0 = 0;  // PixelFuncID's key, packed as in PackState().
constexpr int ST_PIXEL1 = 1;
constexpr int ST_REFS = 2;  // alphaTestRef | stencilTestRef << 8 | alphaTestMask << 16 | stencilTestMask << 24
constexpr int ST_MISC = 3;  // stencilRef | logicOp << 8 | colorTestFunc << 16
constexpr int ST_WRITEMASK = 4;
constexpr int ST_DITHER = 5;  // 4 words, 16 signed bytes.
constexpr int ST_FOGCOLOR = 9;
constexpr int ST_MINZ = 10;
constexpr int ST_MAXZ = 11;
constexpr int ST_STRIDES = 12;  // fb | depth << 16
constexpr int ST_CTEST_MASK = 13;
constexpr int ST_CTEST_REF = 14;
constexpr int ST_FIXA = 15;
constexpr int ST_FIXB = 16;
constexpr int ST_FB_ADDR = 17;  // VRAM offsets.
constexpr int ST_Z_BASE = 18;
constexpr int ST_Z_ROTMASK = 19;
constexpr int ST_Z_ROTSHIFT = 20;
constexpr int ST_Z_XOR = 21;
constexpr int ST_SAMPLER = 22;
constexpr int ST_CLUTFORMAT = 23;
constexpr int ST_TEXENV = 24;
constexpr int ST_TEXLEVEL = 25;  // texLevelOffset (signed byte) | maxTexLevel << 8 | width0Shift << 16 | height0Shift << 24
constexpr int ST_CLUT_ADDR = 26;  // Byte offset in the memory buffer.
constexpr int ST_LEVELS = 27;  // 8 x { w | h << 16, bufw, byte offset, flags }
constexpr int ST_WORDS = 64;

// ST_PIXEL0 bits.
enum PixelBits0 : uint32_t {
	PX0_CLEAR = 1 << 0,
	PX0_COLORTEST = 1 << 1,  // ColorClear in clear mode.
	PX0_STENCILTEST = 1 << 2,  // StencilClear.
	PX0_DEPTHWRITE = 1 << 3,  // DepthClear.
	PX0_DEPTHRANGE = 1 << 4,
	PX0_ALPHABLEND = 1 << 5,
	PX0_ALPHATESTMASK = 1 << 6,
	PX0_STENCILTESTMASK = 1 << 7,
	PX0_DITHER = 1 << 8,
	PX0_LOGICOP = 1 << 9,
	PX0_FOG = 1 << 10,
	PX0_WRITEMASK = 1 << 11,
	PX0_EARLYZ = 1 << 12,
	PX0_FBFORMAT_SHIFT = 16,  // 2 bits
	PX0_ALPHATEST_SHIFT = 20,  // 3 bits each
	PX0_DEPTHTEST_SHIFT = 23,
	PX0_STENCILTEST_SHIFT = 26,
};

// ST_PIXEL1: blend equation and factors, stencil ops.
enum PixelBits1 : uint32_t {
	PX1_BLENDEQ_SHIFT = 0,  // 3 bits
	PX1_BLENDSRC_SHIFT = 4,  // 4 bits
	PX1_BLENDDST_SHIFT = 8,  // 4 bits
	PX1_SFAIL_SHIFT = 12,  // 3 bits each
	PX1_ZFAIL_SHIFT = 16,
	PX1_ZPASS_SHIFT = 20,
};

// ST_SAMPLER bits.
enum SamplerBits : uint32_t {
	SM_TEXFMT_SHIFT = 0,  // 4 bits
	SM_CLUTFMT_SHIFT = 4,  // 2 bits
	SM_CLAMPS = 1 << 6,
	SM_CLAMPT = 1 << 7,
	SM_SWIZZLE = 1 << 8,
	SM_SHAREDCLUT = 1 << 9,
	SM_TEXFUNC_SHIFT = 10,  // 3 bits
	SM_TEXALPHA = 1 << 13,
	SM_DOUBLING = 1 << 14,
	SM_TEXTURES = 1 << 15,
	SM_GOURAUD = 1 << 16,
	SM_THROUGH = 1 << 17,
	SM_MIPFILT = 1 << 18,
	SM_MINFILT = 1 << 19,
	SM_MAGFILT = 1 << 20,
	SM_LEVELMODE_SHIFT = 21,  // 2 bits
	SM_CLUTTRANSFORM = 1 << 23,  // Any of the CLUT shift, mask or offset.
	SM_FORCE_LINEAR = 1 << 24,
	SM_FORCE_NEAREST = 1 << 25,
};

// A texture level's flags.
enum LevelBits : uint32_t {
	LV_VALID = 1 << 0,
	// The offset is in VRAM, and reads wrap around it.
	LV_VRAM = 1 << 1,
};

// Draws binned primitives on the GPU, into its copy of PSP memory. BinManager feeds it in drawing order:
// a state, then primitives packed with the Pack* functions below. It keeps PSP memory and its copy in sync:
// before drawing, it uploads what the CPU changed that drawing reads, and Flush() copies back what it drew.
class Backend {
public:
	virtual ~Backend() {}

	// The index in the current batch of a state, packed if it isn't yet. With snapshot, its texture levels
	// are read from the snapshots SnapshotLevel() took.
	virtual uint32_t PackState(const Rasterizer::RasterizerState &state, bool snapshot) = 0;
	// Changes when the batch is drawn and a new one starts, which forgets the packed states.
	virtual uint32_t BatchId() const = 0;
	// Where the Pack* functions append primitives.
	virtual std::vector<uint32_t> &Prims() = 0;
	// After a primitive using the state was appended at offset.
	virtual void CommitPrim(uint32_t offset, const Rasterizer::RasterizerState &state) = 0;
	// Ends a segment, the primitives drawn together. Within one each pixel sees its primitives in order, but
	// not what other pixels drew; later segments see everything earlier ones drew.
	virtual void EndSegment() = 0;
	// Copies texture levels' memory as it is now, for a primitive that textures from what it draws to. A
	// snapshot is the levels copied since BeginSnapshot(); the rest read memory.
	virtual void BeginSnapshot() = 0;
	virtual void SnapshotLevel(int level, uint32_t addr, uint32_t bytes) = 0;
	// Draws everything queued. With readback, copies what it drew back to PSP memory (waiting for it);
	// otherwise that waits for a later flush with it.
	virtual void Flush(bool readback) = 0;
	// Whether it drew to any of these bytes of VRAM without copying them back yet.
	virtual bool HasGPUWrites(uint32_t addr, uint32_t bytes) const = 0;
	virtual bool AnyGPUWrites() const = 0;
	virtual void GetStats(StringWriter &w) {}
};

// The Vulkan backend (GPU/Vulkan/ComputeRasterVulkan.cpp), or null if the device can't run it.
Backend *CreateVulkanBackend(void *drawContext);

// Appends a primitive to prims: the header words, then its setup. Returns its offset.
inline uint32_t BeginPrim(std::vector<uint32_t> &prims, PrimType type, uint32_t flags, uint32_t state, int x1, int y1, int x2, int y2) {
	const uint32_t offset = (uint32_t)prims.size();
	prims.push_back((uint32_t)type | flags);
	prims.push_back(state);
	prims.push_back((uint32_t)(x1 & 0xFFFF) | ((uint32_t)y1 << 16));
	prims.push_back((uint32_t)(x2 & 0xFFFF) | ((uint32_t)y2 << 16));
	return offset;
}

inline void PushPlane(std::vector<uint32_t> &out, int64_t base, int64_t kx, int64_t ky) {
	out.push_back((uint32_t)(uint64_t)base);
	out.push_back((uint32_t)((uint64_t)base >> 32));
	out.push_back((uint32_t)(uint64_t)kx);
	out.push_back((uint32_t)((uint64_t)kx >> 32));
	out.push_back((uint32_t)(uint64_t)ky);
	out.push_back((uint32_t)((uint64_t)ky >> 32));
}

}  // namespace SoftCompute

namespace Rasterizer {

// The setup of a binned primitive for the compute rasterizer, appended to prims (the state's index in the
// batch is stateIndex). False when it draws nothing, and then nothing was appended. These share their setup
// with DrawTriangle and the rest, so they draw exactly the same pixels.
// Sets drawOnCPU, returning false, for a triangle the shader can't draw (DrawTriangle has to).
bool PackTriangle(const VertexData &v0, const VertexData &v1, const VertexData &v2, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims, bool *drawOnCPU);
bool PackRectangle(const VertexData &v0, const VertexData &v1, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims);
bool PackSprite(const VertexData &v0, const VertexData &v1, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims);
bool PackClearRect(const VertexData &v0, const VertexData &v1, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims);
bool PackLine(const VertexData &v0, const VertexData &v1, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims);
bool PackPoint(const VertexData &v0, const BinCoords &range, const RasterizerState &state, uint32_t stateIndex, std::vector<uint32_t> &prims);

}  // namespace Rasterizer
