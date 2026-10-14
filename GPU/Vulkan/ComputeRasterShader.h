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

// The compute rasterizer's shader, after the #defines ComputeRasterVulkan.cpp generates. In pieces, since
// MSVC limits the length of a string literal.
static const char *const computeRasterShaderParts[] = {
R"(// The software renderer's rasterizer, as a Vulkan compute shader (ComputeRasterVulkan.cpp, which prepends the
// layout #defines from GPU/Software/ComputeRaster.h). A workgroup draws a 16x16 tile of a segment's primitives in
// order, each invocation two pixels side by side (so 16-bit pixels don't share a word with another invocation).
// Everything mirrors GPU/Software: Rasterizer.cpp for coverage and interpolation, Sampler.cpp, and DrawPixel.cpp's
// DrawPixelInline, down to the GE's own arithmetic (GEMath.h).

#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 8, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer MemBuf {
	uint mem[];
};
layout(std430, set = 0, binding = 1) readonly buffer DataBuf {
	uint data[];
};

layout(push_constant) uniform Params {
	uint statesOffset;
	uint primsOffset;
	uint tilesOffset;
	uint itemsOffset;
	uint memBytes;
	// CACHE_COLOR, CACHE_DEPTH: the segment's primitives all draw to one buffer, without aliasing, so each
	// invocation keeps its pixels in registers (gColor, gDepth) and writes them back at the end. cacheState
	// is a state with the buffers' layout.
	uint cacheFlags;
	uint cacheState;
} pc;

// What a segment's primitives use, so a variant of the shader can leave the rest out (ComputeRasterVulkan.cpp
// picks it). All true is the shader that draws everything.
layout(constant_id = SPEC_TEXTURE) const bool USE_TEXTURE = true;
layout(constant_id = SPEC_CLUT) const bool USE_CLUT = true;
layout(constant_id = SPEC_DXT) const bool USE_DXT = true;
layout(constant_id = SPEC_BILINEAR) const bool USE_BILINEAR = true;
layout(constant_id = SPEC_MIPS) const bool USE_MIPS = true;
layout(constant_id = SPEC_FOG) const bool USE_FOG = true;
layout(constant_id = SPEC_ALPHATEST) const bool USE_ALPHATEST = true;
layout(constant_id = SPEC_COLORTEST) const bool USE_COLORTEST = true;
layout(constant_id = SPEC_STENCIL) const bool USE_STENCIL = true;
layout(constant_id = SPEC_DEPTH) const bool USE_DEPTH = true;
layout(constant_id = SPEC_BLEND) const bool USE_BLEND = true;
layout(constant_id = SPEC_LOGICOP) const bool USE_LOGICOP = true;
layout(constant_id = SPEC_DITHER) const bool USE_DITHER = true;
layout(constant_id = SPEC_WRITEMASK) const bool USE_WRITEMASK = true;
layout(constant_id = SPEC_CLEAR) const bool USE_CLEAR = true;
layout(constant_id = SPEC_FRAGMENTS) const bool USE_FRAGMENTS = true;
layout(constant_id = SPEC_RECTS) const bool USE_RECTS = true;
layout(constant_id = SPEC_WIDE) const bool USE_WIDE = true;

const uint CACHE_COLOR = 1u;
const uint CACHE_DEPTH = 2u;

const int COMP_NEVER = 0;
const int COMP_ALWAYS = 1;
const int COMP_EQUAL = 2;
const int COMP_NOTEQUAL = 3;
const int COMP_LESS = 4;
const int COMP_LEQUAL = 5;
const int COMP_GREATER = 6;
const int COMP_GEQUAL = 7;

const uint FMT_565 = 0u;
const uint FMT_5551 = 1u;
const uint FMT_4444 = 2u;
const uint FMT_8888 = 3u;

const uint TFMT_5650 = 0u;
const uint TFMT_5551 = 1u;
const uint TFMT_4444 = 2u;
const uint TFMT_8888 = 3u;
const uint TFMT_CLUT4 = 4u;
const uint TFMT_CLUT8 = 5u;
const uint TFMT_CLUT16 = 6u;
const uint TFMT_CLUT32 = 7u;
const uint TFMT_DXT1 = 8u;
const uint TFMT_DXT3 = 9u;
const uint TFMT_DXT5 = 10u;

// Memory.

uint MemRead32(uint addr) {
	return mem[addr >> 2];
}

uint MemRead16(uint addr) {
	return (mem[addr >> 2] >> ((addr & 2u) << 3)) & 0xFFFFu;
}

uint MemRead8(uint addr) {
	return (mem[addr >> 2] >> ((addr & 3u) << 3)) & 0xFFu;
}

void MemWrite16(uint addr, uint v) {
	const uint i = addr >> 2;
	const uint shift = (addr & 2u) << 3;
	mem[i] = (mem[i] & ~(0xFFFFu << shift)) | ((v & 0xFFFFu) << shift);
}

// Colors (Common/Data/Convert/ColorConv.h).

uint RGB565ToRGBA8888(uint src) {
	const uint darkRB = ((src & 0x1Fu) << 3) | ((src & 0xF800u) << 8);
	const uint fullRB = darkRB | ((darkRB >> 5) & 0x070007u);
	const uint darkG = (src & 0x7E0u) << 5;
	const uint fullG = darkG | ((darkG >> 6) & 0x300u);
	return fullRB | fullG | 0xFF000000u;
}

uint RGBA5551ToRGBA8888(uint src) {
	const uint dark = ((src & 0x1Fu) << 3) | ((src & 0x3E0u) << 6) | ((src & 0x7C00u) << 9);
	uint full = dark | ((dark >> 5) & 0x070707u);
	if ((src >> 15) != 0u)
		full |= 0xFF000000u;
	return full;
}

uint RGBA4444ToRGBA8888(uint src) {
	const uint c = (src & 0xFu) | ((src & 0xF0u) << 4) | ((src & 0xF00u) << 8) | ((src & 0xF000u) << 12);
	return c | (c << 4);
}

uint RGBA8888ToRGB565(uint v) {
	return ((v >> 3) & 0x1Fu) | ((v >> 5) & (0x3Fu << 5)) | ((v >> 8) & (0x1Fu << 11));
}

uint RGBA8888ToRGBA5551(uint v) {
	return ((v >> 3) & 0x1Fu) | ((v >> 6) & (0x1Fu << 5)) | ((v >> 9) & (0x1Fu << 10)) | ((v >> 16) & 0x8000u);
}

uint RGBA8888ToRGBA4444(uint v) {
	const uint c = v >> 4;
	return (c & 0xFu) | ((c >> 4) & 0xF0u) | ((c >> 8) & 0xF00u) | ((c >> 12) & 0xF000u);
}

ivec4 Unpack(uint c) {
	return ivec4(int(c & 0xFFu), int((c >> 8) & 0xFFu), int((c >> 16) & 0xFFu), int(c >> 24));
}

// Vec3<int>::ToRGB(): clamped.
uint PackRGB(ivec3 c) {
	const uvec3 k = uvec3(clamp(c, ivec3(0), ivec3(255)));
	return k.r | (k.g << 8) | (k.b << 16);
}

// The GE's arithmetic (GEMath.h).

int Msb64(uint64_t v) {
	const uvec2 p = unpackUint2x32(v);
	return p.y != 0u ? 32 + findMSB(p.y) : findMSB(p.x);
}

// The float nearest to mag * 2^e2 (mag > 0), ties to even, as a float conversion rounds: with denormals, and
// infinity past the range.
uint FloatBitsRNE(uint64_t mag, int e2) {
	const int msb = Msb64(mag);
	int biased = msb + e2 + 127;
	if (biased >= 255)
		return 0x7F800000u;
	// Bits to drop to leave 24 significant ones, or fewer for a denormal.
	const int shift = biased >= 1 ? msb - 23 : msb - 23 + (1 - biased);
	uint64_t m;
	if (shift <= 0) {
		m = mag << uint(-shift);
	} else if (shift > 62) {
		return 0u;
	} else {
		m = mag >> uint(shift);
		const uint64_t rem = mag & ((uint64_t(1) << uint(shift)) - uint64_t(1));
		const uint64_t halfway = uint64_t(1) << uint(shift - 1);
		if (rem > halfway || (rem == halfway && (m & uint64_t(1)) != uint64_t(0)))
			m += uint64_t(1);
	}
	if (biased >= 1) {
		if (m >= (uint64_t(1) << 24)) {
			m >>= 1;
			biased++;
			if (biased >= 255)
				return 0x7F800000u;
		}
		return (uint(biased) << 23) | (uint(m) & 0x7FFFFFu);
	}
	// A denormal, or the smallest normal if it rounded up to it.
	return uint(m);
}

// TruncateToFloat24((float)((double)v * 2^e2)).
uint ToFloat24(int64_t v, int e2) {
	if (v == int64_t(0))
		return 0u;
	const uint sign = v < int64_t(0) ? 0x80000000u : 0u;
	const uint64_t mag = uint64_t(v < int64_t(0) ? -v : v);
	return (FloatBitsRNE(mag, e2) | sign) & 0xFFFFFF00u;
}

// GERecip of a positive float24.
uint GERecipBits(uint bits) {
	const uint i = (bits >> 8) & 0x7FFFu;
	const int e = int((bits >> 23) & 0xFFu) - 127;
	const ivec2 seg = GE_RECIP[i >> 8];
	const int q = (64 * seg.x + 63 + seg.y * int(i & 255u)) >> 7;
	return FloatBitsRNE(uint64_t(q), -16 - e);
}

// GEUVProduct((double)x * r): the exact product with 24 significant bits kept (truncated), as a float.
uint UVProductBits(uint xb, uint rb) {
	if ((xb & 0x7FFFFFFFu) == 0u)
		return xb;
	const uint sign = (xb ^ rb) & 0x80000000u;
	const uint xe = (xb >> 23) & 0xFFu;
	const uint re = (rb >> 23) & 0xFFu;
	const uint xm = xe == 0u ? (xb & 0x7FFFFFu) : ((xb & 0x7FFFFFu) | 0x800000u);
	const uint rm = re == 0u ? (rb & 0x7FFFFFu) : ((rb & 0x7FFFFFu) | 0x800000u);
	if (rm == 0u)
		return sign;
	int e2 = (xe == 0u ? 1 : int(xe)) + (re == 0u ? 1 : int(re)) - 2 * (127 + 23);
	uint64_t p = uint64_t(xm) * uint64_t(rm);
	const int msb = Msb64(p);
	if (msb > 23) {
		p >>= uint(msb - 23);
		e2 += msb - 23;
	}
	return FloatBitsRNE(p, e2) | sign;
}

// A float times 2^-n, as the float multiply would round it.
uint ScaleDownBits(uint bits, int n) {
	const uint e = (bits >> 23) & 0xFFu;
	if (e == 0xFFu || (bits & 0x7FFFFFFFu) == 0u || n == 0)
		return bits;
	if (e != 0u && int(e) - n >= 1)
		return bits - (uint(n) << 23);
	const uint sign = bits & 0x80000000u;
	const uint m = e == 0u ? (bits & 0x7FFFFFu) : ((bits & 0x7FFFFFu) | 0x800000u);
	return FloatBitsRNE(uint64_t(m), (e == 0u ? 1 : int(e)) - 150 - n) | sign;
}

int GELog16(uint bits) {
	return int((bits >> 19) & 0x0FFFu) - 127 * 16;
}

// The texture coordinates from the UV planes' values at a pixel (Rasterizer's GetTextureCoordinatesGE).
void UVFromPlanes(int64_t qv, int64_t sv, int64_t tv, int expQ, int expS, int expT, out uint s, out uint t, out uint q) {
	q = ToFloat24(qv, expQ);
	if (int(q) <= 0) {
		s = 0u;
		t = 0u;
		return;
	}
	const uint r = GERecipBits(q);
	s = UVProductBits(ToFloat24(sv, expS), r);
	t = UVProductBits(ToFloat24(tv, expT), r);
}

// UVFromPlanes in 32 bits, as GEUVSpanCore does it: right whenever the values and results stay normal
// floats, else it takes the exact path.
void UVFromPlanes32(int qv, int sv, int tv, int expQ, int expS, int expT, out uint s, out uint t, out uint q) {
	bool bad = false;
	uint xb[3];
	const int vals[3] = int[3](qv, sv, tv);
	const int exps[3] = int[3](expQ, expS, expT);
	for (int i = 0; i < 3; ++i) {
		if (vals[i] == 0) {
			xb[i] = 0u;
			continue;
		}
		const uint fb = floatBitsToUint(float(vals[i]));
		const int field = int((fb >> 23) & 0xFFu) + exps[i];
		bad = bad || field < 1 || field > 254;
		xb[i] = (fb + uint(exps[i] << 23)) & 0xFFFFFF00u;
	}
	q = xb[0];
	if (int(q) <= 0) {
		s = 0u;
		t = 0u;
		if (bad)
			UVFromPlanes(int64_t(qv), int64_t(sv), int64_t(tv), expQ, expS, expT, s, t, q);
		return;
	}
	// GERecip, normal for q below 2^126.
	const int qe = int((q >> 23) & 0xFFu) - 127;
	bad = bad || qe > 125;
	const ivec2 seg = GE_RECIP[(q >> 16) & 0x7Fu];
	const int recip = (64 * seg.x + 63 + seg.y * int((q >> 8) & 0xFFu)) >> 7;
	const uint rb = floatBitsToUint(float(recip)) - uint((16 + qe) << 23);
	const uint rA = ((rb >> 8) & 0x7FFFu) | 0x8000u;
	const int re = int((rb >> 23) & 0xFFu);
	// GEUVProduct: the 16-bit significands' product, cut to 24 bits.
	uint out2[2];
	for (int i = 0; i < 2; ++i) {
		const uint x = xb[i + 1];
		const int xe = int((x >> 23) & 0xFFu);
		if (xe == 0) {
			bad = bad || (x & 0x7FFFFFFFu) != 0u;
			out2[i] = x;
			continue;
		}
		const uint p = (((x >> 8) & 0x7FFFu) | 0x8000u) * rA;
		const bool hi = (p & 0x80000000u) != 0u;
		const uint m = hi ? ((p >> 8) & 0x00FFFFFFu) : (p >> 7);
		const int adjust = xe + re + (hi ? 8 : 7) - (254 + 30);
		const int field = 150 + adjust;
		bad = bad || field < 1 || field > 254;
		out2[i] = (floatBitsToUint(float(m)) + uint(adjust << 23)) | (x & 0x80000000u);
	}
	s = out2[0];
	t = out2[1];
	if (bad) {
		UVFromPlanes(int64_t(qv), int64_t(sv), int64_t(tv), expQ, expS, expT, s, t, q);
	}
}

uint ToFloat24Fast(int v, int e2) {
	if (v == 0)
		return 0u;
	const uint fb = floatBitsToUint(float(v));
	const int field = int((fb >> 23) & 0xFFu) + e2;
	if (field < 1 || field > 254)
		return ToFloat24(int64_t(v), e2);
	return (fb + uint(e2 << 23)) & 0xFFFFFF00u;
}

// Planes and edges, { base, kx, ky } in 64 bits. Summed in 32 bits when the primitive's flag says the sums fit.

int64_t Load64(uint o) {
	return int64_t(packUint2x32(uvec2(data[o], data[o + 1u])));
}

int64_t PlaneSum(uint o, bool fits32, int xc, int yc) {
	if (!USE_WIDE || fits32)
		return int64_t(int(data[o] + data[o + 2u] * uint(xc) + data[o + 4u] * uint(yc)));
	return Load64(o) + Load64(o + 2u) * int64_t(xc) + Load64(o + 4u) * int64_t(yc);
}

int PlaneValue(uint o, bool fits32, int xc, int yc) {
	return int(PlaneSum(o, fits32, xc, yc) >> 14);
}

int64_t PlaneValue64(uint o, bool fits32, int xc, int yc) {
	return PlaneSum(o, fits32, xc, yc) >> 14;
}

bool EdgeInside(uint o, bool fits32, int xc, int yc) {
	if (!USE_WIDE || fits32)
		return int(data[o] * uint(xc) + data[o + 1u] * uint(yc) + data[o + 2u]) >= 0;
	return int64_t(int(data[o])) * int64_t(xc) + int64_t(int(data[o + 1u])) * int64_t(yc) + Load64(o + 2u) >= int64_t(0);
}

// Textures (Sampler.cpp).

uint TexBits(uint fmt) {
	switch (fmt) {
	case TFMT_8888: case TFMT_CLUT32: return 32u;
	case TFMT_CLUT8: case TFMT_DXT3: case TFMT_DXT5: return 8u;
	case TFMT_CLUT4: case TFMT_DXT1: return 4u;
	default: return 16u;
	}
}
)",
R"(
uint TexAddr(uint base, uint flags, uint offset) {
	uint a = base + offset;
	if ((flags & LV_VRAM) != 0u)
		a &= 0x001FFFFFu;
	return min(a, pc.memBytes - 4u);
}

uint ClutIndex(uint st, uint index) {
	if ((data[st + ST_SAMPLER] & SM_CLUTTRANSFORM) != 0u) {
		const uint fmt = data[st + ST_CLUTFORMAT];
		const uint shift = (fmt >> 2) & 0x1Fu;
		const uint mask = (fmt >> 8) & 0xFFu;
		const uint offset = ((fmt >> 16) & 0x1Fu) << 4;
		const uint offsetMask = (fmt & 3u) == 3u ? 0xFFu : 0x1FFu;
		return ((index >> shift) & mask) | (offset & offsetMask);
	}
	return index & 0xFFu;
}

uint LookupClut(uint st, uint index) {
	const uint base = data[st + ST_CLUT_ADDR];
	switch ((data[st + ST_SAMPLER] >> SM_CLUTFMT_SHIFT) & 3u) {
	case 0u: return RGB565ToRGBA8888(MemRead16(base + index * 2u));
	case 1u: return RGBA5551ToRGBA8888(MemRead16(base + index * 2u));
	case 2u: return RGBA4444ToRGBA8888(MemRead16(base + index * 2u));
	default: return MemRead32(base + index * 4u);
	}
}

uint MakeCol(int r, int g, int b, int a) {
	return (uint(a) << 24) | (uint(b) << 16) | (uint(g) << 8) | uint(r);
}

// GetDXTTexelColor.
uint DXTTexelColor(uint block, uint x, uint y, int alpha) {
	const uint lines = MemRead32(block);
	const uint colors = MemRead32(block + 4u);
	const uint c1 = colors & 0xFFFFu, c2 = colors >> 16;
	const int blue1 = int((c1 << 3) & 0xF8u), blue2 = int((c2 << 3) & 0xF8u);
	const int green1 = int((c1 >> 3) & 0xFCu), green2 = int((c2 >> 3) & 0xFCu);
	const int red1 = int((c1 >> 8) & 0xF8u), red2 = int((c2 >> 8) & 0xF8u);
	const uint colorIndex = (((lines >> (y * 8u)) & 0xFFu) >> (x * 2u)) & 3u;
	if (colorIndex == 0u)
		return MakeCol(red1, green1, blue1, alpha);
	if (colorIndex == 1u)
		return MakeCol(red2, green2, blue2, alpha);
	if (c1 > c2) {
		if (colorIndex == 2u)
			return MakeCol((red1 + red1 + red2) / 3, (green1 + green1 + green2) / 3, (blue1 + blue1 + blue2) / 3, alpha);
		return MakeCol((red2 + red2 + red1) / 3, (green2 + green2 + green1) / 3, (blue2 + blue2 + blue1) / 3, alpha);
	}
	if (colorIndex == 3u)
		return 0u;
	return MakeCol((red1 + red2) / 2, (green1 + green2) / 2, (blue1 + blue2) / 2, alpha);
}

uint DXT5Lerp(uint alpha1, uint alpha2, uint n, uint steps) {
	const uint a1 = (alpha1 * ((steps - n) << 8)) / steps;
	const uint a2 = (alpha2 * (n << 8)) / steps;
	return ((a1 + a2 + 31u) >> 8) & 0xFFu;
}

uint DXTTexel(uint fmt, uint base, uint flags, uint bufw, uint u, uint v) {
	const uint blockIndex = (v >> 2) * (bufw >> 2) + (u >> 2);
	const uint x = u & 3u, y = v & 3u;
	if (fmt == TFMT_DXT1)
		return DXTTexelColor(TexAddr(base, flags, blockIndex * 8u), x, y, 255);
	const uint block = TexAddr(base, flags, blockIndex * 16u);
	const uint color = DXTTexelColor(block, x, y, 0);
	if (fmt == TFMT_DXT3) {
		const uint alpha = (MemRead16(TexAddr(base, flags, blockIndex * 16u + 8u + y * 2u)) >> (x * 4u)) & 0xFu;
		return color | (alpha << 28);
	}
	// DXT5: alphadata2 (32 bits), alphadata1 (16 bits), alpha1, alpha2.
	const uint data2 = MemRead32(TexAddr(base, flags, blockIndex * 16u + 8u));
	const uint data1 = MemRead32(TexAddr(base, flags, blockIndex * 16u + 12u));
	const uint alpha1 = (data1 >> 16) & 0xFFu, alpha2 = data1 >> 24;
	const uint64_t alphadata = (uint64_t(data1 & 0xFFFFu) << 32) | uint64_t(data2);
	const uint alphaIndex = uint(alphadata >> (y * 12u + x * 3u)) & 7u;
	if (alphaIndex == 0u)
		return color | (alpha1 << 24);
	if (alphaIndex == 1u)
		return color | (alpha2 << 24);
	if (alpha1 > alpha2)
		return color | (DXT5Lerp(alpha1, alpha2, alphaIndex - 1u, 7u) << 24);
	if (alphaIndex == 6u)
		return color;
	if (alphaIndex == 7u)
		return color | 0xFF000000u;
	return color | (DXT5Lerp(alpha1, alpha2, alphaIndex - 1u, 5u) << 24);
}

// A texel of a level, at coordinates already wrapped or clamped.
uint ReadTexel(uint st, int level, uint u, uint v) {
	const uint lv = st + ST_LEVELS + uint(level) * 4u;
	const uint flags = data[lv + 3u];
	if ((flags & LV_VALID) == 0u)
		return 0u;
	const uint bufw = data[lv + 1u];
	const uint base = data[lv + 2u];
	const uint samplerBits = data[st + ST_SAMPLER];
	const uint fmt = (samplerBits >> SM_TEXFMT_SHIFT) & 0xFu;
	if (USE_DXT && fmt >= TFMT_DXT1)
		return DXTTexel(fmt, base, flags, bufw, u, v);

	const uint bits = TexBits(fmt);
	uint offset;
	if ((samplerBits & SM_SWIZZLE) == 0u) {
		offset = v * ((bufw * bits) >> 3) + ((u * bits) >> 3);
	} else {
		const uint b = (u * bits) >> 3;
		offset = (v >> 3) * (((bufw * bits) / 32u) * 32u) + (v & 7u) * 16u + (b >> 4) * 128u + (b & 15u);
	}
	const uint addr = TexAddr(base, flags, offset);
	// Levels past the first have their own palette unless it's shared.
	uint clutOffset = 0u;
	if ((samplerBits & SM_SHAREDCLUT) == 0u) {
		if (fmt == TFMT_CLUT4)
			clutOffset = uint(level) * 16u;
		else if (fmt == TFMT_CLUT8)
			clutOffset = (uint(level) & 1u) * 256u;
	}
	switch (fmt) {
	case TFMT_4444: return RGBA4444ToRGBA8888(MemRead16(addr));
	case TFMT_5551: return RGBA5551ToRGBA8888(MemRead16(addr));
	case TFMT_5650: return RGB565ToRGBA8888(MemRead16(addr));
	case TFMT_8888: return MemRead32(addr);
	case TFMT_CLUT32: if (!USE_CLUT) return 0u; return LookupClut(st, ClutIndex(st, MemRead32(addr)) + clutOffset);
	case TFMT_CLUT16: if (!USE_CLUT) return 0u; return LookupClut(st, ClutIndex(st, MemRead16(addr)) + clutOffset);
	case TFMT_CLUT8: if (!USE_CLUT) return 0u; return LookupClut(st, ClutIndex(st, MemRead8(addr)) + clutOffset);
	default: {
		if (!USE_CLUT)
			return 0u;
		const uint b = MemRead8(addr);
		return LookupClut(st, ClutIndex(st, (u & 1u) != 0u ? (b >> 4) : (b & 0xFu)) + clutOffset);
	}
	}
}

int ClampUV(int v, int size) {
	if (v >= size - 1)
		return size - 1;
	if (v >= 511)
		return 511;
	if (v < 0)
		return 0;
	return v;
}

int WrapUV(int v, int size) {
	return v & (size - 1) & 511;
}

// In 1/16 texel, truncated, or INT_MIN out of int range.
int TexelFixed(float f) {
	return f >= -2147483648.0 && f < 2147483648.0 ? int(f) : int(0x80000000u);
}

ivec2 LevelSize(uint st, int level) {
	const uint wh = data[st + ST_LEVELS + uint(level) * 4u];
	return ivec2(int(wh & 0xFFFFu), int(wh >> 16));
}

ivec2 TexelPair(int base, int size, bool clampIt) {
	if (clampIt) {
		const int hi = size > 512 ? 511 : size - 1;
		return ivec2(max(min(base, hi), 0), max(min(base + 1, hi), 0));
	}
	const int mask = (size - 1) & 511;
	return ivec2(base & mask, (base + 1) & mask);
}

// Sampling, nearest (SampleNearest) or bilinear (SampleLinearLevel), at a level and with levelFrac the next
// one too. Texels are read in one loop, so the texel decoding is in the shader once.
ivec4 SampleTexture(uint st, int level, int levelFrac, bool bilinear, uint sBits, uint tBits) {
	const uint samplerBits = data[st + ST_SAMPLER];
	const bool clampS = (samplerBits & SM_CLAMPS) != 0u;
	const bool clampT = (samplerBits & SM_CLAMPT) != 0u;
	bilinear = USE_BILINEAR && bilinear;
	const int levels = USE_MIPS && levelFrac != 0 ? 2 : 1;
	const int texels = bilinear ? 4 : 1;
	ivec4 result[2] = ivec4[2](ivec4(0), ivec4(0));
	for (int l = 0; l < levels; ++l) {
		const int lv = level + l;
		const ivec2 size = LevelSize(st, lv);
		const float fs = uintBitsToFloat(sBits) * float(size.x) * 16.0;
		const float ft = uintBitsToFloat(tBits) * float(size.y) * 16.0;
		ivec2 us, vs;
		int fracU = 0, fracV = 0;
		if (bilinear) {
			// A level whose pointer is invalid filters to zero.
			if ((data[st + ST_LEVELS + uint(lv) * 4u + 3u] & LV_VALID) == 0u)
				continue;
			const int baseU = TexelFixed(fs) - 8;
			const int baseV = TexelFixed(ft) - 8;
			fracU = baseU & 15;
			fracV = baseV & 15;
			us = TexelPair(baseU >> 4, size.x, clampS);
			vs = TexelPair(baseV >> 4, size.y, clampT);
		} else {
			const int u = TexelFixed(fs) >> 4;
			const int v = TexelFixed(ft) >> 4;
			us.x = clampS ? ClampUV(u, size.x) : WrapUV(u, size.x);
			vs.x = clampT ? ClampUV(v, size.y) : WrapUV(v, size.y);
		}
		ivec4 c[4];
		[[dont_unroll]]
		for (int i = 0; i < texels; ++i)
			c[i] = Unpack(ReadTexel(st, lv, uint((i & 1) != 0 ? us.y : us.x), uint((i & 2) != 0 ? vs.y : vs.x)));
		if (bilinear) {
			// Like the GE: horizontal lerps truncated to 8 bits, then the vertical one.
			const ivec4 top = (c[0] * (16 - fracU) + c[1] * fracU) >> 4;
			const ivec4 bot = (c[2] * (16 - fracU) + c[3] * fracU) >> 4;
			result[l] = (top * (16 - fracV) + bot * fracV) >> 4;
		} else {
			result[l] = c[0];
		}
	}
	if (USE_MIPS && levelFrac != 0)
		return (result[1] * levelFrac + result[0] * (16 - levelFrac)) >> 4;
	return result[0];
}

// GetTextureFunctionOutput.
ivec4 TextureFunction(uint st, ivec4 prim, ivec4 tex) {
	const uint samplerBits = data[st + ST_SAMPLER];
	const bool rgba = (samplerBits & SM_TEXALPHA) != 0u;
	const bool doubling = (samplerBits & SM_DOUBLING) != 0u;
	const int modulatedA = rgba ? ((prim.a + 1) * tex.a) >> 8 : prim.a;
	switch ((samplerBits >> SM_TEXFUNC_SHIFT) & 7u) {
	case 0u:  // MODULATE
		return ivec4(((prim.rgb + ivec3(1)) * (doubling ? tex.rgb * 2 : tex.rgb)) >> 8, modulatedA);
	case 1u:  // DECAL
		if (rgba) {
			const int t = tex.a, invt = 255 - tex.a;
			const ivec3 sum = (prim.rgb + ivec3(1)) * invt + (tex.rgb + ivec3(1)) * t;
			return ivec4(doubling ? sum >> 7 : sum >> 8, prim.a);
		}
		return ivec4(doubling ? tex.rgb * 2 : tex.rgb, prim.a);
	case 2u: {  // BLEND
		const ivec3 env = Unpack(data[st + ST_TEXENV]).rgb;
		const ivec3 sum = (ivec3(255) - tex.rgb) * prim.rgb + tex.rgb * env + ivec3(255);
		return ivec4(doubling ? sum >> 7 : sum >> 8, modulatedA);
	}
	case 3u:  // REPLACE
		return ivec4(doubling ? tex.rgb * 2 : tex.rgb, rgba ? tex.a : prim.a);
	default:  // ADD and the unknown ones.
		return ivec4(doubling ? (prim.rgb + tex.rgb) * 2 : prim.rgb + tex.rgb, modulatedA);
	}
}

// The level and filter for a level of detail (CalculateSamplingParams, after the detail).
void SamplingParams(uint st, int detail, out int level, out int levelFrac, out bool bilinear) {
	const uint samplerBits = data[st + ST_SAMPLER];
	const int maxLevel = int((data[st + ST_TEXLEVEL] >> 8) & 0xFFu);
	detail += int(data[st + ST_TEXLEVEL] << 24) >> 24;
	if (USE_MIPS && detail > 0 && maxLevel > 0) {
		const bool mipFilt = (samplerBits & SM_MIPFILT) != 0u;
		int level8 = min(detail, maxLevel * 16);
		if (!mipFilt)
			level8 += 8;
		level = level8 >> 4;
		levelFrac = mipFilt ? (level8 & 0xF) : 0;
	} else {
		level = 0;
		levelFrac = 0;
	}
	bilinear = detail > 0 ? (samplerBits & SM_MINFILT) != 0u : (samplerBits & SM_MAGFILT) != 0u;
}

ivec4 ApplyTexturing(uint st, ivec4 prim, uint s, uint t, int level, int levelFrac, bool bilinear) {
	const uint samplerBits = data[st + ST_SAMPLER];
	if ((samplerBits & SM_FORCE_LINEAR) != 0u)
		bilinear = true;
	else if ((samplerBits & SM_FORCE_NEAREST) != 0u)
		bilinear = false;
	return TextureFunction(st, prim, SampleTexture(st, level, levelFrac, bilinear, s, t));
}

// The pixel pipeline (DrawPixelInline).

uint FBAddr(uint st, int x, int y, uint bpp) {
	const int stride = int(data[st + ST_STRIDES] & 0xFFFFu);
	return (data[st + ST_FB_ADDR] + uint(x + y * stride) * bpp) & 0x001FFFFFu;
}

uint DepthAddr(uint st, int x, int y) {
	const int stride = int(data[st + ST_STRIDES] >> 16);
	const uint offset = data[st + ST_Z_BASE] + uint(x + y * stride) * 2u;
	const uint rotMask = data[st + ST_Z_ROTMASK];
	const uint f = offset & rotMask;
	const uint stored = ((offset & ~rotMask) | (((f << 1) | (f >> data[st + ST_Z_ROTSHIFT])) & rotMask)) ^ data[st + ST_Z_XOR];
	return stored & 0x001FFFFFu;
}

// The invocation's pixels while cached: the color as stored (16 or 32 bits) and the depth, for pixel gPix.
uint gColor[2];
uint gDepth[2];
bool gColorDirty[2];
bool gDepthDirty[2];
int gPix = 0;

uint FBRead16(uint addr) {
	return (pc.cacheFlags & CACHE_COLOR) != 0u ? gColor[gPix] : MemRead16(addr);
}

uint FBRead32(uint addr) {
	return (pc.cacheFlags & CACHE_COLOR) != 0u ? gColor[gPix] : MemRead32(addr);
}

void FBWrite16(uint addr, uint v) {
	if ((pc.cacheFlags & CACHE_COLOR) != 0u) {
		gColor[gPix] = v & 0xFFFFu;
		gColorDirty[gPix] = true;
	} else {
		MemWrite16(addr, v);
	}
}
)",
R"(
void FBWrite32(uint addr, uint v) {
	if ((pc.cacheFlags & CACHE_COLOR) != 0u) {
		gColor[gPix] = v;
		gColorDirty[gPix] = true;
	} else {
		mem[addr >> 2] = v;
	}
}

uint DepthRead(uint st, int x, int y) {
	return (pc.cacheFlags & CACHE_DEPTH) != 0u ? gDepth[gPix] : MemRead16(DepthAddr(st, x, y));
}

void DepthWrite(uint st, int x, int y, uint z) {
	if ((pc.cacheFlags & CACHE_DEPTH) != 0u) {
		gDepth[gPix] = z & 0xFFFFu;
		gDepthDirty[gPix] = true;
	} else {
		MemWrite16(DepthAddr(st, x, y), z);
	}
}

// Writing back a cached 16-bit pixel: the other half of the word can be another invocation's.
void AtomicWrite16(uint addr, uint v) {
	const uint i = addr >> 2;
	const uint shift = (addr & 2u) << 3;
	atomicAnd(mem[i], ~(0xFFFFu << shift));
	atomicOr(mem[i], (v & 0xFFFFu) << shift);
}

bool Compare(int func, int a, int b) {
	switch (func) {
	case COMP_NEVER: return false;
	case COMP_ALWAYS: return true;
	case COMP_EQUAL: return a == b;
	case COMP_NOTEQUAL: return a != b;
	case COMP_LESS: return a < b;
	case COMP_LEQUAL: return a <= b;
	case COMP_GREATER: return a > b;
	default: return a >= b;
	}
}

bool DepthTestPassed(uint st, int func, int x, int y, int z) {
	return Compare(func, z, int(DepthRead(st, x, y)));
}

// The depth range and depth tests some primitives do before texturing (earlyZChecks).
bool EarlyZPassed(uint st, int x, int y, int z) {
	const uint px0 = data[st + ST_PIXEL0];
	if (!USE_DEPTH || (px0 & PX0_EARLYZ) == 0u)
		return true;
	if ((px0 & PX0_DEPTHRANGE) != 0u && (z < int(data[st + ST_MINZ]) || z > int(data[st + ST_MAXZ])))
		return false;
	return DepthTestPassed(st, int((px0 >> PX0_DEPTHTEST_SHIFT) & 7u), x, y, z);
}

uint ReadColor(uint fmt, uint addr) {
	switch (fmt) {
	case FMT_565: return RGB565ToRGBA8888(FBRead16(addr)) & 0x00FFFFFFu;
	case FMT_5551: return RGBA5551ToRGBA8888(FBRead16(addr));
	case FMT_4444: return RGBA4444ToRGBA8888(FBRead16(addr));
	default: return FBRead32(addr);
	}
}

void WriteColor(uint fmt, uint addr, uint value, uint oldValue, uint writeMask) {
	switch (fmt) {
	case FMT_565:
		value = RGBA8888ToRGB565(value);
		if (writeMask != 0u)
			value = (value & ~writeMask) | (RGBA8888ToRGB565(oldValue) & writeMask);
		FBWrite16(addr, value);
		break;
	case FMT_5551:
		value = RGBA8888ToRGBA5551(value);
		if (writeMask != 0u)
			value = (value & ~writeMask) | (RGBA8888ToRGBA5551(oldValue) & writeMask);
		FBWrite16(addr, value);
		break;
	case FMT_4444:
		value = RGBA8888ToRGBA4444(value);
		if (writeMask != 0u)
			value = (value & ~writeMask) | (RGBA8888ToRGBA4444(oldValue) & writeMask);
		FBWrite16(addr, value);
		break;
	default:
		FBWrite32(addr, (value & ~writeMask) | (oldValue & writeMask));
		break;
	}
}

uint ReadStencil(uint fmt, uint addr) {
	switch (fmt) {
	case FMT_565: return 0u;
	case FMT_5551: return (FBRead16(addr) & 0x8000u) != 0u ? 0xFFu : 0u;
	case FMT_4444: {
		const uint v = FBRead16(addr) >> 12;
		return (v << 4) | v;
	}
	default: return FBRead32(addr) >> 24;
	}
}

void WriteStencil(uint fmt, uint addr, uint writeMask, uint value) {
	if (fmt == FMT_565)
		return;
	if (fmt == FMT_5551) {
		if ((writeMask & 0x8000u) == 0u)
			FBWrite16(addr, (FBRead16(addr) & ~0x8000u) | ((value & 0x80u) << 8));
	} else if (fmt == FMT_4444) {
		const uint mask = writeMask | 0x0FFFu;
		FBWrite16(addr, (FBRead16(addr) & mask) | ((value << 8) & ~mask & 0xFFFFu));
	} else {
		const uint mask = writeMask | 0x00FFFFFFu;
		FBWrite32(addr, (FBRead32(addr) & mask) | ((value << 24) & ~mask));
	}
}

uint StencilOp(uint fmt, uint replace, uint op, uint old) {
	switch (op) {
	case 0u: return old;
	case 1u: return 0u;
	case 2u: return replace;
	case 3u: return ~old & 0xFFu;
	case 4u:
		if (fmt == FMT_8888)
			return old != 0xFFu ? old + 1u : old;
		if (fmt == FMT_5551)
			return 0xFFu;
		if (fmt == FMT_4444)
			return old < 0xF0u ? old + 0x10u : old;
		return old;
	case 5u:
		if (fmt == FMT_4444)
			return old >= 0x10u ? old - 0x10u : old;
		if (fmt == FMT_5551)
			return 0u;
		return old != 0u ? old - 1u : old;
	default: return old;
	}
}

uint LogicOp(uint op, uint oldColor, uint newColor) {
	switch (op) {
	case 0u: return newColor & 0xFF000000u;
	case 1u: return newColor & (oldColor | 0xFF000000u);
	case 2u: return newColor & (~oldColor | 0xFF000000u);
	case 3u: return newColor;
	case 4u: return (~newColor & (oldColor & 0x00FFFFFFu)) | (newColor & 0xFF000000u);
	case 5u: return (oldColor & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 6u: return newColor ^ (oldColor & 0x00FFFFFFu);
	case 7u: return newColor | (oldColor & 0x00FFFFFFu);
	case 8u: return (~(newColor | oldColor) & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 9u: return (~(newColor ^ oldColor) & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 10u: return (~oldColor & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 11u: return newColor | (~oldColor & 0x00FFFFFFu);
	case 12u: return (~newColor & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 13u: return ((~newColor | oldColor) & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	case 14u: return (~(newColor & oldColor) & 0x00FFFFFFu) | (newColor & 0xFF000000u);
	default: return newColor | 0x00FFFFFFu;
	}
}

ivec3 BlendFactor(uint factor, ivec4 src, ivec4 dst, ivec3 other, uint fix) {
	switch (factor) {
	case 0u: return other;
	case 1u: return ivec3(255) - other;
	case 2u: return ivec3(src.a);
	case 3u: return ivec3(255 - src.a);
	case 4u: return ivec3(dst.a);
	case 5u: return ivec3(255 - dst.a);
	case 6u: return ivec3(2 * src.a);
	case 7u: return ivec3(255 - 2 * src.a);
	case 8u: return ivec3(2 * dst.a);
	case 9u: return ivec3(255 - 2 * dst.a);
	case 11u: return ivec3(0);
	case 12u: return ivec3(255);
	default: return Unpack(fix).rgb;
	}
}

ivec3 BlendTerm(ivec3 c, ivec3 f) {
	return ((c * 2 + ivec3(1)) * (f * 2 + ivec3(1))) >> 10;
}

// 255 - 2a goes negative, and then the term subtracts.
ivec3 SignedBlendTerm(ivec3 c, ivec3 f) {
	const ivec3 t = ((2 * c + ivec3(1)) * (2 * abs(f) + ivec3(1))) >> 10;
	return mix(t, -t, lessThan(f, ivec3(0)));
}

// AlphaBlendingResult.
ivec3 AlphaBlend(uint st, ivec4 src, ivec4 dst) {
	const uint px1 = data[st + ST_PIXEL1];
	const uint eq = (px1 >> PX1_BLENDEQ_SHIFT) & 7u;
	const uint srcFactor = (px1 >> PX1_BLENDSRC_SHIFT) & 0xFu;
	const uint dstFactor = (px1 >> PX1_BLENDDST_SHIFT) & 0xFu;
	if (eq <= 2u) {
		const ivec3 sf = BlendFactor(srcFactor, src, dst, dst.rgb, data[st + ST_FIXA]);
		const ivec3 df = BlendFactor(dstFactor, src, dst, src.rgb, data[st + ST_FIXB]);
		const bool signedFactor = srcFactor == 7u || srcFactor == 9u || dstFactor == 7u || dstFactor == 9u;
		const ivec3 s = signedFactor ? SignedBlendTerm(src.rgb, sf) : BlendTerm(src.rgb, sf);
		const ivec3 d = signedFactor ? SignedBlendTerm(dst.rgb, df) : BlendTerm(dst.rgb, df);
		if (eq == 0u)
			return s + d;
		if (eq == 1u)
			return s - d;
		return d - s;
	}
	if (eq == 3u)
		return min(src.rgb, dst.rgb);
	if (eq == 4u)
		return max(src.rgb, dst.rgb);
	if (eq == 5u)
		return abs(src.rgb - dst.rgb);
	return src.rgb;
}

void DrawPixel(uint st, int x, int y, int z, int fog, ivec4 color) {
	const uint px0 = data[st + ST_PIXEL0];
	const uint px1 = data[st + ST_PIXEL1];
	const bool clearMode = USE_CLEAR && (px0 & PX0_CLEAR) != 0u;
	const bool earlyZ = (px0 & PX0_EARLYZ) != 0u;
	const uint fmt = (px0 >> PX0_FBFORMAT_SHIFT) & 3u;
	const uint refs = data[st + ST_REFS];
	color = clamp(color, ivec4(0), ivec4(255));

	if (USE_DEPTH && (px0 & PX0_DEPTHRANGE) != 0u && !earlyZ) {
		if (z < int(data[st + ST_MINZ]) || z > int(data[st + ST_MAXZ]))
			return;
	}

	const int alphaFunc = int((px0 >> PX0_ALPHATEST_SHIFT) & 7u);
	if (USE_ALPHATEST && alphaFunc != COMP_ALWAYS && !clearMode) {
		int a = color.a;
		if ((px0 & PX0_ALPHATESTMASK) != 0u)
			a &= int((refs >> 16) & 0xFFu);
		if (!Compare(alphaFunc, a, int(refs & 0xFFu)))
			return;
	}

	if (USE_FOG && (px0 & PX0_FOG) != 0u && !clearMode) {
		const ivec3 fogColor = Unpack(data[st + ST_FOGCOLOR]).rgb;
		color.rgb = (color.rgb * fog + fogColor * (255 - fog) + ivec3(255)) >> 8;
	}

	if (USE_COLORTEST && (px0 & PX0_COLORTEST) != 0u && !clearMode) {
		const uint c = PackRGB(color.rgb) & data[st + ST_CTEST_MASK];
		const uint ref = data[st + ST_CTEST_REF];
		const uint func = (data[st + ST_MISC] >> 16) & 0xFFu;
		if (func == uint(COMP_NEVER))
			return;
		if (func == uint(COMP_EQUAL) && c != ref)
			return;
		if (func == uint(COMP_NOTEQUAL) && c == ref)
			return;
	}

	const uint writeMask = USE_WRITEMASK && (px0 & PX0_WRITEMASK) != 0u ? data[st + ST_WRITEMASK] : 0u;
	const uint bpp = fmt == FMT_8888 ? 4u : 2u;
	const uint fbAddr = (pc.cacheFlags & CACHE_COLOR) != 0u ? 0u : FBAddr(st, x, y, bpp);
	const int depthFunc = int((px0 >> PX0_DEPTHTEST_SHIFT) & 7u);
	uint stencil = clearMode ? uint(color.a) : ReadStencil(fmt, fbAddr);
	if (clearMode) {
		if (USE_DEPTH && (px0 & PX0_DEPTHWRITE) != 0u)
			DepthWrite(st, x, y, uint(z));
	} else if (USE_STENCIL && (px0 & PX0_STENCILTEST) != 0u) {
		const uint stencilReplace = (px0 & PX0_STENCILTESTMASK) != 0u ? (data[st + ST_MISC] & 0xFFu) : ((refs >> 8) & 0xFFu);
		uint testStencil = stencil;
		if ((px0 & PX0_STENCILTESTMASK) != 0u)
			testStencil &= (refs >> 24) & 0xFFu;
		// The test is ref func stencil.
		if (!Compare(int((px0 >> PX0_STENCILTEST_SHIFT) & 7u), int((refs >> 8) & 0xFFu), int(testStencil))) {
			stencil = StencilOp(fmt, stencilReplace, (px1 >> PX1_SFAIL_SHIFT) & 7u, stencil);
			WriteStencil(fmt, fbAddr, writeMask, stencil);
			return;
		}
		if (USE_DEPTH && !earlyZ && depthFunc != COMP_ALWAYS && !DepthTestPassed(st, depthFunc, x, y, z & 0xFFFF)) {
			stencil = StencilOp(fmt, stencilReplace, (px1 >> PX1_ZFAIL_SHIFT) & 7u, stencil);
			WriteStencil(fmt, fbAddr, writeMask, stencil);
			return;
		}
		stencil = StencilOp(fmt, stencilReplace, (px1 >> PX1_ZPASS_SHIFT) & 7u, stencil);
	} else if (!earlyZ) {
		if (USE_DEPTH && depthFunc != COMP_ALWAYS && !DepthTestPassed(st, depthFunc, x, y, z & 0xFFFF))
			return;
	}

	if (USE_DEPTH && (px0 & PX0_DEPTHWRITE) != 0u && !clearMode)
		DepthWrite(st, x, y, uint(z));

	const uint oldColor = ReadColor(fmt, fbAddr);
	uint newColor;
	// Dithering happens before the logic op and regardless of framebuffer format or clear mode.
	const int dither = USE_DITHER && (px0 & PX0_DITHER) != 0u ? int(data[st + ST_DITHER + uint(y & 3)] << (24u - 8u * uint(x & 3))) >> 24 : 0;
	if (USE_BLEND && (px0 & PX0_ALPHABLEND) != 0u && !clearMode) {
		const ivec3 blended = AlphaBlend(st, color, Unpack(oldColor)) + ivec3(dither);
		newColor = PackRGB(blended) | (stencil << 24);
	} else {
		newColor = PackRGB(color.rgb + ivec3(dither)) | (stencil << 24);
	}

	if (USE_LOGICOP && (px0 & PX0_LOGICOP) != 0u && !clearMode)
		newColor = LogicOp((data[st + ST_MISC] >> 8) & 0xFFu, oldColor, newColor);

	if (clearMode) {
		if ((px0 & PX0_COLORTEST) == 0u)
			newColor = (newColor & 0xFF000000u) | (oldColor & 0x00FFFFFFu);
		if ((px0 & PX0_STENCILTEST) == 0u)
			newColor = (newColor & 0x00FFFFFFu) | (oldColor & 0xFF000000u);
	}

	WriteColor(fmt, fbAddr, newColor, oldColor, writeMask);
}

// Primitives. Each type works out a pixel's fragment (FragmentAt), up to texturing; DrawPrim does the rest,
// so texturing and the pixel pipeline are in the shader once.

struct Frag {
	int z;
	int fog;
	ivec4 color;
	// Added after texturing, unless in clear mode.
	ivec3 sec;
	bool textured;
	uint s;
	uint t;
	int level;
	int levelFrac;
	bool bilinear;
};

bool DoubleSecondary(uint st) {
	const uint samplerBits = data[st + ST_SAMPLER];
	return (samplerBits & SM_TEXTURES) != 0u && (samplerBits & SM_DOUBLING) != 0u;
}

bool TriangleCovers(uint d, bool e32, int x, int yc) {
	const int xc = x * 16 + 8;
	return EdgeInside(d, e32, xc, yc) && EdgeInside(d + 4u, e32, xc, yc) && EdgeInside(d + 8u, e32, xc, yc);
}

bool TriangleFragment(uint prim, uint st, int x, int y, out Frag f) {
	const uint flags = data[prim + PRIM_TYPE_FLAGS];
	const uint d = prim + PRIM_HEADER_WORDS;
	const bool e32 = (flags & PRIMF_EDGES32) != 0u;
	const bool p32 = (flags & PRIMF_PLANES32) != 0u;
	const int xc = x * 16 + 8, yc = y * 16 + 8;
)",
R"(
	const uint snap = data[d + TRI_SNAP];
	const int snapEdge = int(snap << 24) >> 24;
	for (int k = 0; k < 3; ++k) {
		if (k != snapEdge && !EdgeInside(d + TRI_EDGES + uint(k) * 4u, e32, xc, yc))
			return false;
	}
	if (snapEdge >= 0) {
		// The span's first pixel (snapping left) or last is checked at the span's other end.
		int at = x;
		if ((snap & 0x100u) != 0u)
			at = (x & 3) == 0 ? (x | 3) : x;
		else
			at = (x & 3) == 3 ? (x & ~3) : x;
		if (!EdgeInside(d + TRI_EDGES + uint(snapEdge) * 4u, e32, at * 16 + 8, yc))
			return false;
	}

	uint plane = d + TRI_PLANES;
	if ((flags & PRIMF_FLATZ) != 0u) {
		f.z = int(data[d + TRI_Z]);
	} else {
		// A value floored below 0 is 0.
		f.z = max(PlaneValue(plane, p32, xc, yc), 0);
		plane += PLANE_WORDS;
	}
	if (!EarlyZPassed(st, x, y, f.z))
		return false;

	f.fog = 255;
	if ((flags & PRIMF_NOFOG) == 0u) {
		f.fog = clamp(PlaneValue(plane, p32, xc, yc), 0, 255);
		plane += PLANE_WORDS;
	}
	if ((flags & PRIMF_FLAT_C0) != 0u) {
		f.color = Unpack(data[d + TRI_C0]);
	} else {
		for (int c = 0; c < 4; ++c) {
			f.color[c] = clamp(PlaneValue(plane, p32, xc, yc), 0, 255);
			plane += PLANE_WORDS;
		}
	}
	if ((flags & PRIMF_FLAT_C1) != 0u) {
		f.sec = Unpack(data[d + TRI_C1]).rgb;
	} else {
		for (int c = 0; c < 3; ++c) {
			f.sec[c] = clamp(PlaneValue(plane, p32, xc, yc), 0, 255);
			plane += PLANE_WORDS;
		}
	}
	if (DoubleSecondary(st))
		f.sec *= 2;

	f.textured = USE_TEXTURE && (flags & PRIMF_UV) != 0u;
	if (f.textured) {
		const int expS = int(data[d + TRI_EXP_S]), expT = int(data[d + TRI_EXP_T]), expQ = int(data[d + TRI_EXP_Q]);
		uint q;
		if (!USE_WIDE || p32)
			UVFromPlanes32(PlaneValue(plane, true, xc, yc), PlaneValue(plane + 6u, true, xc, yc), PlaneValue(plane + 12u, true, xc, yc), expQ, expS, expT, f.s, f.t, q);
		else
			UVFromPlanes(PlaneValue64(plane, false, xc, yc), PlaneValue64(plane + 6u, false, xc, yc), PlaneValue64(plane + 12u, false, xc, yc), expQ, expS, expT, f.s, f.t, q);
		if ((flags & PRIMF_LODQ) != 0u) {
			// The q the mip level comes from: one per span of four pixels, at the span's second pixel in the
			// direction the row is walked, or else its first pixel inside (Rasterizer's LodQFromPlanes).
			const bool rtl = (flags & PRIMF_Q_RTL) != 0u;
			const int spanX = x & ~3;
			int pick = spanX + (rtl ? 2 : 1);
			if (!TriangleCovers(d + TRI_EDGES, e32, pick, yc)) {
				for (int i = 0; i < 4; ++i) {
					const int px = rtl ? spanX + 3 - i : spanX + i;
					if (TriangleCovers(d + TRI_EDGES, e32, px, yc)) {
						pick = px;
						break;
					}
				}
			}
			q = !USE_WIDE || p32 ? ToFloat24Fast(PlaneValue(plane, true, pick * 16 + 8, yc), expQ) : ToFloat24(PlaneValue64(plane, false, pick * 16 + 8, yc), expQ);
		}
		if ((flags & PRIMF_THROUGH_UV) != 0u) {
			const uint shifts = data[st + ST_TEXLEVEL];
			f.s = ScaleDownBits(f.s, int((shifts >> 16) & 0xFFu));
			f.t = ScaleDownBits(f.t, int(shifts >> 24));
		}
		int detail = int(data[d + TRI_LOD_BASE]);
		if ((flags & PRIMF_LOD_USES_Q) != 0u)
			detail -= GELog16(q);
		SamplingParams(st, detail, f.level, f.levelFrac, f.bilinear);
	}
	return true;
}

bool RectFragment(uint prim, uint st, int x, int y, out Frag f) {
	const uint flags = data[prim + PRIM_TYPE_FLAGS];
	const uint d = prim + PRIM_HEADER_WORDS;
	f.z = int(data[d + RECT_Z]);
	if (!EarlyZPassed(st, x, y, f.z))
		return false;
	f.fog = int(data[d + RECT_FOG]);
	f.color = Unpack(data[d + RECT_C0]);
	f.sec = Unpack(data[d + RECT_C1]).rgb;
	if (DoubleSecondary(st))
		f.sec *= 2;
	f.textured = USE_TEXTURE && (flags & PRIMF_UV) != 0u;
	if (f.textured) {
		const bool p32 = (flags & PRIMF_PLANES32) != 0u;
		const int xc = x * 16 + 8, yc = y * 16 + 8;
		const uint plane = d + RECT_PLANES;
		uint q;
		if (!USE_WIDE || p32)
			UVFromPlanes32(PlaneValue(plane, true, xc, yc), PlaneValue(plane + 6u, true, xc, yc), PlaneValue(plane + 12u, true, xc, yc), int(data[d + RECT_EXP_Q]), int(data[d + RECT_EXP_S]), int(data[d + RECT_EXP_T]), f.s, f.t, q);
		else
			UVFromPlanes(PlaneValue64(plane, false, xc, yc), PlaneValue64(plane + 6u, false, xc, yc), PlaneValue64(plane + 12u, false, xc, yc), int(data[d + RECT_EXP_Q]), int(data[d + RECT_EXP_S]), int(data[d + RECT_EXP_T]), f.s, f.t, q);
		int detail = int(data[d + RECT_LOD_BASE]);
		if ((flags & PRIMF_LOD_USES_Q) != 0u)
			detail -= GELog16(q);
		SamplingParams(st, detail, f.level, f.levelFrac, f.bilinear);
	}
	return true;
}

bool SpriteFragment(uint prim, uint st, int x, int y, out Frag f) {
	const uint flags = data[prim + PRIM_TYPE_FLAGS];
	const uint d = prim + PRIM_HEADER_WORDS;
	f.z = int(data[d + SPRITE_Z]);
	if (!EarlyZPassed(st, x, y, f.z))
		return false;
	f.fog = 255;
	f.color = Unpack(data[d + SPRITE_C0]);
	f.sec = ivec3(0);
	f.textured = USE_TEXTURE && (flags & PRIMF_TEXTURED) != 0u;
	if (f.textured) {
		// The texel 1:1, at level 0: as a coordinate, its center comes out as that texel.
		const uint xy1 = data[prim + PRIM_XY1];
		const int u = int(data[d + SPRITE_S]) + (x - int(xy1 & 0xFFFFu)) * int(data[d + SPRITE_DS]);
		const int v = int(data[d + SPRITE_T]) + (y - int(xy1 >> 16)) * int(data[d + SPRITE_DT]);
		const uint shifts = data[st + ST_TEXLEVEL];
		f.s = floatBitsToUint(float(u) * exp2(-float((shifts >> 16) & 0xFFu)));
		f.t = floatBitsToUint(float(v) * exp2(-float(shifts >> 24)));
		f.level = 0;
		f.levelFrac = 0;
		f.bilinear = false;
	}
	return true;
}

// The next of a FRAGMENTS primitive's fragments at the pixel, from index on.
bool NextFragment(uint prim, uint st, int x, int y, inout uint index, out Frag f) {
	const uint d = prim + PRIM_HEADER_WORDS;
	const uint count = data[d + FRAG_COUNT];
	const uint xy = uint(x) | (uint(y) << 16);
	while (index < count) {
		const uint frag = d + FRAG_FIRST + index * FRAG_WORDS;
		index++;
		if (data[frag + FRAGW_XY] != xy)
			continue;
		f.z = int(data[frag + FRAGW_Z]);
		if (!EarlyZPassed(st, x, y, f.z))
			continue;
		const uint fogFlags = data[frag + FRAGW_FOG_FLAGS];
		f.fog = int(fogFlags & 0xFFu);
		f.color = Unpack(data[frag + FRAGW_C0]);
		const uint sec = data[frag + FRAGW_C1];
		f.sec = ivec3(int(sec & 0x3FFu), int((sec >> 10) & 0x3FFu), int((sec >> 20) & 0x3FFu));
		f.textured = USE_TEXTURE && (fogFlags & 0x100u) != 0u;
		f.s = data[frag + FRAGW_S];
		f.t = data[frag + FRAGW_T];
		f.level = int((fogFlags >> 16) & 0xFFu);
		f.levelFrac = int(fogFlags >> 24);
		f.bilinear = (fogFlags & 0x200u) != 0u;
		return true;
	}
	return false;
}

void DrawClearRect(uint prim, uint st, int x, int y) {
	const uint d = prim + PRIM_HEADER_WORDS;
	const uint px0 = data[st + ST_PIXEL0];
	const uint fmt = (px0 >> PX0_FBFORMAT_SHIFT) & 3u;
	if (USE_DEPTH && (px0 & PX0_DEPTHWRITE) != 0u)
		DepthWrite(st, x, y, data[d + CLEAR_Z]);

	// ClearRectangle's mask of what to keep.
	const bool colorClear = (px0 & PX0_COLORTEST) != 0u;
	const bool stencilClear = (px0 & PX0_STENCILTEST) != 0u;
	uint keepOldMask = 0xFFFFFFFFu;
	if (colorClear && stencilClear) {
		keepOldMask = 0u;
	} else if (fmt == FMT_565) {
		if (colorClear)
			keepOldMask = 0u;
	} else if (fmt == FMT_5551) {
		if (colorClear)
			keepOldMask = 0xFFFF8000u;
		else if (stencilClear)
			keepOldMask = 0xFFFF7FFFu;
	} else if (fmt == FMT_4444) {
		if (colorClear)
			keepOldMask = 0xFFFFF000u;
		else if (stencilClear)
			keepOldMask = 0xFFFF0FFFu;
	} else {
		if (colorClear)
			keepOldMask = 0xFF000000u;
		else if (stencilClear)
			keepOldMask = 0x00FFFFFFu;
	}
	if (USE_WRITEMASK && (px0 & PX0_WRITEMASK) != 0u)
		keepOldMask |= data[st + ST_WRITEMASK];
	if (keepOldMask == 0xFFFFFFFFu)
		return;

	const uint newColor = data[d + CLEAR_COLOR];
	if (fmt == FMT_8888) {
		const uint addr = (pc.cacheFlags & CACHE_COLOR) != 0u ? 0u : FBAddr(st, x, y, 4u);
		FBWrite32(addr, (FBRead32(addr) & keepOldMask) | (newColor & ~keepOldMask));
	} else {
		uint newColor16;
		if (fmt == FMT_565)
			newColor16 = RGBA8888ToRGB565(newColor);
		else if (fmt == FMT_5551)
			newColor16 = RGBA8888ToRGBA5551(newColor);
		else
			newColor16 = RGBA8888ToRGBA4444(newColor);
		const uint addr = (pc.cacheFlags & CACHE_COLOR) != 0u ? 0u : FBAddr(st, x, y, 2u);
		FBWrite16(addr, (FBRead16(addr) & keepOldMask) | (newColor16 & ~keepOldMask));
	}
}

void DrawPrim(uint prim, int x, int y) {
	const uint xy1 = data[prim + PRIM_XY1];
	const uint xy2 = data[prim + PRIM_XY2];
	if (x < int(xy1 & 0xFFFFu) || x > int(xy2 & 0xFFFFu) || y < int(xy1 >> 16) || y > int(xy2 >> 16))
		return;
	const uint st = pc.statesOffset + data[prim + PRIM_STATE] * ST_WORDS;
	const uint type = data[prim + PRIM_TYPE_FLAGS] & 0xFu;
	if (USE_CLEAR && type == 3u) {
		DrawClearRect(prim, st, x, y);
		return;
	}
	const bool clearMode = USE_CLEAR && (data[st + ST_PIXEL0] & PX0_CLEAR) != 0u;
	uint index = 0u;
	[[dont_unroll]]
	while (true) {
		Frag f;
		bool any;
		if (type == 0u)
			any = index++ == 0u && TriangleFragment(prim, st, x, y, f);
		else if (USE_RECTS && type == 1u)
			any = index++ == 0u && RectFragment(prim, st, x, y, f);
		else if (USE_RECTS && type == 2u)
			any = index++ == 0u && SpriteFragment(prim, st, x, y, f);
		else if (USE_FRAGMENTS)
			any = NextFragment(prim, st, x, y, index, f);
		else
			any = false;
		if (!any)
			break;
		if (USE_TEXTURE && f.textured)
			f.color = ApplyTexturing(st, f.color, f.s, f.t, f.level, f.levelFrac, f.bilinear);
		if (!clearMode)
			f.color.rgb += f.sec;
		DrawPixel(st, x, y, f.z, f.fog, f.color);
	}
}

void main() {
	const uint tile = pc.tilesOffset + gl_WorkGroupID.x * 3u;
	const uint tileXY = data[tile];
	const uint itemStart = pc.itemsOffset + data[tile + 1u];
	const uint itemCount = data[tile + 2u];
	const int x = int(tileXY & 0xFFFFu) * 16 + int(gl_LocalInvocationID.x) * 2;
	const int y = int(tileXY >> 16) * 16 + int(gl_LocalInvocationID.y);

	const uint cst = pc.cacheState;
	const bool color32 = ((data[cst + ST_PIXEL0] >> PX0_FBFORMAT_SHIFT) & 3u) == FMT_8888;
	[[dont_unroll]]
	for (int p = 0; p < 2; ++p) {
		gColorDirty[p] = false;
		gDepthDirty[p] = false;
		if ((pc.cacheFlags & CACHE_COLOR) != 0u) {
			const uint addr = FBAddr(cst, x + p, y, color32 ? 4u : 2u);
			gColor[p] = color32 ? MemRead32(addr) : MemRead16(addr);
		}
		if ((pc.cacheFlags & CACHE_DEPTH) != 0u)
			gDepth[p] = MemRead16(DepthAddr(cst, x + p, y));
	}

	for (uint i = 0u; i < itemCount; ++i) {
		const uint prim = pc.primsOffset + data[itemStart + i];
		[[dont_unroll]]
		for (int p = 0; p < 2; ++p) {
			gPix = p;
			DrawPrim(prim, x + p, y);
		}
	}

	[[dont_unroll]]
	for (int p = 0; p < 2; ++p) {
		if (gColorDirty[p]) {
			const uint addr = FBAddr(cst, x + p, y, color32 ? 4u : 2u);
			if (color32)
				mem[addr >> 2] = gColor[p];
			else
				AtomicWrite16(addr, gColor[p]);
		}
		if (gDepthDirty[p])
			AtomicWrite16(DepthAddr(cst, x + p, y), gDepth[p]);
	}
}

)",
};
