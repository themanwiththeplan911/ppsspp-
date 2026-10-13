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

#include "ppsspp_config.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "Common/Data/Text/StringWriter.h"
#include "Common/GPU/thin3d.h"
#include "Common/GPU/Vulkan/VulkanContext.h"
#include "Common/GPU/Vulkan/VulkanRenderManager.h"
#include "Common/Log.h"
#include "Common/StringUtils.h"
#include "Common/TimeUtil.h"
#include "Core/Config.h"
#include "Core/MemMap.h"
#include "GPU/Common/TextureDecoder.h"
#include "GPU/GPUState.h"
#include "GPU/Software/ComputeRaster.h"
#include "GPU/Software/GEMath.h"
#include "GPU/Software/Rasterizer.h"
#include "GPU/Software/SoftGpu.h"
#include "GPU/Vulkan/ComputeRasterShader.h"
#include "GPU/Vulkan/VulkanUtil.h"

namespace SoftCompute {

// The GPU's copy of memory is tracked in pages: uploaded when the CPU changed one the drawing reads, and copied
// back when the GPU drew to one.
static constexpr uint32_t PAGE_SHIFT = 12;
static constexpr uint32_t PAGE_SIZE = 1 << PAGE_SHIFT;
// Self-texturing snapshots, and data from the CPU (CLUTs, textures that aren't in PSP memory).
static constexpr uint32_t SNAPSHOT_SIZE = 4 * 1024 * 1024;
static constexpr uint32_t SCRATCH_SIZE = 8 * 1024 * 1024;
// A depth buffer's layout moves bytes within 64 KB (GPU/Common/DepthSwizzle.h), so its pages are tracked by that.
static constexpr uint32_t DEPTH_BLOCK = 64 * 1024;
static constexpr int TILE_SHIFT = 4;
static constexpr int TILES_X = 1024 >> TILE_SHIFT;
static constexpr int TILES_Y = 1024 >> TILE_SHIFT;
// Draw what's queued once a batch gets this large, without waiting for a flush.
static constexpr size_t MAX_BATCH_PRIM_WORDS = 4 * 1024 * 1024;
static constexpr size_t MAX_BATCH_UPLOAD_BYTES = 32 * 1024 * 1024;
// Timestamps, two per batch between readbacks.
static constexpr uint32_t MAX_QUERIES = 256;

struct PushParams {
	uint32_t statesOffset;
	uint32_t primsOffset;
	uint32_t tilesOffset;
	uint32_t itemsOffset;
	uint32_t memBytes;
};

static void AddDefine(std::string &s, const char *name, uint32_t value) {
	s += StringFromFormat("#define %s %uu\n", name, value);
}

// The layout from ComputeRaster.h, for the shader.
static std::string ShaderSource() {
	std::string s = "#version 450\n#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
#define DEFINE(x) AddDefine(s, #x, (uint32_t)(x))
	DEFINE(PRIM_TYPE_FLAGS); DEFINE(PRIM_STATE); DEFINE(PRIM_XY1); DEFINE(PRIM_XY2); DEFINE(PRIM_HEADER_WORDS);
	DEFINE(PRIMF_FLATZ); DEFINE(PRIMF_FLAT_C0); DEFINE(PRIMF_FLAT_C1); DEFINE(PRIMF_NOFOG); DEFINE(PRIMF_UV);
	DEFINE(PRIMF_EDGES32); DEFINE(PRIMF_PLANES32); DEFINE(PRIMF_LODQ); DEFINE(PRIMF_Q_RTL); DEFINE(PRIMF_THROUGH_UV);
	DEFINE(PRIMF_LOD_USES_Q); DEFINE(PRIMF_TEXTURED);
	DEFINE(TRI_EDGES); DEFINE(TRI_SNAP); DEFINE(TRI_Z); DEFINE(TRI_C0); DEFINE(TRI_C1); DEFINE(TRI_LOD_BASE);
	DEFINE(TRI_EXP_S); DEFINE(TRI_EXP_T); DEFINE(TRI_EXP_Q); DEFINE(TRI_PLANES); DEFINE(PLANE_WORDS);
	DEFINE(RECT_Z); DEFINE(RECT_FOG); DEFINE(RECT_C0); DEFINE(RECT_C1); DEFINE(RECT_LOD_BASE); DEFINE(RECT_EXP_S);
	DEFINE(RECT_EXP_T); DEFINE(RECT_EXP_Q); DEFINE(RECT_PLANES);
	DEFINE(SPRITE_S); DEFINE(SPRITE_T); DEFINE(SPRITE_DS); DEFINE(SPRITE_DT); DEFINE(SPRITE_Z); DEFINE(SPRITE_C0);
	DEFINE(CLEAR_Z); DEFINE(CLEAR_COLOR);
	DEFINE(FRAG_COUNT); DEFINE(FRAG_FIRST); DEFINE(FRAGW_XY); DEFINE(FRAGW_Z); DEFINE(FRAGW_FOG_FLAGS); DEFINE(FRAGW_C0);
	DEFINE(FRAGW_C1); DEFINE(FRAGW_S); DEFINE(FRAGW_T); DEFINE(FRAG_WORDS);
	DEFINE(ST_PIXEL0); DEFINE(ST_PIXEL1); DEFINE(ST_REFS); DEFINE(ST_MISC); DEFINE(ST_WRITEMASK); DEFINE(ST_DITHER);
	DEFINE(ST_FOGCOLOR); DEFINE(ST_MINZ); DEFINE(ST_MAXZ); DEFINE(ST_STRIDES); DEFINE(ST_CTEST_MASK); DEFINE(ST_CTEST_REF);
	DEFINE(ST_FIXA); DEFINE(ST_FIXB); DEFINE(ST_FB_ADDR); DEFINE(ST_Z_BASE); DEFINE(ST_Z_ROTMASK); DEFINE(ST_Z_ROTSHIFT);
	DEFINE(ST_Z_XOR); DEFINE(ST_SAMPLER); DEFINE(ST_CLUTFORMAT); DEFINE(ST_TEXENV); DEFINE(ST_TEXLEVEL); DEFINE(ST_CLUT_ADDR);
	DEFINE(ST_LEVELS); DEFINE(ST_WORDS);
	DEFINE(PX0_CLEAR); DEFINE(PX0_COLORTEST); DEFINE(PX0_STENCILTEST); DEFINE(PX0_DEPTHWRITE); DEFINE(PX0_DEPTHRANGE);
	DEFINE(PX0_ALPHABLEND); DEFINE(PX0_ALPHATESTMASK); DEFINE(PX0_STENCILTESTMASK); DEFINE(PX0_DITHER); DEFINE(PX0_LOGICOP);
	DEFINE(PX0_FOG); DEFINE(PX0_WRITEMASK); DEFINE(PX0_EARLYZ); DEFINE(PX0_FBFORMAT_SHIFT); DEFINE(PX0_ALPHATEST_SHIFT);
	DEFINE(PX0_DEPTHTEST_SHIFT); DEFINE(PX0_STENCILTEST_SHIFT);
	DEFINE(PX1_BLENDEQ_SHIFT); DEFINE(PX1_BLENDSRC_SHIFT); DEFINE(PX1_BLENDDST_SHIFT); DEFINE(PX1_SFAIL_SHIFT);
	DEFINE(PX1_ZFAIL_SHIFT); DEFINE(PX1_ZPASS_SHIFT);
	DEFINE(SM_TEXFMT_SHIFT); DEFINE(SM_CLUTFMT_SHIFT); DEFINE(SM_CLAMPS); DEFINE(SM_CLAMPT); DEFINE(SM_SWIZZLE);
	DEFINE(SM_SHAREDCLUT); DEFINE(SM_TEXFUNC_SHIFT); DEFINE(SM_TEXALPHA); DEFINE(SM_DOUBLING); DEFINE(SM_TEXTURES);
	DEFINE(SM_GOURAUD); DEFINE(SM_THROUGH); DEFINE(SM_MIPFILT); DEFINE(SM_MINFILT); DEFINE(SM_MAGFILT);
	DEFINE(SM_LEVELMODE_SHIFT); DEFINE(SM_CLUTTRANSFORM); DEFINE(SM_FORCE_LINEAR); DEFINE(SM_FORCE_NEAREST);
	DEFINE(LV_VALID); DEFINE(LV_VRAM);
#undef DEFINE
	s += "const ivec2 GE_RECIP[128] = ivec2[128](";
	for (int i = 0; i < 128; ++i)
		s += StringFromFormat("%sivec2(%d, %d)", i == 0 ? "" : ", ", geRecipSegments[i].b, geRecipSegments[i].m);
	s += ");\n";
	for (const char *part : computeRasterShaderParts)
		s += part;
	return s;
}

class ComputeRasterVulkan : public Backend {
public:
	explicit ComputeRasterVulkan(Draw::DrawContext *draw);
	~ComputeRasterVulkan();
	bool Init();

	uint32_t PackState(const Rasterizer::RasterizerState &state, bool snapshot) override;
	uint32_t BatchId() const override {
		return batchId_;
	}
	std::vector<uint32_t> &Prims() override {
		return prims_;
	}
	void CommitPrim(uint32_t offset, const Rasterizer::RasterizerState &state) override;
	void EndSegment() override;
	void BeginSnapshot() override;
	void SnapshotLevel(int level, uint32_t addr, uint32_t bytes) override;
	void Flush(bool readback) override;
	bool HasGPUWrites(uint32_t addr, uint32_t bytes) const override;
	bool AnyGPUWrites() const override {
		return anyGPUDirty_;
	}
	void GetStats(StringWriter &w) override;

private:
	struct Buffer {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation alloc = VK_NULL_HANDLE;
		uint8_t *mapped = nullptr;
		size_t size = 0;
	};
	struct Op {
		enum class Type {
			SEGMENT,
			COPY,
		};
		Type type;
		// SEGMENT: primitives first to end (indices in primOffsets_), then where its tiles are.
		uint32_t first;
		uint32_t end;
		uint32_t tileFirst;
		uint32_t tileCount;
		// COPY: within the memory buffer.
		uint32_t src;
		uint32_t dst;
		uint32_t size;
	};
	struct Upload {
		uint32_t memOffset;
		uint32_t size;
		// In uploadData_.
		size_t dataOffset;
	};

	bool CreateBuffer(Buffer &buf, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage, bool map);
	void DestroyBuffer(Buffer &buf);
	bool UpdateDescriptorSet();
	uint8_t *AllocStaging(size_t bytes, VkBuffer *buffer, VkDeviceSize *offset);

	void Submit();
	void ResetBatch();
	template <typename F>
	void ForPrimTiles(uint32_t off, F f);
	void VerifyPages(uint32_t memOffset, uint32_t bytes);
	void VerifyVRAM(uint32_t vramOffset, uint32_t bytes);
	void MarkGPUWrite(uint32_t vramOffset, uint32_t bytes);
	const uint8_t *PageSource(uint32_t page) const;
	void AddUpload(uint32_t memOffset, const void *src, uint32_t bytes);
	uint32_t HostData(const void *src, uint32_t bytes, uint32_t padTo);
	void ResolveLevel(const Rasterizer::RasterizerState &state, int level, bool snapshot, uint32_t *out);

	VulkanContext *vulkan_ = nullptr;
	VulkanRenderManager *renderManager_ = nullptr;

	VkDescriptorSetLayout descSetLayout_ = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
	VkPipeline pipeline_ = VK_NULL_HANDLE;
	VkDescriptorPool descPool_ = VK_NULL_HANDLE;
	VkDescriptorSet descSet_ = VK_NULL_HANDLE;
	int descSetsLeft_ = 0;

	// VRAM, RAM, the snapshots and the scratch area, as the shader reads and draws them.
	Buffer mem_;
	// States, primitives and tile lists of a batch.
	Buffer data_;
	Buffer readback_;
	Buffer staging_[VulkanContext::MAX_INFLIGHT_FRAMES];
	size_t stagingUsed_[VulkanContext::MAX_INFLIGHT_FRAMES]{};
	int stagingFrame_ = -1;

	uint32_t ramSize_ = 0;
	uint32_t ramOffset_ = 0;
	uint32_t snapshotOffset_ = 0;
	uint32_t scratchOffset_ = 0;
	uint32_t memBytes_ = 0;

	// What the GPU's copy of VRAM and RAM holds (or will, after the uploads queued), at the same offsets.
	std::vector<uint8_t> shadow_;
	// Pages checked against PSP memory since the last flush.
	std::vector<uint32_t> verified_;
	// VRAM pages the GPU may have drawn to since the last flush.
	std::vector<uint32_t> gpuDirty_;
	bool anyGPUDirty_ = false;
	// The last ranges verified and marked as drawn to, which the next primitive likely repeats. Valid while gen
	// matches verifyGen_, which a flush advances.
	struct LastRange {
		uint32_t start;
		uint32_t end;
		uint32_t gen;
	};
	LastRange lastVerified_{};
	LastRange lastMarked_{};
	uint32_t verifyGen_ = 1;

	// The batch being built.
	uint32_t batchId_ = 1;
	std::vector<uint32_t> states_;
	std::vector<uint32_t> prims_;
	std::vector<uint32_t> primOffsets_;
	uint32_t segmentStart_ = 0;
	std::vector<Op> ops_;
	std::vector<Upload> uploads_;
	std::vector<uint8_t> uploadData_;
	uint32_t scratchUsed_ = 0;
	std::unordered_map<const void *, uint32_t> hostData_;
	// The current self-texturing snapshot, per level.
	uint32_t snapUsed_ = 0;
	uint32_t snapOffset_[8]{};
	bool snapValid_[8]{};

	// Tile binning scratch.
	std::vector<uint32_t> tileCount_;
	std::vector<uint32_t> tilePos_;
	std::vector<uint16_t> tilesTouched_;
	std::vector<uint32_t> tileData_;
	// Pairs of a primitive's offset and a tile it's in, for one segment.
	std::vector<uint64_t> tilePrims_;

	// Stats since the last GetStats.
	int statPrims_ = 0;
	int statBatches_ = 0;
	int statSegments_ = 0;
	int statFlushes_ = 0;
	size_t statUploadBytes_ = 0;
	size_t statReadbackBytes_ = 0;
	double statSyncTime_ = 0.0;
	// GPU time of the batches, from timestamps around each (read at the next readback).
	VkQueryPool queryPool_ = VK_NULL_HANDLE;
	uint32_t queriesUsed_ = 0;
	double statGPUTime_ = 0.0;
	double timestampPeriod_ = 0.0;
};

ComputeRasterVulkan::ComputeRasterVulkan(Draw::DrawContext *draw) {
	vulkan_ = (VulkanContext *)draw->GetNativeObject(Draw::NativeObject::CONTEXT);
	renderManager_ = (VulkanRenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
}

bool ComputeRasterVulkan::CreateBuffer(Buffer &buf, size_t size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage, bool map) {
	VkBufferCreateInfo b{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	b.size = size;
	b.usage = usage;
	b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocCreateInfo{};
	allocCreateInfo.usage = memUsage;
	if (map)
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo allocInfo{};
	VkResult res = vmaCreateBuffer(vulkan_->Allocator(), &b, &allocCreateInfo, &buf.buffer, &buf.alloc, &allocInfo);
	if (res != VK_SUCCESS) {
		ERROR_LOG(Log::G3D, "Software compute: failed to allocate a buffer of %d bytes", (int)size);
		buf.buffer = VK_NULL_HANDLE;
		buf.alloc = VK_NULL_HANDLE;
		return false;
	}
	buf.mapped = (uint8_t *)allocInfo.pMappedData;
	buf.size = size;
	return true;
}

void ComputeRasterVulkan::DestroyBuffer(Buffer &buf) {
	if (buf.buffer != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeleteBufferAllocation(buf.buffer, buf.alloc);
	buf.mapped = nullptr;
	buf.size = 0;
}

bool ComputeRasterVulkan::Init() {
	if (!vulkan_ || !renderManager_)
		return false;
	if (!vulkan_->GetDeviceFeatures().enabled.standard.shaderInt64) {
		WARN_LOG(Log::G3D, "Software compute: the device doesn't support 64-bit integers in shaders");
		return false;
	}

	ramSize_ = std::max(Memory::g_MemorySize, (u32)0x02000000);
	ramOffset_ = MEM_RAM_OFFSET;
	snapshotOffset_ = ramOffset_ + ramSize_;
	scratchOffset_ = snapshotOffset_ + SNAPSHOT_SIZE;
	memBytes_ = scratchOffset_ + SCRATCH_SIZE;
	if (vulkan_->GetPhysicalDeviceProperties().properties.limits.maxStorageBufferRange < memBytes_) {
		WARN_LOG(Log::G3D, "Software compute: storage buffers too small");
		return false;
	}

	const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	if (!CreateBuffer(mem_, memBytes_, storage, VMA_MEMORY_USAGE_GPU_ONLY, false))
		return false;
	if (!CreateBuffer(data_, 4 * 1024 * 1024, storage, VMA_MEMORY_USAGE_GPU_ONLY, false))
		return false;
	if (!CreateBuffer(readback_, MEM_VRAM_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, true))
		return false;

	VkDevice device = vulkan_->GetDevice();
	VkDescriptorSetLayoutBinding bindings[2]{};
	for (int i = 0; i < 2; ++i) {
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dsl{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	dsl.bindingCount = 2;
	dsl.pBindings = bindings;
	if (vkCreateDescriptorSetLayout(device, &dsl, nullptr, &descSetLayout_) != VK_SUCCESS)
		return false;
	VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushParams) };
	VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pl.setLayoutCount = 1;
	pl.pSetLayouts = &descSetLayout_;
	pl.pushConstantRangeCount = 1;
	pl.pPushConstantRanges = &push;
	if (vkCreatePipelineLayout(device, &pl, nullptr, &pipelineLayout_) != VK_SUCCESS)
		return false;

	std::string error;
	const std::string source = ShaderSource();
	VkShaderModule module = CompileShaderModule(vulkan_, VK_SHADER_STAGE_COMPUTE_BIT, source.c_str(), &error);
	if (!module) {
		ERROR_LOG(Log::G3D, "Software compute: shader failed to compile: %s", error.c_str());
		return false;
	}
	VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pci.stage.module = module;
	pci.stage.pName = "main";
	pci.layout = pipelineLayout_;
	VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_);
	vulkan_->Delete().QueueDeleteShaderModule(module);
	if (res != VK_SUCCESS)
		return false;
	if (!UpdateDescriptorSet())
		return false;

	if (vulkan_->GetPhysicalDeviceProperties().properties.limits.timestampComputeAndGraphics) {
		VkQueryPoolCreateInfo qp{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
		qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
		qp.queryCount = MAX_QUERIES;
		if (vkCreateQueryPool(device, &qp, nullptr, &queryPool_) != VK_SUCCESS)
			queryPool_ = VK_NULL_HANDLE;
		timestampPeriod_ = vulkan_->GetPhysicalDeviceProperties().properties.limits.timestampPeriod * 1e-9;
	}

	shadow_.assign(ramOffset_ + ramSize_, 0);
	verified_.assign(((ramOffset_ + ramSize_) >> PAGE_SHIFT) / 32 + 1, 0);
	gpuDirty_.assign((MEM_VRAM_SIZE >> PAGE_SHIFT) / 32, 0);
	tileCount_.assign(TILES_X * TILES_Y, 0);
	tilePos_.assign(TILES_X * TILES_Y, 0);

	// The copy starts out zero, as the shadow does.
	VkCommandBuffer cmd = renderManager_->GetInitCmd();
	vkCmdFillBuffer(cmd, mem_.buffer, 0, VK_WHOLE_SIZE, 0);
	VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

	INFO_LOG(Log::G3D, "Software compute rasterizer ready, %d MB of GPU memory", (int)(memBytes_ >> 20));
	return true;
}

ComputeRasterVulkan::~ComputeRasterVulkan() {
	if (!vulkan_)
		return;
	DestroyBuffer(mem_);
	DestroyBuffer(data_);
	DestroyBuffer(readback_);
	for (Buffer &buf : staging_)
		DestroyBuffer(buf);
	if (pipeline_ != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeletePipeline(pipeline_);
	if (pipelineLayout_ != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeletePipelineLayout(pipelineLayout_);
	if (descSetLayout_ != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeleteDescriptorSetLayout(descSetLayout_);
	if (descPool_ != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeleteDescriptorPool(descPool_);
	if (queryPool_ != VK_NULL_HANDLE)
		vulkan_->Delete().QueueDeleteQueryPool(queryPool_);
}

// The set binds whole buffers, so it changes when one does. Sets already recorded are left alone (the GPU may
// still use them), and new ones come from a pool replaced when it runs out.
bool ComputeRasterVulkan::UpdateDescriptorSet() {
	VkDevice device = vulkan_->GetDevice();
	if (descSetsLeft_ == 0) {
		if (descPool_ != VK_NULL_HANDLE)
			vulkan_->Delete().QueueDeleteDescriptorPool(descPool_);
		constexpr int SETS = 64;
		VkDescriptorPoolSize size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, SETS * 2 };
		VkDescriptorPoolCreateInfo dp{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		dp.maxSets = SETS;
		dp.poolSizeCount = 1;
		dp.pPoolSizes = &size;
		if (vkCreateDescriptorPool(device, &dp, nullptr, &descPool_) != VK_SUCCESS)
			return false;
		descSetsLeft_ = SETS;
	}
	VkDescriptorSetAllocateInfo alloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	alloc.descriptorPool = descPool_;
	alloc.descriptorSetCount = 1;
	alloc.pSetLayouts = &descSetLayout_;
	if (vkAllocateDescriptorSets(device, &alloc, &descSet_) != VK_SUCCESS)
		return false;
	descSetsLeft_--;

	VkDescriptorBufferInfo infos[2]{};
	infos[0].buffer = mem_.buffer;
	infos[0].range = VK_WHOLE_SIZE;
	infos[1].buffer = data_.buffer;
	infos[1].range = VK_WHOLE_SIZE;
	VkWriteDescriptorSet writes[2]{};
	for (int i = 0; i < 2; ++i) {
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descSet_;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &infos[i];
	}
	vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
	return true;
}

// Staging memory, in a ring per frame: a frame's ring is reused once the frame is done, which starting it again
// waits for, or after a sync (which waits for everything).
uint8_t *ComputeRasterVulkan::AllocStaging(size_t bytes, VkBuffer *buffer, VkDeviceSize *offset) {
	const int frame = vulkan_->GetCurFrame();
	if (frame != stagingFrame_) {
		stagingUsed_[frame] = 0;
		stagingFrame_ = frame;
	}
	Buffer &buf = staging_[frame];
	size_t &used = stagingUsed_[frame];
	if (used + bytes > buf.size) {
		// The old one may still be read, so it's deleted later, and the new one starts empty.
		DestroyBuffer(buf);
		const size_t size = std::max(bytes * 2, (size_t)16 * 1024 * 1024);
		if (!CreateBuffer(buf, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, true))
			return nullptr;
		used = 0;
	}
	*buffer = buf.buffer;
	*offset = used;
	uint8_t *p = buf.mapped + used;
	used += (bytes + 255) & ~(size_t)255;
	return p;
}

const uint8_t *ComputeRasterVulkan::PageSource(uint32_t page) const {
	const uint32_t offset = page << PAGE_SHIFT;
	if (offset < MEM_VRAM_SIZE)
		return Memory::GetPointerUnchecked(0x04000000) + offset;
	return Memory::GetPointerUnchecked(0x08000000) + (offset - ramOffset_);
}

void ComputeRasterVulkan::AddUpload(uint32_t memOffset, const void *src, uint32_t bytes) {
	// Next to the last one? Then it's one copy.
	if (!uploads_.empty()) {
		Upload &last = uploads_.back();
		if (last.memOffset + last.size == memOffset && last.dataOffset + last.size == uploadData_.size()) {
			last.size += bytes;
			uploadData_.insert(uploadData_.end(), (const uint8_t *)src, (const uint8_t *)src + bytes);
			statUploadBytes_ += bytes;
			return;
		}
	}
	uploads_.push_back(Upload{ memOffset, bytes, uploadData_.size() });
	uploadData_.insert(uploadData_.end(), (const uint8_t *)src, (const uint8_t *)src + bytes);
	statUploadBytes_ += bytes;
}

// Makes the GPU's copy of the pages match PSP memory, unless the GPU drew to them (then it has the newer data).
// Once per page per flush: after that, changes the CPU makes are seen at the next flush, as with the threads.
void ComputeRasterVulkan::VerifyPages(uint32_t memOffset, uint32_t bytes) {
	if (bytes == 0)
		return;
	const uint32_t first = memOffset >> PAGE_SHIFT;
	const uint32_t last = std::min((uint32_t)(memOffset + bytes - 1), (uint32_t)(ramOffset_ + ramSize_ - 1)) >> PAGE_SHIFT;
	const uint32_t vramPages = MEM_VRAM_SIZE >> PAGE_SHIFT;
	for (uint32_t page = first; page <= last; ++page) {
		uint32_t &word = verified_[page >> 5];
		const uint32_t bit = 1U << (page & 31);
		if (word & bit)
			continue;
		word |= bit;
		if (page < vramPages && (gpuDirty_[page >> 5] & bit))
			continue;
		if (page >= vramPages && (page << PAGE_SHIFT) - ramOffset_ >= Memory::g_MemorySize)
			continue;
		const uint8_t *src = PageSource(page);
		uint8_t *shadow = &shadow_[page << PAGE_SHIFT];
		if (memcmp(src, shadow, PAGE_SIZE) != 0) {
			memcpy(shadow, src, PAGE_SIZE);
			AddUpload(page << PAGE_SHIFT, src, PAGE_SIZE);
		}
	}
}

void ComputeRasterVulkan::VerifyVRAM(uint32_t vramOffset, uint32_t bytes) {
	vramOffset &= MEM_VRAM_SIZE - 1;
	// Consecutive primitives mostly draw to the same pages.
	if (vramOffset >= lastVerified_.start && vramOffset + bytes <= lastVerified_.end && lastVerified_.gen == verifyGen_)
		return;
	lastVerified_ = { vramOffset, vramOffset + bytes, verifyGen_ };
	bytes = std::min(bytes, MEM_VRAM_SIZE);
	const uint32_t first = std::min(bytes, MEM_VRAM_SIZE - vramOffset);
	VerifyPages(vramOffset, first);
	VerifyPages(0, bytes - first);
}

void ComputeRasterVulkan::MarkGPUWrite(uint32_t vramOffset, uint32_t bytes) {
	if (bytes == 0)
		return;
	bytes = std::min(bytes, MEM_VRAM_SIZE);
	vramOffset &= MEM_VRAM_SIZE - 1;
	if (vramOffset >= lastMarked_.start && vramOffset + bytes <= lastMarked_.end && lastMarked_.gen == verifyGen_)
		return;
	lastMarked_ = { vramOffset, vramOffset + bytes, verifyGen_ };
	const uint32_t first = vramOffset >> PAGE_SHIFT;
	const uint32_t count = ((vramOffset + bytes - 1) >> PAGE_SHIFT) - first + 1;
	const uint32_t vramPages = MEM_VRAM_SIZE >> PAGE_SHIFT;
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t page = (first + i) % vramPages;
		gpuDirty_[page >> 5] |= 1U << (page & 31);
	}
	anyGPUDirty_ = true;
}

// Data from the CPU (a CLUT, a texture that isn't in PSP memory), copied to the scratch area for this batch.
// Zero past bytes, up to padTo.
uint32_t ComputeRasterVulkan::HostData(const void *src, uint32_t bytes, uint32_t padTo) {
	const uint32_t size = (std::max(bytes, padTo) + 15) & ~15;
	if (scratchUsed_ + size > SCRATCH_SIZE)
		return scratchOffset_;
	const uint32_t offset = scratchOffset_ + scratchUsed_;
	scratchUsed_ += size;
	AddUpload(offset, src, bytes);
	if (size > bytes) {
		static const uint8_t zeros[4096]{};
		for (uint32_t pos = bytes; pos < size; pos += sizeof(zeros))
			AddUpload(offset + pos, zeros, std::min((uint32_t)sizeof(zeros), size - pos));
	}
	return offset;
}

// Where a texture level's texels are for the shader, and making sure the GPU has them.
void ComputeRasterVulkan::ResolveLevel(const Rasterizer::RasterizerState &state, int level, bool snapshot, uint32_t *out) {
	const uint32_t w = state.samplerID.cached.sizes[level].w;
	const uint32_t h = state.samplerID.cached.sizes[level].h;
	const uint32_t bufw = state.texbufw[level];
	out[0] = w | (h << 16);
	out[1] = bufw;
	out[2] = 0;
	out[3] = 0;

	if (snapshot && snapValid_[level]) {
		out[2] = snapshotOffset_ + snapOffset_[level];
		out[3] = LV_VALID;
		return;
	}
	const u8 *ptr = state.texptr[level];
	if (!ptr)
		return;
	const uint32_t addr = state.texaddr[level];
	const uint32_t bits = textureBitsPerPixel[state.samplerID.texfmt];
	// Swizzled rows come in blocks of 8.
	const uint32_t rows = (h + 7) & ~7;
	const uint32_t bytes = bufw * bits / 8 * rows + 64;
	if (Memory::IsVRAMAddress(addr) && !Memory::IsDepthTexVRAMAddress(addr) && ptr == Memory::GetPointerUnchecked(addr)) {
		out[2] = addr & (MEM_VRAM_SIZE - 1);
		out[3] = LV_VALID | LV_VRAM;
		VerifyVRAM(out[2], bytes);
		return;
	}
	const uint32_t ramAddr = addr & 0x3FFFFFFF;
	if (ramAddr >= 0x08000000 && ramAddr < 0x08000000 + Memory::g_MemorySize && ptr == Memory::GetPointerUnchecked(addr)) {
		const uint32_t offset = ramAddr - 0x08000000;
		out[2] = ramOffset_ + offset;
		out[3] = LV_VALID;
		VerifyPages(out[2], std::min(bytes, Memory::g_MemorySize - offset));
		return;
	}
	// Not in PSP memory as the GPU has it (the PPGe atlas, a deswizzled depth view): upload what's read.
	const uint32_t exact = bufw * bits / 8 * h;
	auto it = hostData_.find(ptr);
	if (it != hostData_.end()) {
		out[2] = it->second;
	} else {
		out[2] = HostData(ptr, exact, bytes);
		hostData_[ptr] = out[2];
	}
	out[3] = LV_VALID;
}

uint32_t ComputeRasterVulkan::PackState(const Rasterizer::RasterizerState &state, bool snapshot) {
	using namespace Rasterizer;
	if (prims_.size() > MAX_BATCH_PRIM_WORDS || uploadData_.size() > MAX_BATCH_UPLOAD_BYTES || scratchUsed_ > SCRATCH_SIZE - 128 * 1024)
		Submit();

	const uint32_t index = (uint32_t)(states_.size() / ST_WORDS);
	states_.resize(states_.size() + ST_WORDS);
	uint32_t *w = &states_[index * ST_WORDS];
	const PixelFuncID &id = state.pixelID;
	const SamplerID &sid = state.samplerID;

	uint32_t px0 = 0;
	px0 |= id.clearMode ? PX0_CLEAR : 0;
	px0 |= id.colorTest ? PX0_COLORTEST : 0;
	px0 |= id.stencilTest ? PX0_STENCILTEST : 0;
	px0 |= id.depthWrite ? PX0_DEPTHWRITE : 0;
	px0 |= id.applyDepthRange ? PX0_DEPTHRANGE : 0;
	px0 |= id.alphaBlend ? PX0_ALPHABLEND : 0;
	px0 |= id.hasAlphaTestMask ? PX0_ALPHATESTMASK : 0;
	px0 |= id.hasStencilTestMask ? PX0_STENCILTESTMASK : 0;
	px0 |= id.dithering ? PX0_DITHER : 0;
	px0 |= id.applyLogicOp ? PX0_LOGICOP : 0;
	px0 |= id.applyFog ? PX0_FOG : 0;
	px0 |= id.applyColorWriteMask ? PX0_WRITEMASK : 0;
	px0 |= id.earlyZChecks ? PX0_EARLYZ : 0;
	px0 |= (uint32_t)id.fbFormat << PX0_FBFORMAT_SHIFT;
	px0 |= (uint32_t)id.alphaTestFunc << PX0_ALPHATEST_SHIFT;
	px0 |= (uint32_t)id.depthTestFunc << PX0_DEPTHTEST_SHIFT;
	px0 |= (uint32_t)id.stencilTestFunc << PX0_STENCILTEST_SHIFT;
	w[ST_PIXEL0] = px0;
	w[ST_PIXEL1] = ((uint32_t)id.alphaBlendEq << PX1_BLENDEQ_SHIFT) | ((uint32_t)id.alphaBlendSrc << PX1_BLENDSRC_SHIFT) | ((uint32_t)id.alphaBlendDst << PX1_BLENDDST_SHIFT) |
		((uint32_t)id.sFail << PX1_SFAIL_SHIFT) | ((uint32_t)id.zFail << PX1_ZFAIL_SHIFT) | ((uint32_t)id.zPass << PX1_ZPASS_SHIFT);
	w[ST_REFS] = (uint32_t)id.alphaTestRef | ((uint32_t)id.stencilTestRef << 8) | ((uint32_t)id.cached.alphaTestMask << 16) | ((uint32_t)id.cached.stencilTestMask << 24);
	w[ST_MISC] = (uint32_t)id.cached.stencilRef | ((uint32_t)id.cached.logicOp << 8) | ((uint32_t)id.cached.colorTestFunc << 16);
	w[ST_WRITEMASK] = id.cached.colorWriteMask;
	memcpy(&w[ST_DITHER], id.cached.ditherMatrix, 16);
	w[ST_FOGCOLOR] = id.cached.fogColor;
	w[ST_MINZ] = (uint32_t)id.cached.minz;
	w[ST_MAXZ] = (uint32_t)id.cached.maxz;
	w[ST_STRIDES] = (uint32_t)id.cached.framebufStride | ((uint32_t)id.cached.depthbufStride << 16);
	w[ST_CTEST_MASK] = id.cached.colorTestMask;
	w[ST_CTEST_REF] = id.cached.colorTestRef;
	w[ST_FIXA] = id.cached.alphaBlendSrc;
	w[ST_FIXB] = id.cached.alphaBlendDst;
	// A target change flushes, so these are the ones the state draws to.
	w[ST_FB_ADDR] = gstate.getFrameBufAddress() & (MEM_VRAM_SIZE - 1);
	w[ST_Z_BASE] = depthbuf.base;
	w[ST_Z_ROTMASK] = depthbuf.layout.rotMask;
	w[ST_Z_ROTSHIFT] = (uint32_t)depthbuf.layout.rotShift;
	w[ST_Z_XOR] = depthbuf.layout.xorBits;

	uint32_t sm = 0;
	sm |= state.shadeGouraud ? SM_GOURAUD : 0;
	sm |= state.throughMode ? SM_THROUGH : 0;
	if (state.enableTextures) {
		sm |= SM_TEXTURES;
		sm |= (uint32_t)sid.texfmt << SM_TEXFMT_SHIFT;
		sm |= (uint32_t)sid.clutfmt << SM_CLUTFMT_SHIFT;
		sm |= sid.clampS ? SM_CLAMPS : 0;
		sm |= sid.clampT ? SM_CLAMPT : 0;
		sm |= sid.swizzle ? SM_SWIZZLE : 0;
		sm |= sid.useSharedClut ? SM_SHAREDCLUT : 0;
		sm |= (uint32_t)sid.texFunc << SM_TEXFUNC_SHIFT;
		sm |= sid.useTextureAlpha ? SM_TEXALPHA : 0;
		sm |= sid.useColorDoubling ? SM_DOUBLING : 0;
		sm |= state.mipFilt ? SM_MIPFILT : 0;
		sm |= state.minFilt ? SM_MINFILT : 0;
		sm |= state.magFilt ? SM_MAGFILT : 0;
		sm |= (uint32_t)state.texLevelMode << SM_LEVELMODE_SHIFT;
		sm |= (sid.hasClutShift || sid.hasClutMask || sid.hasClutOffset) ? SM_CLUTTRANSFORM : 0;
		if (g_Config.iTexFiltering == TEX_FILTER_FORCE_LINEAR)
			sm |= SM_FORCE_LINEAR;
		else if (g_Config.iTexFiltering == TEX_FILTER_FORCE_NEAREST)
			sm |= SM_FORCE_NEAREST;
		w[ST_CLUTFORMAT] = sid.cached.clutFormat;
		w[ST_TEXENV] = sid.cached.texBlendColor;
		w[ST_TEXLEVEL] = (uint32_t)(uint8_t)state.texLevelOffset | ((uint32_t)state.maxTexLevel << 8) | ((uint32_t)sid.width0Shift << 16) | ((uint32_t)sid.height0Shift << 24);
		if ((sid.texfmt & 4) != 0 && sid.cached.clut) {
			auto it = hostData_.find(sid.cached.clut);
			if (it != hostData_.end()) {
				w[ST_CLUT_ADDR] = it->second;
			} else {
				// Indices can reach past the 1 KB the CLUT has, which reads zeros here.
				w[ST_CLUT_ADDR] = HostData(sid.cached.clut, 1024, 2048);
				hostData_[sid.cached.clut] = w[ST_CLUT_ADDR];
			}
		}
		for (int i = 0; i <= state.maxTexLevel; ++i)
			ResolveLevel(state, i, snapshot, &w[ST_LEVELS + i * 4]);
	}
	w[ST_SAMPLER] = sm;
	return index;
}

void ComputeRasterVulkan::CommitPrim(uint32_t offset, const Rasterizer::RasterizerState &state) {
	primOffsets_.push_back(offset);
	statPrims_++;

	// What it can draw to: the rows of its pixels in the color buffer, and the depth buffer if it tests or
	// writes it. The GPU needs those as they are first.
	const uint32_t xy1 = prims_[offset + PRIM_XY1], xy2 = prims_[offset + PRIM_XY2];
	const uint32_t x1 = xy1 & 0xFFFF, y1 = xy1 >> 16, x2 = xy2 & 0xFFFF, y2 = xy2 >> 16;
	const PixelFuncID &id = state.pixelID;
	const uint32_t bpp = id.FBFormat() == GE_FORMAT_8888 ? 4 : 2;
	const uint32_t fbStride = id.cached.framebufStride;
	const uint32_t fbAddr = gstate.getFrameBufAddress() & (MEM_VRAM_SIZE - 1);
	const uint32_t colorStart = fbAddr + (x1 + y1 * fbStride) * bpp;
	const uint32_t colorEnd = fbAddr + (x2 + y2 * fbStride) * bpp + bpp;
	VerifyVRAM(colorStart, colorEnd - colorStart);
	MarkGPUWrite(colorStart, colorEnd - colorStart);

	const bool depthRead = !id.clearMode && id.DepthTestFunc() != GE_COMP_ALWAYS;
	if (depthRead || id.depthWrite) {
		const uint32_t zStride = id.cached.depthbufStride;
		const uint32_t zStart = (depthbuf.base + (x1 + y1 * zStride) * 2) & ~(DEPTH_BLOCK - 1);
		const uint32_t zEnd = (depthbuf.base + (x2 + y2 * zStride) * 2 + 2 + DEPTH_BLOCK - 1) & ~(DEPTH_BLOCK - 1);
		VerifyVRAM(zStart, zEnd - zStart);
		if (id.depthWrite)
			MarkGPUWrite(zStart, zEnd - zStart);
	}
}

void ComputeRasterVulkan::EndSegment() {
	if (primOffsets_.size() > segmentStart_) {
		Op op{ Op::Type::SEGMENT };
		op.first = segmentStart_;
		op.end = (uint32_t)primOffsets_.size();
		ops_.push_back(op);
		segmentStart_ = op.end;
	}
}

void ComputeRasterVulkan::BeginSnapshot() {
	snapUsed_ = 0;
	for (bool &valid : snapValid_)
		valid = false;
}

void ComputeRasterVulkan::SnapshotLevel(int level, uint32_t addr, uint32_t bytes) {
	EndSegment();
	if (!Memory::IsVRAMAddress(addr) || bytes > MEM_VRAM_SIZE || snapUsed_ + bytes > SNAPSHOT_SIZE)
		return;
	const uint32_t src = addr & (MEM_VRAM_SIZE - 1);
	VerifyVRAM(src, bytes);
	snapOffset_[level] = snapUsed_;
	snapValid_[level] = true;
	snapUsed_ = (snapUsed_ + bytes + 15) & ~15;

	const uint32_t dst = snapshotOffset_ + snapOffset_[level];
	const uint32_t first = std::min(bytes, MEM_VRAM_SIZE - src);
	Op op{ Op::Type::COPY };
	op.src = src;
	op.dst = dst;
	op.size = first;
	ops_.push_back(op);
	if (first < bytes) {
		op.src = 0;
		op.dst = dst + first;
		op.size = bytes - first;
		ops_.push_back(op);
	}
}

void ComputeRasterVulkan::ResetBatch() {
	states_.clear();
	prims_.clear();
	primOffsets_.clear();
	segmentStart_ = 0;
	ops_.clear();
	uploads_.clear();
	uploadData_.clear();
	scratchUsed_ = 0;
	hostData_.clear();
	batchId_++;
}

// Calls f(tile) for the tiles a primitive can draw in: those its pixel bounds touch, and for a triangle only
// where every edge (but the snapping one, whose pixels look a few to the side) has a pixel center inside.
template <typename F>
void ComputeRasterVulkan::ForPrimTiles(uint32_t off, F f) {
	const uint32_t xy1 = prims_[off + PRIM_XY1], xy2 = prims_[off + PRIM_XY2];
	const int tx1 = std::min((int)(xy1 & 0xFFFF) >> TILE_SHIFT, TILES_X - 1), ty1 = std::min((int)(xy1 >> 16) >> TILE_SHIFT, TILES_Y - 1);
	const int tx2 = std::min((int)(xy2 & 0xFFFF) >> TILE_SHIFT, TILES_X - 1), ty2 = std::min((int)(xy2 >> 16) >> TILE_SHIFT, TILES_Y - 1);
	const bool triangle = (prims_[off + PRIM_TYPE_FLAGS] & 0xF) == (uint32_t)PrimType::TRIANGLE;
	if (!triangle || (tx1 == tx2 && ty1 == ty2)) {
		for (int ty = ty1; ty <= ty2; ++ty) {
			for (int tx = tx1; tx <= tx2; ++tx)
				f(ty * TILES_X + tx);
		}
		return;
	}

	const uint32_t *setup = &prims_[off + PRIM_HEADER_WORDS];
	const int snapEdge = (int8_t)(setup[TRI_SNAP] & 0xFF);
	int64_t a[3], b[3], c[3];
	int edges = 0;
	for (int k = 0; k < 3; ++k) {
		if (k == snapEdge)
			continue;
		const uint32_t *e = setup + TRI_EDGES + k * 4;
		a[edges] = (int32_t)e[0];
		b[edges] = (int32_t)e[1];
		c[edges] = (int64_t)((uint64_t)e[2] | ((uint64_t)e[3] << 32));
		edges++;
	}
	constexpr int TILE_SUB = SCREEN_SCALE_FACTOR << TILE_SHIFT;
	for (int ty = ty1; ty <= ty2; ++ty) {
		const int64_t yc0 = (int64_t)ty * TILE_SUB + SCREEN_SCALE_FACTOR / 2, yc1 = yc0 + TILE_SUB - SCREEN_SCALE_FACTOR;
		for (int tx = tx1; tx <= tx2; ++tx) {
			const int64_t xc0 = (int64_t)tx * TILE_SUB + SCREEN_SCALE_FACTOR / 2, xc1 = xc0 + TILE_SUB - SCREEN_SCALE_FACTOR;
			bool inside = true;
			for (int k = 0; k < edges && inside; ++k) {
				// The edge function's largest value over the tile's pixel centers.
				const int64_t best = c[k] + a[k] * (a[k] > 0 ? xc1 : xc0) + b[k] * (b[k] > 0 ? yc1 : yc0);
				inside = best >= 0;
			}
			if (inside)
				f(ty * TILES_X + tx);
		}
	}
}

// Records the batch for the GPU (it runs with the frame's init commands, or at the next sync).
void ComputeRasterVulkan::Submit() {
	EndSegment();
	if (ops_.empty() && uploads_.empty()) {
		ResetBatch();
		return;
	}

	// Each segment's tiles and the primitives touching them, in order.
	tileData_.clear();
	std::vector<uint32_t> items;
	for (Op &op : ops_) {
		if (op.type != Op::Type::SEGMENT)
			continue;
		tilesTouched_.clear();
		tilePrims_.clear();
		for (uint32_t i = op.first; i < op.end; ++i) {
			const uint32_t off = primOffsets_[i];
			ForPrimTiles(off, [&](int t) {
				tilePrims_.push_back(((uint64_t)off << 16) | (uint32_t)t);
				if (tileCount_[t]++ == 0)
					tilesTouched_.push_back((uint16_t)t);
			});
		}
		op.tileFirst = (uint32_t)(tileData_.size() / 3);
		op.tileCount = (uint32_t)tilesTouched_.size();
		uint32_t pos = (uint32_t)items.size();
		for (uint16_t t : tilesTouched_) {
			tileData_.push_back((uint32_t)(t % TILES_X) | ((uint32_t)(t / TILES_X) << 16));
			tileData_.push_back(pos);
			tileData_.push_back(tileCount_[t]);
			tilePos_[t] = pos;
			pos += tileCount_[t];
			tileCount_[t] = 0;
		}
		items.resize(pos);
		for (uint64_t tp : tilePrims_)
			items[tilePos_[tp & 0xFFFF]++] = (uint32_t)(tp >> 16);
		statSegments_++;
	}

	// The data buffer: states, primitives, tile headers, tile items.
	const uint32_t statesOffset = 0;
	const uint32_t primsOffset = statesOffset + (uint32_t)states_.size();
	const uint32_t tilesOffset = primsOffset + (uint32_t)prims_.size();
	const uint32_t itemsOffset = tilesOffset + (uint32_t)tileData_.size();
	const size_t dataWords = itemsOffset + items.size();
	const size_t dataBytes = std::max(dataWords * 4, (size_t)4);
	const size_t stagingBytes = dataBytes + uploadData_.size();

	VkBuffer stagingBuf;
	VkDeviceSize stagingOffset;
	uint8_t *staging = AllocStaging(stagingBytes, &stagingBuf, &stagingOffset);
	if (!staging) {
		ERROR_LOG(Log::G3D, "Software compute: out of staging memory, dropping a batch");
		ResetBatch();
		return;
	}
	uint8_t *p = staging;
	auto put = [&](const std::vector<uint32_t> &v) {
		if (!v.empty())
			memcpy(p, v.data(), v.size() * 4);
		p += v.size() * 4;
	};
	put(states_);
	put(prims_);
	put(tileData_);
	put(items);
	if (!uploadData_.empty())
		memcpy(staging + dataBytes, uploadData_.data(), uploadData_.size());

	if (dataBytes > data_.size) {
		DestroyBuffer(data_);
		const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		if (!CreateBuffer(data_, dataBytes * 2, storage, VMA_MEMORY_USAGE_GPU_ONLY, false) || !UpdateDescriptorSet()) {
			ERROR_LOG(Log::G3D, "Software compute: out of memory for a batch");
			ResetBatch();
			return;
		}
	}

	VkCommandBuffer cmd = renderManager_->GetInitCmd();
	VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	// What earlier batches read and drew comes before overwriting it.
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

	VkBufferCopy dataCopy{ stagingOffset, 0, dataBytes };
	vkCmdCopyBuffer(cmd, stagingBuf, data_.buffer, 1, &dataCopy);
	if (!uploads_.empty()) {
		std::vector<VkBufferCopy> copies;
		copies.reserve(uploads_.size());
		for (const Upload &up : uploads_)
			copies.push_back(VkBufferCopy{ stagingOffset + dataBytes + up.dataOffset, up.memOffset, up.size });
		vkCmdCopyBuffer(cmd, stagingBuf, mem_.buffer, (uint32_t)copies.size(), copies.data());
	}

	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

	const bool timed = queryPool_ != VK_NULL_HANDLE && queriesUsed_ + 2 <= MAX_QUERIES;
	if (timed) {
		vkCmdResetQueryPool(cmd, queryPool_, queriesUsed_, 2);
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool_, queriesUsed_);
	}
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descSet_, 0, nullptr);
	for (const Op &op : ops_) {
		if (op.type == Op::Type::SEGMENT) {
			if (op.tileCount == 0)
				continue;
			PushParams params{ statesOffset, primsOffset, tilesOffset + op.tileFirst * 3, itemsOffset, memBytes_ };
			vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
			vkCmdDispatch(cmd, op.tileCount, 1, 1);
			// The next segment, or copy, sees what it drew.
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
		} else {
			VkBufferCopy copy{ op.src, op.dst, op.size };
			vkCmdCopyBuffer(cmd, mem_.buffer, mem_.buffer, 1, &copy);
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
		}
	}

	if (timed) {
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool_, queriesUsed_ + 1);
		queriesUsed_ += 2;
	}
	statBatches_++;
	ResetBatch();
}

bool ComputeRasterVulkan::HasGPUWrites(uint32_t addr, uint32_t bytes) const {
	if (!anyGPUDirty_ || bytes == 0)
		return false;
	bytes = std::min(bytes, MEM_VRAM_SIZE);
	const uint32_t offset = addr & (MEM_VRAM_SIZE - 1);
	const uint32_t first = offset >> PAGE_SHIFT;
	const uint32_t count = ((offset + bytes - 1) >> PAGE_SHIFT) - first + 1;
	const uint32_t vramPages = MEM_VRAM_SIZE >> PAGE_SHIFT;
	for (uint32_t i = 0; i < count; ++i) {
		const uint32_t page = (first + i) % vramPages;
		if (gpuDirty_[page >> 5] & (1U << (page & 31)))
			return true;
	}
	return false;
}

void ComputeRasterVulkan::Flush(bool readback) {
	Submit();
	statFlushes_++;

	if (readback && anyGPUDirty_) {
		// Copy back the pages the GPU drew to.
		VkCommandBuffer cmd = renderManager_->GetInitCmd();
		VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

		std::vector<VkBufferCopy> copies;
		const uint32_t vramPages = MEM_VRAM_SIZE >> PAGE_SHIFT;
		for (uint32_t page = 0; page < vramPages; ++page) {
			if (!(gpuDirty_[page >> 5] & (1U << (page & 31))))
				continue;
			const VkDeviceSize offset = (VkDeviceSize)page << PAGE_SHIFT;
			if (!copies.empty() && copies.back().srcOffset + copies.back().size == offset)
				copies.back().size += PAGE_SIZE;
			else
				copies.push_back(VkBufferCopy{ offset, offset, PAGE_SIZE });
		}
		vkCmdCopyBuffer(cmd, mem_.buffer, readback_.buffer, (uint32_t)copies.size(), copies.data());
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

		const double start = time_now_d();
		renderManager_->SyncInitCommands();
		statSyncTime_ += time_now_d() - start;
		vmaInvalidateAllocation(vulkan_->Allocator(), readback_.alloc, 0, VK_WHOLE_SIZE);
		if (queriesUsed_ != 0) {
			uint64_t stamps[MAX_QUERIES];
			if (vkGetQueryPoolResults(vulkan_->GetDevice(), queryPool_, 0, queriesUsed_, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
				for (uint32_t i = 0; i < queriesUsed_; i += 2)
					statGPUTime_ += (double)(stamps[i + 1] - stamps[i]) * timestampPeriod_;
			}
			queriesUsed_ = 0;
		}

		// Only what the GPU changed goes to memory: the CPU may have written elsewhere in these pages since, and
		// then the shadow keeps the GPU's old bytes, so the next upload sends the CPU's.
		uint8_t *vram = Memory::GetPointerWriteUnchecked(0x04000000);
		for (const VkBufferCopy &copy : copies) {
			const uint32_t *gpu = (const uint32_t *)(readback_.mapped + copy.dstOffset);
			uint32_t *shadow = (uint32_t *)(shadow_.data() + copy.srcOffset);
			uint32_t *cpu = (uint32_t *)(vram + copy.srcOffset);
			const size_t words = (size_t)copy.size / 4;
			for (size_t i = 0; i < words; ++i) {
				const uint32_t g = gpu[i];
				const uint32_t s = shadow[i];
				if (g == s)
					continue;
				const uint32_t changed = g ^ s;
				uint32_t mask = 0;
				for (int b = 0; b < 4; ++b) {
					if (changed & (0xFFU << (b * 8)))
						mask |= 0xFFU << (b * 8);
				}
				cpu[i] = (cpu[i] & ~mask) | (g & mask);
				shadow[i] = g;
			}
			statReadbackBytes_ += copy.size;
		}
		std::fill(gpuDirty_.begin(), gpuDirty_.end(), 0);
		anyGPUDirty_ = false;
		// The GPU is idle, so all the staging memory can be reused.
		for (size_t &used : stagingUsed_)
			used = 0;
	}

	// The CPU may change memory now, so check again what's read next.
	std::fill(verified_.begin(), verified_.end(), 0);
	verifyGen_++;
}

void ComputeRasterVulkan::GetStats(StringWriter &w) {
	w.F("Compute: %d prims, %d batches, %d segments, %d flushes\n", statPrims_, statBatches_, statSegments_, statFlushes_);
	w.F("Uploaded %d KB, read back %d KB, sync wait %0.2f ms, GPU %0.2f ms\n", (int)(statUploadBytes_ / 1024), (int)(statReadbackBytes_ / 1024), statSyncTime_ * 1000.0, statGPUTime_ * 1000.0);
	statPrims_ = 0;
	statBatches_ = 0;
	statSegments_ = 0;
	statFlushes_ = 0;
	statUploadBytes_ = 0;
	statReadbackBytes_ = 0;
	statSyncTime_ = 0.0;
	statGPUTime_ = 0.0;
}

Backend *CreateVulkanBackend(void *drawContext) {
	ComputeRasterVulkan *backend = new ComputeRasterVulkan((Draw::DrawContext *)drawContext);
	if (!backend->Init()) {
		delete backend;
		return nullptr;
	}
	return backend;
}

}  // namespace SoftCompute
