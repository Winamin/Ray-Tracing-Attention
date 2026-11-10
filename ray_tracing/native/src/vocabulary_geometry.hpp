#pragma once
#include "optical_geometry.hpp"

// All block planes form one native BLAS. One root instance connects its TLAS.
// The driver owns the BVH topology; the caller supplies fixed plane bounds.
class ImmutableBlockGeometry {
public:
    Engine& e;
    uint32_t count;
    Buffer *vertices{},*instances{},*storage{},*scratch{};
    Leaf plane;
    std::vector<VkAccelerationStructureKHR> tops;
    VkAccelerationStructureGeometryKHR geometry{};
    bool built=false;

    ImmutableBlockGeometry(Engine& engine,uint32_t receivers,uint32_t blocks):e(engine) {
        uint64_t total=uint64_t(receivers)*blocks;
        if(!total||total>0xffffffffu)throw std::runtime_error("Block surface primitive address range exceeded");
        count=uint32_t(total);
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|
            VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        vertices=e.buffer(uint64_t(count)*sizeof(VkAabbPositionsKHR),usage);
        plane.geometry={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        plane.geometry.geometryType=VK_GEOMETRY_TYPE_AABBS_KHR;
        auto& aabb=plane.geometry.geometry.aabbs;
        aabb.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        aabb.data.deviceAddress=vertices->address;aabb.stride=sizeof(VkAabbPositionsKHR);
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.geometryCount=1;info.pGeometries=&plane.geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        e.getSizes(e.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&count,&sizes);
        plane.storage=e.buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
        plane.scratch=e.buffer(sizes.buildScratchSize+e.alignment,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create.type=info.type;create.buffer=plane.storage->handle;create.size=plane.storage->size;
        check(e.createAS(e.device,&create,nullptr,&plane.handle),"create immutable block BLAS");
        VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        address.accelerationStructure=plane.handle;
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0]=instance.transform.matrix[1][1]=instance.transform.matrix[2][2]=1;
        instance.mask=255;instance.accelerationStructureReference=e.asAddress(e.device,&address);
        instances=e.buffer(sizeof(instance),usage);e.upload(instances,&instance,sizeof(instance));
        geometry={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.data.deviceAddress=instances->address;
        info.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;info.pGeometries=&geometry;
        uint32_t rootCount=1u;
        e.getSizes(e.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&rootCount,&sizes);
        storage=e.buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
        scratch=e.buffer(sizes.buildScratchSize+e.alignment,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        create.type=info.type;create.buffer=storage->handle;create.size=storage->size;
        VkAccelerationStructureKHR top{};
        check(e.createAS(e.device,&create,nullptr,&top),"create immutable block TLAS");tops.push_back(top);
    }

    void build(VkCommandBuffer c,uint32_t active,bool initial=true) {
        if(built||!initial||active!=count)throw std::runtime_error("Block geometry is an immutable compiled field");
        Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        info.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.dstAccelerationStructure=plane.handle;info.geometryCount=1;info.pGeometries=&plane.geometry;
        info.scratchData.deviceAddress=(plane.scratch->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);
        VkAccelerationStructureBuildRangeInfoKHR range{count,0,0,0};const auto* ranges=&range;
        e.buildAS(c,1,&info,&ranges);
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        info.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;info.dstAccelerationStructure=tops[0];info.pGeometries=&geometry;
        info.scratchData.deviceAddress=(scratch->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);
        range.primitiveCount=1u;e.buildAS(c,1,&info,&ranges);
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        built=true;
    }

    uint64_t scratchBytes()const{return (scratch?scratch->allocationSize:0u)+(plane.scratch?plane.scratch->allocationSize:0u);}
    uint64_t sceneBytes()const{return vertices->size+instances->size+storage->size+plane.storage->size+scratchBytes();}
    void releaseScratch(){e.release(scratch);e.release(plane.scratch);}
    ~ImmutableBlockGeometry(){
        if(!e.device)return;vkDeviceWaitIdle(e.device);
        for(auto top:tops)e.destroyAS(e.device,top,nullptr);
        e.destroyAS(e.device,plane.handle,nullptr);
        e.release(scratch);e.release(plane.scratch);e.release(storage);e.release(plane.storage);
        e.release(instances);e.release(vertices);
    }
};

using VocabularyGeometry=ImmutableBlockGeometry;
