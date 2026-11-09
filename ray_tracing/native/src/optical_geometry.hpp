#pragma once
#include "vulkan_engine.hpp"
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
template<class T> static void input(std::vector<T>& data,size_t count){data.resize(count);if(!std::cin.read(reinterpret_cast<char*>(data.data()),count*sizeof(T)))throw std::runtime_error("Truncated optical input");}
template<class T> static void append(std::vector<char>& out,const std::vector<T>& data){const auto* at=reinterpret_cast<const char*>(data.data());out.insert(out.end(),at,at+data.size()*sizeof(T));}

class ShapeMesh {
public:
    Engine& e;uint32_t count,perInstance,groups;Buffer *vertices{},*instances{};
    std::vector<Leaf> leaves;std::vector<VkAccelerationStructureKHR> tops;
    std::vector<Buffer*> topScratch;std::vector<VkAccelerationStructureGeometryKHR> topGeometry;
    std::vector<Buffer*> topStorage;
    std::vector<VkAccelerationStructureInstanceKHR> hostInstances;
    ShapeMesh(Engine& engine,uint32_t n,uint32_t verts,uint32_t groupCount,bool translated=false):e(engine),count(n),perInstance(verts),groups(groupCount) {
        auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        vertices=e.buffer(size_t(count)*perInstance*12,usage);std::vector<float> blank(size_t(count)*perInstance*3);
        if(translated)for(uint32_t i=0;i<count;i++)for(uint32_t p=0;p<perInstance/3;p++){
            size_t at=(size_t(i)*perInstance+p*3)*3;float x=2.f*(p%40),y=2.f*(p/40);
            blank[at]=x;blank[at+1]=y;blank[at+3]=x+1;blank[at+4]=y;blank[at+6]=x;blank[at+7]=y+1;
        }
        e.upload(vertices,blank.data(),vertices->size);
        leaves.resize(count);hostInstances.resize(count);
        for(uint32_t i=0;i<count;i++) {
            auto& leaf=leaves[i];leaf.first=i;leaf.count=verts/3;
            leaf.geometry={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};leaf.geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR;leaf.geometry.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto& tr=leaf.geometry.geometry.triangles;tr.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;tr.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;tr.vertexData.deviceAddress=vertices->address+size_t(i)*perInstance*12;tr.vertexStride=12;tr.maxVertex=perInstance-1;tr.indexType=VK_INDEX_TYPE_NONE_KHR;
            VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.geometryCount=1;bi.pGeometries=&leaf.geometry;
            uint32_t primitives=perInstance/3;VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};e.getSizes(e.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&bi,&primitives,&sizes);
            auto* storage=e.buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);leaf.storage=storage;leaf.scratch=e.buffer(std::max(sizes.buildScratchSize,sizes.updateScratchSize)+e.alignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};ci.type=bi.type;ci.buffer=storage->handle;ci.size=storage->size;check(e.createAS(e.device,&ci,nullptr,&leaf.handle),"create optical shape BLAS");
            auto& in=hostInstances[i];in.transform.matrix[0][0]=in.transform.matrix[1][1]=in.transform.matrix[2][2]=1;if(translated)in.transform.matrix[0][3]=float(i)*128.f;in.instanceCustomIndex=i;in.mask=groups==4?1u<<(i%8):255;
            VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};address.accelerationStructure=leaf.handle;in.accelerationStructureReference=e.asAddress(e.device,&address);
        }
        instances=e.buffer(count*sizeof(hostInstances[0]),usage);e.upload(instances,hostInstances.data(),instances->size);
        for(uint32_t group=0;group<groups;group++) {
            uint32_t first=group*(count/groups),primitives=count/groups;VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};geometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;geometry.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;geometry.geometry.instances.data.deviceAddress=instances->address+first*sizeof(hostInstances[0]);topGeometry.push_back(geometry);
            VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.geometryCount=1;bi.pGeometries=&topGeometry.back();
            VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};e.getSizes(e.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&bi,&primitives,&sizes);auto* storage=e.buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);topScratch.push_back(e.buffer(std::max(sizes.buildScratchSize,sizes.updateScratchSize)+e.alignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
            topStorage.push_back(storage);VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};ci.type=bi.type;ci.buffer=storage->handle;ci.size=storage->size;VkAccelerationStructureKHR top{};check(e.createAS(e.device,&ci,nullptr,&top),"create optical shape TLAS");tops.push_back(top);
        }
        auto c=e.command();e.begin(c);build(c,count,true);check(vkEndCommandBuffer(c),"end optical shape initialization");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
    }
    void build(VkCommandBuffer command,uint32_t active,bool initial=false,bool onlyTop=false,uint32_t first=0,uint32_t topGroup=~0u) {
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        if(!onlyTop)for(uint32_t i=initial&&topGroup==~0u?0u:first;i<(initial&&topGroup==~0u?count:active);i++) {
            auto& leaf=leaves[i];VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.mode=initial?VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR:VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;bi.srcAccelerationStructure=initial?VK_NULL_HANDLE:leaf.handle;bi.dstAccelerationStructure=leaf.handle;bi.geometryCount=1;bi.pGeometries=&leaf.geometry;bi.scratchData.deviceAddress=(leaf.scratch->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);infos.push_back(bi);ranges.push_back({perInstance/3,0,0,0});
        }
        if(!infos.empty()){std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rp;for(auto& r:ranges)rp.push_back(&r);e.buildAS(command,uint32_t(infos.size()),infos.data(),rp.data());Engine::barrier(command,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);}
        for(uint32_t group=topGroup==~0u?0u:topGroup;group<(topGroup==~0u?groups:topGroup+1u);group++) {
            VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.mode=initial?VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR:VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;bi.srcAccelerationStructure=initial?VK_NULL_HANDLE:tops[group];bi.dstAccelerationStructure=tops[group];bi.geometryCount=1;bi.pGeometries=&topGeometry[group];bi.scratchData.deviceAddress=(topScratch[group]->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);VkAccelerationStructureBuildRangeInfoKHR range{count/groups,0,0,0};const auto* rp=&range;e.buildAS(command,1,&bi,&rp);
        }
        Engine::barrier(command,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    VkDeviceSize scratchBytes()const{VkDeviceSize bytes=0;for(auto& leaf:leaves)if(leaf.scratch)bytes+=leaf.scratch->allocationSize;for(auto* top:topScratch)if(top)bytes+=top->allocationSize;return bytes;}
    VkDeviceSize sceneBytes()const{VkDeviceSize bytes=vertices->size+instances->size+scratchBytes();for(auto& leaf:leaves)bytes+=leaf.storage->size;for(auto* top:topStorage)bytes+=top->size;return bytes;}
    void releaseScratch(){for(auto& leaf:leaves)e.release(leaf.scratch);for(auto*& top:topScratch)e.release(top);}
    ~ShapeMesh(){vkDeviceWaitIdle(e.device);for(auto top:tops)e.destroyAS(e.device,top,nullptr);for(auto& leaf:leaves){e.destroyAS(e.device,leaf.handle,nullptr);e.release(leaf.scratch);e.release(leaf.storage);}for(auto*& item:topScratch)e.release(item);for(auto*& item:topStorage)e.release(item);e.release(vertices);e.release(instances);}
};

