#pragma once
#define NOMINMAX
#include <vulkan/vulkan_core.h>
#include "cooperative_matrix.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static void check(VkResult r,const char* label){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(label)+": Vulkan "+std::to_string(r));}
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Buffer {VkBuffer handle{};VkDeviceMemory memory{};VkDeviceSize size{},allocationSize{};VkDeviceAddress address{};void* map{};bool ownsMemory=true,deviceLocal=false;uint32_t poolIndex=~0u;};
struct Leaf {VkAccelerationStructureKHR handle{};VkAccelerationStructureGeometryKHR geometry{};Buffer *scratch{},*storage{};uint32_t first{},count{};};
struct Measure {double gpu{},wall{};};
class Engine {
public:
    VkInstance instance{};VkPhysicalDevice physical{};VkDevice device{};VkQueue queue{};
    VkPhysicalDeviceMemoryProperties memory{};uint32_t family{},alignment{},capacity{},width{};
    float period{};std::string deviceName;VkCommandPool pool{};VkFence fence{};VkQueryPool timestamps{};uint32_t timestampCount{};bool cooperativeMatrix{},cooperativeMatrix2{};
    PFN_vkCreateAccelerationStructureKHR createAS{};PFN_vkDestroyAccelerationStructureKHR destroyAS{};
    PFN_vkGetAccelerationStructureBuildSizesKHR getSizes{};PFN_vkCmdBuildAccelerationStructuresKHR buildAS{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR asAddress{};
    std::vector<std::unique_ptr<Buffer>> buffers;Buffer *uploadStaging{},*downloadStaging{};
    struct MemoryPool {VkDeviceMemory memory;VkDeviceSize size,used;uint32_t type,kind,live;};
    std::vector<MemoryPool> memoryPools;
    uint64_t allocated{},physicalAllocated{},physicalDeviceAllocated{};std::filesystem::path directory;VkPipelineLayout layout{};
    Buffer* buffer(VkDeviceSize size,VkBufferUsageFlags usage,bool host=false) {
        auto item=std::make_unique<Buffer>();auto* b=item.get();buffers.push_back(std::move(item));b->size=size;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=size;ci.usage=usage;
        check(vkCreateBuffer(device,&ci,nullptr,&b->handle),"create buffer");
        VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};requirements.pNext=&dedicated;
        VkBufferMemoryRequirementsInfo2 request{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};request.buffer=b->handle;
        vkGetBufferMemoryRequirements2(device,&request,&requirements);auto req=requirements.memoryRequirements;
        b->allocationSize=req.size;
        VkMemoryPropertyFlags want=host?(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t type=~0u;for(uint32_t i=0;i<memory.memoryTypeCount;i++)if((req.memoryTypeBits&(1u<<i))&&(memory.memoryTypes[i].propertyFlags&want)==want){type=i;break;}
        if(type==~0u)throw std::runtime_error("No suitable Vulkan memory type");
        b->deviceLocal=(memory.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!=0;
        uint32_t kind=usage&VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR?2u:
            (usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)&&!(usage&VK_BUFFER_USAGE_TRANSFER_DST_BIT)?1u:0u;
        VkDeviceSize bindingOffset=0;
        if(!host&&req.size<16u*1024u*1024u&&!dedicated.requiresDedicatedAllocation){
            MemoryPool* selected=nullptr;
            for(auto& pool:memoryPools){auto at=(pool.used+req.alignment-1)&~(req.alignment-1);if(pool.memory&&pool.type==type&&pool.kind==kind&&at+req.size<=pool.size){selected=&pool;bindingOffset=at;break;}}
            if(!selected){
                VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=std::max<VkDeviceSize>(req.size,(kind==1u?8u:64u)*1024u*1024u);ai.memoryTypeIndex=type;
                VkDeviceMemory backing{};check(vkAllocateMemory(device,&ai,nullptr,&backing),"allocate optical memory pool");
                memoryPools.push_back({backing,ai.allocationSize,0,type,kind,0});selected=&memoryPools.back();physicalAllocated+=ai.allocationSize;physicalDeviceAllocated+=ai.allocationSize;
            }
            b->memory=selected->memory;b->ownsMemory=false;b->poolIndex=uint32_t(selected-memoryPools.data());selected->used=bindingOffset+req.size;selected->live++;
        }else{
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)?VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT:0;
            VkMemoryDedicatedAllocateInfo standalone{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};standalone.buffer=b->handle;
            if(dedicated.requiresDedicatedAllocation)flags.pNext=&standalone;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=req.size;ai.memoryTypeIndex=type;
            check(vkAllocateMemory(device,&ai,nullptr,&b->memory),"allocate buffer");physicalAllocated+=req.size;
            if(memory.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)physicalDeviceAllocated+=req.size;
        }
        allocated+=req.size;
        check(vkBindBufferMemory(device,b->handle,b->memory,bindingOffset),"bind memory");
        if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT){VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};bi.buffer=b->handle;b->address=vkGetBufferDeviceAddress(device,&bi);}
        if(host)check(vkMapMemory(device,b->memory,0,size,0,&b->map),"map staging");
        return b;
    }
    VkCommandBuffer command() {
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};info.commandPool=pool;info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;info.commandBufferCount=1;
        VkCommandBuffer c{};check(vkAllocateCommandBuffers(device,&info,&c),"allocate command");return c;
    }
    void begin(VkCommandBuffer c){VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(c,&info),"begin command");}
    double submit(VkCommandBuffer c) {
        auto start=Clock::now();check(vkResetFences(device,1,&fence),"reset fence");
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};info.commandBufferCount=1;info.pCommandBuffers=&c;
        check(vkQueueSubmit(queue,1,&info,fence),"submit command");
        VkResult waited;do{waited=vkWaitForFences(device,1,&fence,VK_TRUE,500'000'000ull);}while(waited==VK_TIMEOUT);
        check(waited,"wait command");return ms(start);
    }
    static void barrier(VkCommandBuffer c,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags from,VkAccessFlags to) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=from;b.dstAccessMask=to;
        vkCmdPipelineBarrier(c,src,dst,0,1,&b,0,nullptr,0,nullptr);
    }
    void release(Buffer*& current) {
        if(!current)return;
        if(current->map)vkUnmapMemory(device,current->memory);
        vkDestroyBuffer(device,current->handle,nullptr);allocated-=current->allocationSize;
        if(current->ownsMemory){vkFreeMemory(device,current->memory,nullptr);physicalAllocated-=current->allocationSize;if(current->deviceLocal)physicalDeviceAllocated-=current->allocationSize;}
        else {auto& backing=memoryPools[current->poolIndex];if(--backing.live==0){vkFreeMemory(device,backing.memory,nullptr);backing.memory=VK_NULL_HANDLE;physicalAllocated-=backing.size;physicalDeviceAllocated-=backing.size;}}
        buffers.erase(std::remove_if(buffers.begin(),buffers.end(),[&](const auto& item){return item.get()==current;}),buffers.end());current=nullptr;
    }
    Buffer* staging(Buffer*& current,VkDeviceSize size,VkBufferUsageFlags usage) {
        if(current&&current->size>=size)return current;
        release(current);
        current=buffer(size,usage,true);return current;
    }
    void upload(Buffer* target,const void* data,VkDeviceSize size) {
        auto* staging=this->staging(uploadStaging,std::min<VkDeviceSize>(size,64u*1024u*1024u),VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        for(VkDeviceSize offset=0;offset<size;){VkDeviceSize bytes=std::min(staging->size,size-offset);std::memcpy(staging->map,static_cast<const char*>(data)+offset,bytes);
            auto c=command();begin(c);VkBufferCopy copy{0,offset,bytes};vkCmdCopyBuffer(c,staging->handle,target->handle,1,&copy);
            barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT);
            check(vkEndCommandBuffer(c),"end upload");submit(c);vkFreeCommandBuffers(device,pool,1,&c);offset+=bytes;}
    }
    template<class T>std::vector<T> download(Buffer* from,size_t count,VkDeviceSize offset=0) {
        auto* staging=this->staging(downloadStaging,count*sizeof(T),VK_BUFFER_USAGE_TRANSFER_DST_BIT);auto c=command();begin(c);
        barrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy{offset,0,count*sizeof(T)};vkCmdCopyBuffer(c,from->handle,staging->handle,1,&copy);
        barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        check(vkEndCommandBuffer(c),"end download");submit(c);vkFreeCommandBuffers(device,pool,1,&c);
        std::vector<T> data(count);std::memcpy(data.data(),staging->map,count*sizeof(T));return data;
    }
    VkPipeline pipeline(const char* name,VkPipelineLayout customLayout=VK_NULL_HANDLE,uint32_t operation=0xffffffffu,uint32_t tile=1u,uint32_t projection=0xffffffffu,uint32_t spatialTile=0xffffffffu,uint32_t localWidth=0xffffffffu,uint32_t featureTile=0xffffffffu,uint32_t readoutUnroll=0xffffffffu,uint32_t readoutWords=0xffffffffu,uint32_t coefficientFormat=0xffffffffu,uint32_t upFormat=0xffffffffu) {
        std::ifstream file(directory/(std::string(name)+".spv"),std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("Missing SPIR-V");
        size_t bytes=size_t(file.tellg());file.seekg(0);std::vector<uint32_t> code(bytes/4);file.read(reinterpret_cast<char*>(code.data()),bytes);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=bytes;mi.pCode=code.data();VkShaderModule module{};
        check(vkCreateShaderModule(device,&mi,nullptr,&module),"create shader");
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pi.layout=customLayout?customLayout:layout;
        pi.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=module;pi.stage.pName="main";
        uint32_t constants[10]{operation,tile,projection,spatialTile,localWidth==0xffffffffu?128u:localWidth,featureTile,readoutUnroll,readoutWords,coefficientFormat,upFormat};VkSpecializationMapEntry entries[10]{{0,0,4},{1,4,4},{2,8,4},{3,12,4},{4,16,4},{5,20,4},{6,24,4},{7,28,4},{8,32,4},{9,36,4}};uint32_t constantCount=upFormat!=0xffffffffu?10u:coefficientFormat!=0xffffffffu?9u:readoutWords!=0xffffffffu?8u:readoutUnroll!=0xffffffffu?7u:featureTile!=0xffffffffu?6u:localWidth!=0xffffffffu?5u:spatialTile!=0xffffffffu?4u:projection!=0xffffffffu?3u:tile==1u?1u:2u;VkSpecializationInfo specialization{constantCount,entries,size_t(constantCount)*4u,constants};if(operation!=0xffffffffu)pi.stage.pSpecializationInfo=&specialization;
        VkPipeline result{};auto r=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pi,nullptr,&result);vkDestroyShaderModule(device,module,nullptr);check(r,"create pipeline");return result;
    }
    void initialize(const std::filesystem::path& dir,uint32_t cap){directory=dir;capacity=cap;bool integerDot=false;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Qwen pure optical";app.apiVersion=VK_MAKE_VERSION(1,3,0);
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ii.pApplicationInfo=&app;check(vkCreateInstance(&ii,nullptr,&instance),"create Vulkan instance");
        uint32_t count{};check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"count GPU");std::vector<VkPhysicalDevice> gpus(count);check(vkEnumeratePhysicalDevices(instance,&count,gpus.data()),"list GPU");
        for(auto gpu:gpus){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(gpu,&p);if(p.vendorID==0x10de){physical=gpu;break;}}
        if(!physical)throw std::runtime_error("NVIDIA RT GPU required for this benchmark");
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR dotProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR};
        if(integerDot)subgroup.pNext=&dotProps;
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};asProps.pNext=&subgroup;
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};props.pNext=&asProps;vkGetPhysicalDeviceProperties2(physical,&props);
        alignment=asProps.minAccelerationStructureScratchOffsetAlignment;period=props.properties.limits.timestampPeriod;deviceName=props.properties.deviceName;
        if(subgroup.subgroupSize!=32||!(subgroup.supportedOperations&VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))throw std::runtime_error("32-lane arithmetic subgroups required");
        if(integerDot&&!dotProps.integerDotProduct4x8BitPackedSignedAccelerated)throw std::runtime_error("Hardware packed signed integer dot product required");
        VkPhysicalDeviceBufferDeviceAddressFeatures address{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR};
        uint32_t extensionCount=0;vkEnumerateDeviceExtensionProperties(physical,nullptr,&extensionCount,nullptr);std::vector<VkExtensionProperties> available(extensionCount);vkEnumerateDeviceExtensionProperties(physical,nullptr,&extensionCount,available.data());
        bool hasCooperative=std::any_of(available.begin(),available.end(),[](const auto& p){return std::strcmp(p.extensionName,VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME)==0;});
        VkPhysicalDeviceShaderFloat16Int8Features half{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceCooperativeMatrixFeaturesNV coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV};
        bool hasCooperative2=std::any_of(available.begin(),available.end(),[](const auto& p){return std::strcmp(p.extensionName,"VK_NV_cooperative_matrix2")==0;});
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopKHR{VkStructureType(1000506000)};
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coop2{VkStructureType(1000593000)};
        VkPhysicalDeviceMaintenance4FeaturesKHR maintenance{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR};
        VkPhysicalDevice16BitStorageFeatures storage16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        bool hasVector=std::any_of(available.begin(),available.end(),[](const auto& p){return std::strcmp(p.extensionName,"VK_NV_cooperative_matrix_decode_vector")==0;});
        VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV vectorDecode{VkStructureType(1000689000)};
        if(hasCooperative){address.pNext=&half;half.pNext=&coop;}
        if(hasCooperative2){coop.pNext=&coopKHR;coopKHR.pNext=&coop2;coop2.pNext=&maintenance;maintenance.pNext=&storage16;}
        if(hasVector)storage16.pNext=&vectorDecode;
        if(integerDot)address.pNext=&dot;
        rq.pNext=&as;as.pNext=&address;VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&rq;vkGetPhysicalDeviceFeatures2(physical,&features);
        if(!rq.rayQuery||!as.accelerationStructure||!address.bufferDeviceAddress)throw std::runtime_error("Vulkan RT unavailable");
        if(integerDot&&!dot.shaderIntegerDotProduct)throw std::runtime_error("Integer dot product unavailable");
        cooperativeMatrix=hasCooperative&&half.shaderFloat16&&coop.cooperativeMatrix;
        cooperativeMatrix2=hasCooperative2&&maintenance.maintenance4&&storage16.storageBuffer16BitAccess&&coopKHR.cooperativeMatrix&&coop2.cooperativeMatrixWorkgroupScope&&coop2.cooperativeMatrixFlexibleDimensions&&coop2.cooperativeMatrixTensorAddressing&&coop2.cooperativeMatrixBlockLoads&&coop2.cooperativeMatrixReductions&&coop2.cooperativeMatrixConversions&&coop2.cooperativeMatrixPerElementOperations;
        cooperativeMatrix2=cooperativeMatrix2&&hasVector&&vectorDecode.cooperativeMatrixDecodeVector&&half.shaderFloat16&&features.features.shaderInt16&&features.features.shaderInt64&&(subgroup.supportedOperations&VK_SUBGROUP_FEATURE_CLUSTERED_BIT);
        if(cooperativeMatrix){auto get=(PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)vkGetInstanceProcAddr(instance,"vkGetPhysicalDeviceCooperativeMatrixPropertiesNV");uint32_t n=0;check(get(physical,&n,nullptr),"query cooperative matrix shapes");std::vector<VkCooperativeMatrixPropertiesNV> shapes(n,{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV});check(get(physical,&n,shapes.data()),"query cooperative matrix shapes");cooperativeMatrix=std::any_of(shapes.begin(),shapes.end(),[](const auto& p){return p.MSize==16&&p.NSize==8&&p.KSize==16&&p.AType==VK_COMPONENT_TYPE_FLOAT16_NV&&p.BType==VK_COMPONENT_TYPE_FLOAT16_NV&&p.CType==VK_COMPONENT_TYPE_FLOAT32_NV&&p.DType==VK_COMPONENT_TYPE_FLOAT32_NV&&p.scope==VK_SCOPE_SUBGROUP_NV;});}
        rq={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};as={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};address={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        rq.rayQuery=as.accelerationStructure=address.bufferDeviceAddress=VK_TRUE;rq.pNext=&as;as.pNext=&address;
        if(cooperativeMatrix){half={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};half.shaderFloat16=VK_TRUE;coop={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV};coop.cooperativeMatrix=VK_TRUE;address.pNext=&half;half.pNext=&coop;}
        if(cooperativeMatrix2){coopKHR={VkStructureType(1000506000)};coopKHR.cooperativeMatrix=VK_TRUE;coop2={VkStructureType(1000593000)};coop2.cooperativeMatrixWorkgroupScope=coop2.cooperativeMatrixFlexibleDimensions=coop2.cooperativeMatrixTensorAddressing=coop2.cooperativeMatrixBlockLoads=coop2.cooperativeMatrixReductions=coop2.cooperativeMatrixConversions=coop2.cooperativeMatrixPerElementOperations=VK_TRUE;coop.pNext=&coopKHR;coopKHR.pNext=&coop2;maintenance={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR};maintenance.maintenance4=VK_TRUE;coop2.pNext=&maintenance;storage16={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};storage16.storageBuffer16BitAccess=VK_TRUE;maintenance.pNext=&storage16;}
        if(cooperativeMatrix2){vectorDecode={VkStructureType(1000689000)};vectorDecode.cooperativeMatrixDecodeVector=VK_TRUE;storage16.pNext=&vectorDecode;}
        if(integerDot)address.pNext=&dot;
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());family=~0u;
        for(uint32_t i=0;i<count;i++)if((families[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&families[i].timestampValidBits==64){family=i;break;}
        if(family==~0u)throw std::runtime_error("Timestamp compute queue unavailable");
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        std::vector<const char*> extensions{"VK_KHR_acceleration_structure","VK_KHR_ray_query","VK_KHR_deferred_host_operations"};if(integerDot)extensions.push_back("VK_KHR_shader_integer_dot_product");if(cooperativeMatrix)extensions.push_back(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
        if(cooperativeMatrix2){extensions.push_back("VK_KHR_cooperative_matrix");extensions.push_back("VK_NV_cooperative_matrix2");extensions.push_back("VK_NV_cooperative_matrix_decode_vector");}
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.pNext=&rq;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;di.enabledExtensionCount=uint32_t(extensions.size());di.ppEnabledExtensionNames=extensions.data();VkPhysicalDeviceFeatures enabled{};enabled.shaderStorageBufferArrayDynamicIndexing=features.features.shaderStorageBufferArrayDynamicIndexing;di.pEnabledFeatures=&enabled;
        if(cooperativeMatrix2){enabled.shaderInt16=features.features.shaderInt16;enabled.shaderInt64=features.features.shaderInt64;}
        check(vkCreateDevice(physical,&di,nullptr,&device),"create Vulkan device");vkGetDeviceQueue(device,family,0,&queue);vkGetPhysicalDeviceMemoryProperties(physical,&memory);
#define LOAD(symbol,member) member=reinterpret_cast<decltype(member)>(vkGetDeviceProcAddr(device,#symbol));if(!member)throw std::runtime_error(#symbol)
        LOAD(vkCreateAccelerationStructureKHR,createAS);LOAD(vkDestroyAccelerationStructureKHR,destroyAS);LOAD(vkGetAccelerationStructureBuildSizesKHR,getSizes);LOAD(vkCmdBuildAccelerationStructuresKHR,buildAS);LOAD(vkGetAccelerationStructureDeviceAddressKHR,asAddress);
#undef LOAD
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=family;check(vkCreateCommandPool(device,&cp,nullptr,&pool),"create command pool");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fi,nullptr,&fence),"create fence");
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qp.queryType=VK_QUERY_TYPE_TIMESTAMP;qp.queryCount=timestampCount=2+64*(RT_LAYER_COUNT*7+4);check(vkCreateQueryPool(device,&qp,nullptr,&timestamps),"create query pool");
    }
    void ensureTimestamps(uint32_t count){if(count<=timestampCount)return;check(vkDeviceWaitIdle(device),"wait for timestamp resize");vkDestroyQueryPool(device,timestamps,nullptr);timestamps=VK_NULL_HANDLE;VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qp.queryType=VK_QUERY_TYPE_TIMESTAMP;qp.queryCount=timestampCount=count;check(vkCreateQueryPool(device,&qp,nullptr,&timestamps),"resize timestamp pool");}
    Measure measure(VkCommandBuffer c,uint32_t repeats){double wall=submit(c);uint64_t stamps[2];check(vkGetQueryPoolResults(device,timestamps,0,2,16,stamps,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"read optical timing");return {double(stamps[1]-stamps[0])*period/1e6,wall};}
    ~Engine(){if(!device)return;vkDeviceWaitIdle(device);if(timestamps)vkDestroyQueryPool(device,timestamps,nullptr);if(fence)vkDestroyFence(device,fence,nullptr);if(pool)vkDestroyCommandPool(device,pool,nullptr);for(auto& b:buffers){if(b->map)vkUnmapMemory(device,b->memory);vkDestroyBuffer(device,b->handle,nullptr);if(b->ownsMemory)vkFreeMemory(device,b->memory,nullptr);}for(auto& backing:memoryPools)if(backing.memory)vkFreeMemory(device,backing.memory,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);}
};
