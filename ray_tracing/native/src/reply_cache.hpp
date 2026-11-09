#pragma once
#include "optical_path_key.hpp"

// The BVH locates angular candidates and validates optical state fingerprints.
// Successor addresses keep token queries independent of conversation branches.
class ReplyTreeCache {
public:
    using Key=OpticalPathKey::Key;using Address=OpticalPathKey::Address;
    Engine& e;
    static constexpr uint32_t nodeCapacity=65536u,tile=256u;
    ShapeMesh scene;
    Buffer *receipt{},*nodes{},*control{},*tokens{},*addresses{};
    VkDescriptorSetLayout setLayout{};
    VkPipelineLayout layout{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    VkPipeline pipeline{};
    uint64_t sceneBytes{};
    uint32_t cursor{},size{};
    struct Config {uint32_t operation;Address address;uint32_t position,count,first,index;};
    ReplyTreeCache(Engine& engine,Buffer* controls,Buffer* words,Buffer* paths):e(engine),scene(engine,nodeCapacity/tile,tile*6u,1),control(controls),tokens(words),addresses(paths){
        receipt=e.buffer(64u*64u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        nodes=e.buffer(size_t(nodeCapacity)*96u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkDescriptorSetLayoutBinding bindings[7]{};
        for(uint32_t i=0;i<7;i++)bindings[i]={i,i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=7;sl.pBindings=bindings;
        check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"cache chain layout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Config)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;
        check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"cache chain pipeline layout");
        VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=sizes;
        check(vkCreateDescriptorPool(e.device,&dp,nullptr,&pool),"cache chain pool");
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};alloc.descriptorPool=pool;alloc.descriptorSetCount=1;alloc.pSetLayouts=&setLayout;
        check(vkAllocateDescriptorSets(e.device,&alloc,&set),"cache chain set");
        rebind(controls,words);pipeline=e.pipeline("reply_cache",layout);
        auto c=e.command();e.begin(c);vkCmdFillBuffer(c,nodes->handle,0,nodes->size,0u);
        Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        check(vkEndCommandBuffer(c),"end cache chain initialization");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        sceneBytes=scene.sceneBytes()+nodes->size+receipt->size;
    }
    void rebind(Buffer* controls,Buffer* words){
        control=controls;tokens=words;
        VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};as.accelerationStructureCount=1;as.pAccelerationStructures=scene.tops.data();
        Buffer* buffers[6]{scene.vertices,receipt,control,tokens,nodes,addresses};VkDescriptorBufferInfo infos[6]{};VkWriteDescriptorSet writes[7]{};
        for(uint32_t i=0;i<7;i++){auto& w=writes[i];w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=set;w.dstBinding=i;w.descriptorCount=1;w.descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            if(i){infos[i-1]={buffers[i-1]->handle,0,buffers[i-1]->size};w.pBufferInfo=&infos[i-1];}else w.pNext=&as;}
        vkUpdateDescriptorSets(e.device,7,writes,0,nullptr);
    }
    void dispatch(VkCommandBuffer c,Config cfg){
        vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);vkCmdDispatch(c,cfg.operation==2u?(cfg.count+31u)/32u:1u,1,1);
        Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);
    }
    void probe(VkCommandBuffer c,Address address,uint32_t position,uint32_t count){dispatch(c,{1u,address,position,count,0u,0u});}
    void probeParallel(VkCommandBuffer c,uint32_t count){
        if(!count||count>64u)throw std::runtime_error("Parallel optical cache expects 1..64 independent rays");
        dispatch(c,{2u,Address{},0u,count,0u,0u});
    }
    void record(VkCommandBuffer c,Address address,uint32_t position,uint32_t count,uint32_t index){
        dispatch(c,{0u,address,position,count,cursor,index});
        uint32_t first=cursor/tile,last=((cursor+count-1u)%nodeCapacity)/tile;
        if(first<=last)scene.build(c,last+1u,false,false,first);
        else{scene.build(c,nodeCapacity/tile,false,false,first);scene.build(c,last+1u,false,false,0u);}
    }
    void commit(uint32_t count){cursor=(cursor+count)%nodeCapacity;size=std::min(nodeCapacity,size+count);}
    ~ReplyTreeCache(){vkDeviceWaitIdle(e.device);vkDestroyPipeline(e.device,pipeline,nullptr);vkDestroyDescriptorPool(e.device,pool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);e.release(nodes);e.release(receipt);}
};
