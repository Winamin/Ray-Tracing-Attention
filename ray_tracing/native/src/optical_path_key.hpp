#pragma once
#include "resident_dimensions.hpp"
#include "optical_geometry.hpp"

// The angular chart locates candidates; all live operator inputs participate in
// the verification fingerprint. Conversation IDs and token text are absent.
class OpticalPathKey {
public:
    using Key=std::array<uint32_t,8>;
    struct Address {Key key;std::array<uint32_t,2> chart;};
    struct Config {uint32_t pass,position,capacity,index,epoch,validate;Key expected;};
    Engine& e;Buffer *addresses{},*scratch{};uint32_t capacity{};
    VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkDescriptorPool pool{};VkDescriptorSet set{};VkPipeline pipeline{};
    OpticalPathKey(Engine& engine,Buffer* state,Buffer* field,Buffer* history,Buffer* control,uint32_t cap):e(engine){
        addresses=e.buffer(65u*sizeof(Address),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkDescriptorSetLayoutBinding bindings[6]{};for(uint32_t i=0;i<6u;i++)bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1u,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=6;sl.pBindings=bindings;check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"optical path key layout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Config)};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"optical path key pipeline layout");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6u};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&size;check(vkCreateDescriptorPool(e.device,&dp,nullptr,&pool),"optical path key pool");
        VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};a.descriptorPool=pool;a.descriptorSetCount=1;a.pSetLayouts=&setLayout;check(vkAllocateDescriptorSets(e.device,&a,&set),"optical path key set");
        rebind(state,field,history,control,cap);pipeline=e.pipeline("optical_path_key",layout);
    }
    uint32_t groups(uint32_t count)const{
        uint64_t words=MODEL_D+uint64_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128u*128u+MODEL_S*9u)+uint64_t(MODEL_FULL)*count*(HISTORY_BYTES/4u);
        return uint32_t((words+1023u)/1024u);
    }
    void rebind(Buffer* state,Buffer* field,Buffer* history,Buffer* control,uint32_t cap){
        if(cap>capacity){if(scratch)e.release(scratch);scratch=e.buffer(size_t(groups(cap))*32u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);capacity=cap;}
        Buffer* buffers[6]{state,field,history,control,addresses,scratch};VkDescriptorBufferInfo infos[6]{};VkWriteDescriptorSet writes[6]{};
        for(uint32_t i=0;i<6u;i++){infos[i]={buffers[i]->handle,0,buffers[i]->size};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}
        vkUpdateDescriptorSets(e.device,6,writes,0,nullptr);
    }
    void capture(VkCommandBuffer c,uint32_t position,uint32_t index,uint32_t epoch,const Key* expected=nullptr){
        Config cfg{0u,position,capacity,index,epoch,uint32_t(expected!=nullptr),expected?*expected:Key{}};
        vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        for(uint32_t pass=0;pass<2u;pass++){cfg.pass=pass;vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);uint32_t count=groups(position);vkCmdDispatch(c,pass?1u:std::min(count,65535u),pass?1u:(count+65534u)/65535u,1u);Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);}
    }
    ~OpticalPathKey(){vkDeviceWaitIdle(e.device);vkDestroyPipeline(e.device,pipeline,nullptr);vkDestroyDescriptorPool(e.device,pool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);e.release(addresses);e.release(scratch);}
};
