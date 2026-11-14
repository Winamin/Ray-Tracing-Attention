#pragma once
#include "resident_dimensions.hpp"
#include "optical_geometry.hpp"

// The angular chart locates candidates; all live operator inputs participate in
// the verification fingerprint. Conversation IDs and token text are absent.
class OpticalPathKey {
public:
    using Key=std::array<uint32_t,8>;
    struct Address {Key key;std::array<uint32_t,2> chart;};
    struct Config {
        uint32_t pass,position,capacity,index,epoch,validate;Key expected;
        uint32_t first,rows,blocks,blockFirst,blockCount,replace,decode;
    };
    static constexpr uint32_t blockTokens=16u;
    Engine& e;Buffer *addresses{},*scratch{},*historySummary{};uint32_t capacity{},historyBlocks{};
    std::array<uint32_t,MODEL_FULL> historyPositions{};bool historyValid=false;
    VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkDescriptorPool pool{};VkDescriptorSet set{};VkPipeline pipeline{};
    OpticalPathKey(Engine& engine,Buffer* state,Buffer* field,Buffer* history,Buffer* control,uint32_t cap):e(engine){
        addresses=e.buffer(65u*sizeof(Address),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkDescriptorSetLayoutBinding bindings[7]{};for(uint32_t i=0;i<7u;i++)bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1u,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=7;sl.pBindings=bindings;check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"optical path key layout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Config)};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"optical path key pipeline layout");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,7u};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&size;check(vkCreateDescriptorPool(e.device,&dp,nullptr,&pool),"optical path key pool");
        VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};a.descriptorPool=pool;a.descriptorSetCount=1;a.pSetLayouts=&setLayout;check(vkAllocateDescriptorSets(e.device,&a,&set),"optical path key set");
        rebind(state,field,history,control,cap);pipeline=e.pipeline("optical_path_key",layout);
    }
    uint32_t groups()const{
        uint64_t words=MODEL_D+uint64_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128u*128u+MODEL_S*9u);
        return uint32_t((words+1023u)/1024u);
    }
    void rebind(Buffer* state,Buffer* field,Buffer* history,Buffer* control,uint32_t cap){
        if(!cap)throw std::runtime_error("Optical history capacity must be nonzero");
        if(!scratch)scratch=e.buffer(size_t(groups())*32u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if(cap!=capacity){
            e.release(historySummary);capacity=cap;historyBlocks=(cap+blockTokens-1u)/blockTokens;
            // Eight aggregate words followed by one digest per existing history block.
            historySummary=e.buffer((1u+size_t(MODEL_FULL)*historyBlocks)*32u,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        invalidateHistory();
        Buffer* buffers[7]{state,field,history,control,addresses,scratch,historySummary};VkDescriptorBufferInfo infos[7]{};VkWriteDescriptorSet writes[7]{};
        for(uint32_t i=0;i<7u;i++){infos[i]={buffers[i]->handle,0,buffers[i]->size};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}
        vkUpdateDescriptorSets(e.device,7,writes,0,nullptr);
    }
    void invalidateHistory(){historyValid=false;}
    void resetHistory(VkCommandBuffer c){
        vkCmdFillBuffer(c,historySummary->handle,0,historySummary->size,0u);
        Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        historyPositions.fill(0u);historyValid=true;
    }
    void bind(VkCommandBuffer c){
        vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
    }
    void historyRange(VkCommandBuffer c,uint32_t layer,uint32_t first,uint32_t rows,uint32_t position,bool replace,bool decode=false){
        if(!rows)return;
        Config cfg{};cfg.pass=2u;cfg.position=position;cfg.capacity=capacity;cfg.index=layer;
        cfg.first=first;cfg.rows=rows;cfg.blocks=historyBlocks;cfg.replace=uint32_t(replace);cfg.decode=uint32_t(decode);
        bind(c);
        auto dispatch=[&](uint32_t firstSlot,uint32_t count){
            if(!count)return;
            cfg.blockFirst=firstSlot/blockTokens;cfg.blockCount=(firstSlot+count-1u)/blockTokens-cfg.blockFirst+1u;
            vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
            vkCmdDispatch(c,std::min(cfg.blockCount,65535u),(cfg.blockCount+65534u)/65535u,1u);
            Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);
        };
        if(rows>=capacity)dispatch(0u,capacity);
        else{uint32_t slot=first%capacity,count=std::min(rows,capacity-slot);dispatch(slot,count);dispatch(0u,rows-count);}
    }
    // Ordinary decode only hashes newly packed rows. Rewrites replace the
    // affected 16-token summaries, removing their previous aggregate exactly.
    void updateHistory(VkCommandBuffer c,uint32_t layer,uint32_t first,uint32_t rows,bool decode=false){
        if(layer>=MODEL_FULL||rows>UINT32_MAX-first)throw std::runtime_error("Invalid optical history update");
        if(!rows||!historyValid)return;
        if(first>historyPositions[layer]){invalidateHistory();return;}
        uint32_t position=std::max(historyPositions[layer],first+rows);
        historyRange(c,layer,first,rows,position,first!=historyPositions[layer]||position>capacity,decode);
        historyPositions[layer]=position;
    }
    void rebuildHistory(VkCommandBuffer c,uint32_t position){
        resetHistory(c);uint32_t rows=std::min(position,capacity),first=position-rows;
        for(uint32_t layer=0u;layer<MODEL_FULL;layer++){historyRange(c,layer,first,rows,position,false);historyPositions[layer]=position;}
    }
    void capture(VkCommandBuffer c,uint32_t position,uint32_t index,uint32_t epoch,const Key* expected=nullptr){
        if(!historyValid||std::any_of(historyPositions.begin(),historyPositions.end(),[&](uint32_t saved){return saved!=position;}))rebuildHistory(c,position);
        Config cfg{};cfg.position=position;cfg.capacity=capacity;cfg.index=index;cfg.epoch=epoch;
        cfg.validate=uint32_t(expected!=nullptr);cfg.expected=expected?*expected:Key{};
        bind(c);
        for(uint32_t pass=0;pass<2u;pass++){cfg.pass=pass;vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);uint32_t count=groups();vkCmdDispatch(c,pass?1u:std::min(count,65535u),pass?1u:(count+65534u)/65535u,1u);Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);}
    }
    ~OpticalPathKey(){vkDeviceWaitIdle(e.device);vkDestroyPipeline(e.device,pipeline,nullptr);vkDestroyDescriptorPool(e.device,pool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);e.release(addresses);e.release(scratch);e.release(historySummary);}
};
