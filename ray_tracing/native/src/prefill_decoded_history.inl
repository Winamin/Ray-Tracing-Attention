// Standard cooperative prefill uses these transient
// work buffers; original history packets, local classes and the single BVH stay.
Buffer *prefillDecodedData{},*prefillDecodedMeta{};
VkPipeline prefillDecodePipeline{},prefillScratchAttention{};
uint32_t prefillDecodedEpoch=0u;
static constexpr uint32_t prefillDecodedMagic=0x50444831u;
void allocatePrefillDecodedHistory(){
    e.release(prefillDecodedData);e.release(prefillDecodedMeta);
    prefillDecodedEpoch=0u;
    if(!MODEL_FULL||!MODEL_GQA||RT_ATTENTION_HEAD_DIM<64u||
       attentionBackend!=0u||!wideCooperative||!e.cooperativeMatrix2||
       !e.supportsWorkgroup(256u,uint64_t(64u)*RT_ATTENTION_HEAD_DIM*2u))return;
    VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
    uint64_t blocks=(uint64_t(capacity)+127u)/128u;
    uint64_t dataBytes=uint64_t(capacity)*MODEL_GQA*4u;
    uint64_t metaBytes=64u+(blocks+capacity)*16u;
    if(dataBytes>properties.limits.maxStorageBufferRange||metaBytes>properties.limits.maxStorageBufferRange||
       dataBytes>UINT32_MAX||metaBytes>UINT32_MAX)return;
    auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    prefillDecodedData=e.buffer(dataBytes,usage,false,true);
    prefillDecodedMeta=e.buffer(metaBytes,usage,false,true);
}
uint32_t recordPrefillDecodedHistory(VkCommandBuffer c,ResidentConfig cfg,VkDescriptorSet selected){
    // The immediate producer/consumer pair never crosses a source update,
    // layer, command segment or snapshot. Every use has a new generation.
    if(!prefillDecodedData||!prefillDecodedMeta||!prefillDecodePipeline||!prefillScratchAttention||
       attentionBackend!=0u||cfg.rows<8u||
       cfg.mode>2u||prefillDecodedEpoch==UINT32_MAX)return 0u;
    uint64_t final=cfg.mode==0u?uint64_t(cfg.rows)-1u:
        cfg.mode==1u?uint64_t(cfg.position):uint64_t(cfg.position)+cfg.rows-1u;
    // Canonical nonrotating prefix only. Ring windows keep the original loader.
    if(final>=capacity)return 0u;
    uint32_t visible=uint32_t(final)+1u,epoch=++prefillDecodedEpoch;
    uint32_t blocks=(capacity+127u)/128u,activeBlocks=(visible+127u)/128u;
    uint32_t header[16]{prefillDecodedMagic,epoch,cfg.layer,capacity,
        cfg.rows,cfg.position,cfg.mode,visible,blocks,activeBlocks,MODEL_GQA,RT_ATTENTION_HEAD_DIM,0u,0u,0u,0u};
    vkCmdUpdateBuffer(c,prefillDecodedMeta->handle,0u,sizeof(header),header);
    Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    cfg.checkpoint=epoch;cfg.stage=0u;
    bindCompute(c,prefillDecodePipeline,selected);
    vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
    vkCmdDispatch(c,activeBlocks,MODEL_GQA/RT_ATTENTION_HEAD_DIM,1u);
    // Publishes all FP16 work and provenance after real RT hits, before the
    // sole paired matrix consumer. No source writes exist between these calls.
    sync(c);
    return epoch;
}
