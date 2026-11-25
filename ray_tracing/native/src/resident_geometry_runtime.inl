static constexpr uint32_t geometryScoreSegment = 8u;

void geometryDispatch(VkCommandBuffer c, ResidentConfig cfg, uint32_t operation,
                      uint32_t x, uint32_t y=1u, uint32_t z=1u, VkDescriptorSet target=VK_NULL_HANDLE)
{
    cfg.stage=operation;
    cfg.padding0=2u;
    cfg.padding1=geometryScoreStride;
    if(operation==5u){y=(x+32767u)/32768u;x=std::min(x,32768u);}
    else if(operation==0u||operation==1u||operation==4u){z=(x+32767u)/32768u;x=std::min(x,32768u);}
    bindCompute(c,geometryPipelines[operation],target?target:layerSets[cfg.layer]);
    vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
    uint32_t bucket=operation>=5u?1u:operation==4u||operation<=1u?2u:0u;
    uint32_t stamp=operation==8u?~0u:startRT(c,bucket);
    vkCmdDispatch(c,x,y,z);
    syncCompute(c);
    endRT(c,stamp);
}

void geometryPublishError(VkCommandBuffer c, ResidentConfig cfg)
{
    geometryDispatch(c,cfg,8u,1u);
}

void prepareGeometryAttention(uint32_t rows,uint32_t position,uint32_t origin)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(e.physical,&properties);
    uint32_t last=position+rows-1u;
    if(!rows||last<origin)throw std::runtime_error("Invalid geometry attention interval");
    geometryScoreStride=std::min(last-origin+1u,capacity);
    uint64_t rowBytes=uint64_t(geometryScoreStride)*RT_ATTENTION_HEADS*4u;
    geometryScoreRows=uint32_t(std::min<uint64_t>({rows,32u,properties.limits.maxStorageBufferRange/rowBytes}));
    if(!geometryScoreRows)throw std::runtime_error("Geometry attention row exceeds the device storage range");
    uint64_t bytes=rowBytes*geometryScoreRows;
    if(bytes<=geometryScores->size)return;
    uint64_t ceiling=std::min<uint64_t>(uint64_t(capacity),std::max<uint64_t>(uint64_t(geometryScoreStride)*4u,4096u));
    uint64_t wanted=std::max<uint64_t>(bytes,std::min<uint64_t>(ceiling,uint64_t(geometryScoreStride)*8u)*RT_ATTENTION_HEADS*4u*geometryScoreRows);
    e.release(geometryScores);
    geometryScores=e.buffer(wanted,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    for(uint32_t layer=0;layer<depth;layer++)bind(layerSets[layer],false,layer);
    bind(initialSet,true);
}

void executeGeometryMLP(VkCommandBuffer c,ResidentConfig cfg)
{
    vkCmdFillBuffer(c,featureWork->handle,0u,featureWork->size,0u);
    sync(c);
    uint32_t rows=(cfg.rows+matrixBatchRows-1u)/matrixBatchRows;
    for(uint32_t pass=0u;pass<2u;pass++)
    {
        ResidentConfig part=cfg;
        part.mode=pass;
        geometryDispatch(c,part,0u,MODEL_FF,rows);
    }
    geometryDispatch(c,cfg,1u,MODEL_D,rows);
    geometryPublishError(c,cfg);
}

void executeGeometryAttention(VkCommandBuffer c,ResidentConfig cfg,VkDeviceSize weightsOffset=VK_WHOLE_SIZE)
{
    for(uint32_t first=0;first<cfg.rows;first+=geometryScoreRows)
    {
        uint32_t rows=std::min(geometryScoreRows,cfg.rows-first);
        ResidentConfig part=cfg;
        part.index=first;
        geometryDispatch(c,part,2u,(geometryScoreStride+geometryScoreSegment-1u)/geometryScoreSegment,RT_ATTENTION_HEADS,rows);
        geometryDispatch(c,part,3u,RT_ATTENTION_HEADS,rows);
        if(weightsOffset!=VK_WHOLE_SIZE)
        {
            sync(c);
            copy(c,geometryScores,uint64_t(rows)*RT_ATTENTION_HEADS*geometryScoreStride*4u,
                 weightsOffset+uint64_t(first)*RT_ATTENTION_HEADS*geometryScoreStride*4u);
            sync(c);
        }
    }
    geometryPublishError(c,cfg);
}

void executeGeometryWord(VkCommandBuffer c,ResidentConfig cfg)
{
    wordMark(c,cfg,0u);
    wordTrace(c,cfg);
    wordMark(c,cfg,1u);
    geometryDispatch(c,cfg,5u,MODEL_V);
    geometryDispatch(c,cfg,6u,128u);
    wordMark(c,cfg,2u);
    geometryDispatch(c,cfg,7u,1u);
    geometryPublishError(c,cfg);
    wordMark(c,cfg,3u);
}

void geometryBench(uint32_t threads,uint32_t mode)
{
    if(!geometrySpace)throw std::runtime_error("Geometry benchmark requires rt_geometry");
    if(!threads)threads=65536u;
    auto c=e.command();begin(c);
    vkCmdFillBuffer(c,geometryStatus->handle,0u,32u,0u);sync(c);
    ResidentConfig cfg{0u,1u,0u,0u,0u,2u,0u,capacity,depth,mode,2u,0u,0u};
    geometryDispatch(c,cfg,mode?10u:9u,(threads+127u)/128u);
    check(vkEndCommandBuffer(c),"end geometry benchmark");
    auto started=Clock::now();e.submit(c);
    auto counters=e.download<uint32_t>(geometryStatus,8u);
    vkFreeCommandBuffers(e.device,e.pool,1u,&c);
    std::vector<char> response;
    append(response,std::vector<uint64_t>{uint64_t(counters[1]),uint64_t(counters[3])});
    append(response,std::vector<double>{ms(started)});
    reply(response);
}

void geometryStatistics()
{
    if(!geometrySpace)throw std::runtime_error("Geometry statistics require rt_geometry");
    auto counters=e.download<uint32_t>(geometryStatus,8u);
    uint64_t queries=uint64_t(counters[1])|(uint64_t(counters[2])<<32u);
    uint64_t historyQueries=uint64_t(counters[3])|(uint64_t(counters[4])<<32u);
    std::vector<char> response;
    append(response,std::vector<uint64_t>{queries,historyQueries,geometrySpace->residentBytes,geometryScores->size,
        RTArithmeticSpace::exponentialSegments,1u,geometryScoreStride,geometryScoreRows,counters[0]});
    reply(response);
}

void geometryAttentionCheck(uint32_t rows,uint32_t count,uint32_t layer,uint32_t queryStart,uint32_t historyStart,
                            const std::vector<float>& q,const std::vector<float>& k,
                            const std::vector<float>& v,const std::vector<float>& gate)
{
    uint32_t initial[8]{0u,0u,1u,0u,0xffffffffu,0xffffffffu,0u,0u};
    e.upload(control,initial,sizeof(initial));
    setup();
    std::vector<uint32_t> marks(size_t(std::max(1u,MODEL_FULL))*capacity*2u,0u);
    for(size_t slot=0;slot<marks.size()/2u;slot++)marks[slot*2u]=0xffffffffu;
    uint32_t historyBase=fullIndex(layer)*capacity;
    for(uint32_t row=0;row<count;row++)marks[(historyBase+(historyStart+row)%capacity)*2u]=historyStart+row;
    e.upload(slots,marks.data(),marks.size()*4u);
    for(uint32_t first=0;first<count;first+=workRows)
    {
        uint32_t batch=std::min(workRows,count-first);
        std::vector<float> keyRows(size_t(batch)*MODEL_A,0.f),valueRows(size_t(batch)*MODEL_W,0.f);
        for(uint32_t row=0;row<batch;row++)for(uint32_t head=0;head<RT_ATTENTION_HEADS;head++)
            for(uint32_t axis=0;axis<RT_ATTENTION_HEAD_DIM;axis++)
            {
                uint32_t compact=(head/(RT_ATTENTION_HEADS/(MODEL_GQA/RT_ATTENTION_HEAD_DIM)))*RT_ATTENTION_HEAD_DIM+axis;
                keyRows[size_t(row)*MODEL_A+head*RT_ATTENTION_HEAD_DIM+axis]=k[size_t(first+row)*MODEL_GQA+compact];
                valueRows[size_t(row)*MODEL_W+MODEL_Q+head*RT_ATTENTION_HEAD_DIM+axis]=v[size_t(first+row)*MODEL_GQA+compact];
            }
        e.upload(key,keyRows.data(),keyRows.size()*4u);
        e.upload(feedForward,valueRows.data(),valueRows.size()*4u);
        auto c=e.command();begin(c);
        ResidentConfig cfg{34u,batch,layer,historyStart+first,0u,2u,0u,capacity,depth,0u,2u,0u,historyStart};
        launch(c,math,cfg,1u,batch);
        cfg.stage=6u;launch(c,math,cfg,1u,batch);
        check(vkEndCommandBuffer(c),"end geometry debug history packing");
        e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
    }
    auto rebuild=e.command();begin(rebuild);
    history->build(rebuild,history->count,true);
    check(vkEndCommandBuffer(rebuild),"end geometry debug history BVH");
    e.submit(rebuild);vkFreeCommandBuffers(e.device,e.pool,1u,&rebuild);
    std::vector<float> queryRows(size_t(rows)*MODEL_A,0.f),gateRows(size_t(rows)*MODEL_W,0.f);
    for(uint32_t row=0;row<rows;row++)
    {
        std::copy_n(q.data()+size_t(row)*MODEL_Q,MODEL_Q,queryRows.data()+size_t(row)*MODEL_A);
        std::copy_n(gate.data()+size_t(row)*MODEL_Q,MODEL_Q,gateRows.data()+size_t(row)*MODEL_W);
    }
    e.upload(query,queryRows.data(),queryRows.size()*4u);
    e.upload(feedForward,gateRows.data(),gateRows.size()*4u);
    prepareGeometryAttention(rows,queryStart,historyStart);
    uint64_t outputBytes=uint64_t(rows)*MODEL_Q*4u,physicalBytes=uint64_t(rows)*RT_ATTENTION_HEADS*8u;
    uint64_t weightBytes=uint64_t(rows)*RT_ATTENTION_HEADS*geometryScoreStride*4u;
    uint64_t packetBytes=uint64_t(count)*HISTORY_BYTES,total=outputBytes+physicalBytes+weightBytes+packetBytes+32u;
    if(total>0xffffffffu-4u)throw std::runtime_error("Geometry debug response exceeds its protocol size");
    ensureReadback(size_t(total+4u));
    e.ensureTimestamps(2u);
    auto c=e.command();begin(c);
    recordRT=false;
    vkCmdResetQueryPool(c,e.timestamps,0u,2u);
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0u);
    ResidentConfig cfg{0u,rows,layer,queryStart,0u,2u,0u,capacity,depth,0u,2u,0u,historyStart};
    executeGeometryAttention(c,cfg,outputBytes+physicalBytes);
    vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
    for(uint32_t row=0;row<rows;row++)
    {
        copy(c,detector,MODEL_Q*4u,uint64_t(row)*MODEL_Q*4u,uint64_t(row)*MODEL_A*4u);
        copy(c,physical,RT_ATTENTION_HEADS*8u,outputBytes+uint64_t(row)*RT_ATTENTION_HEADS*8u,
             uint64_t(row)*std::max(MODEL_STATE_HEADS,uint32_t(RT_ATTENTION_HEADS))*8u);
    }
    uint64_t packetOffset=outputBytes+physicalBytes+weightBytes;
    for(uint32_t row=0;row<count;row++)copy(c,keyGeometry,HISTORY_BYTES,packetOffset+uint64_t(row)*HISTORY_BYTES,
        uint64_t(historyBase+(historyStart+row)%capacity)*HISTORY_BYTES);
    copy(c,geometryStatus,32u,packetOffset+packetBytes);
    copy(c,control,4u,total,12u);hostBarrier(c);
    check(vkEndCommandBuffer(c),"end geometry attention debug");
    auto started=Clock::now();e.submit(c);
    uint64_t times[2]{};
    check(vkGetQueryPoolResults(e.device,e.timestamps,0u,2u,sizeof(times),times,8u,VK_QUERY_RESULT_64_BIT),"geometry attention debug timing");
    vkFreeCommandBuffers(e.device,e.pool,1u,&c);
    uint32_t error=*reinterpret_cast<uint32_t*>(static_cast<char*>(readback->map)+total);
    if(error)throw std::runtime_error("Geometry attention error "+std::to_string(error));
    std::vector<char> response;
    append(response,std::vector<uint32_t>{0x41544731u,rows,count,RT_ATTENTION_HEADS,MODEL_GQA/RT_ATTENTION_HEAD_DIM,
        RT_ATTENTION_HEAD_DIM,HISTORY_BYTES,32u,geometryScoreStride,0u,0u,error});
    append(response,std::vector<double>{double(times[1]-times[0])*e.period/1e6,0.0,0.0,ms(started)});
    auto* data=static_cast<char*>(readback->map);response.insert(response.end(),data,data+total);
    reply(response);
}
