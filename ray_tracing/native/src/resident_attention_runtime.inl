    void ensureAttentionPipelines(){
        if(attentionReceiptPipelines[0])return;
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext=&subgroup;vkGetPhysicalDeviceProperties2(e.physical,&properties);
        const VkSubgroupFeatureFlags required=VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|
            VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        const auto& limits=properties.properties.limits;
        uint32_t sharedBytes=128u*RT_ATTENTION_HEAD_DIM+(attentionBackend==2u?20u:4u)*attentionBlock+attentionBlock/8u+512u;
        if(subgroup.subgroupSize!=32u||(subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)==0u||
           (subgroup.supportedOperations&required)!=required||limits.maxComputeWorkGroupInvocations<512u||
           limits.maxComputeWorkGroupSize[0]<512u||limits.maxComputeSharedMemorySize<sharedBytes)
            throw std::runtime_error("Grouped attention requires 32-lane compute subgroups with basic, vote, ballot and arithmetic operations, local size 512 and sufficient shared memory");
        for(uint32_t operation=0u;operation<3u;operation++)
            attentionReceiptPipelines[operation]=e.pipeline(attentionBackend==2u?"resident_attention_receipt":"resident_attention_dense",layout,operation,attentionBlock,0xffffffffu,wordTile,operation==0u?512u:128u,featureTile);
    }
    void prepareAttention(uint32_t rows,uint32_t position,uint32_t origin=0u){
        ensureAttentionPipelines();
        AttentionReceiptLayout receipt{attentionBlock,attentionBackend==2u};
        uint64_t bytes=receipt.records(rows,position,capacity,origin)*receipt.stride()*4u;
        if(bytes<=attentionReceipts->size)return;
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        if(bytes>properties.limits.maxStorageBufferRange||bytes>UINT32_MAX)
            throw std::runtime_error("Attention receipts exceed the device storage-buffer range; reduce prefill batch");
        e.release(attentionReceipts);
        attentionReceipts=e.buffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkDescriptorBufferInfo info{attentionReceipts->handle,0,attentionReceipts->size};
        std::vector<VkWriteDescriptorSet> writes(depth+1u);
        for(uint32_t layer=0;layer<=depth;layer++){
            auto& write=writes[layer];write={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet=layer==depth?initialSet:layerSets[layer];write.dstBinding=42u;
            write.descriptorCount=1u;write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.pBufferInfo=&info;
        }
        vkUpdateDescriptorSets(e.device,uint32_t(writes.size()),writes.data(),0u,nullptr);
    }
    void receiptDispatch(VkCommandBuffer c,ResidentConfig cfg,uint32_t operation,uint32_t x,uint32_t y=1u){
        if(attentionStatsEnabled)cfg.debug|=65536u;
        cfg.stage=operation;cfg.padding0=2u;std::memcpy(&cfg.padding1,&routeAperture,4u);
        bindCompute(c,attentionReceiptPipelines[operation],layerSets[cfg.layer]);
        vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatch(c,x,y,1u);sync(c);
    }
    void traceAttentionReceipts(VkCommandBuffer c,ResidentConfig cfg){
        cfg.index=0u;cfg.checkpoint=0u;
        receiptDispatch(c,cfg,2u,(cfg.rows+127u)/128u);
        uint32_t finalPosition=cfg.mode==0u?cfg.rows-1u:cfg.mode==1u?cfg.position:cfg.position+cfg.rows-1u;
        uint32_t groups=AttentionReceiptLayout{attentionBlock}.groups(finalPosition,capacity);
        uint32_t stamp=startRT(c,0u);
        receiptDispatch(c,cfg,0u,groups,cfg.rows);endRT(c,stamp);attentionFusedDispatches++;
        for(uint32_t row=0;row<cfg.rows;row++){
            uint32_t position=cfg.mode==0u?row:cfg.mode==1u?cfg.position:cfg.position+row;
            uint32_t active=std::min(position+1u,capacity),rowGroups=(active+attentionBlock-1u)/attentionBlock;
            attentionPlannedRays+=(active+15u)/16u+2u*rowGroups;
            attentionCoveredTerms+=uint64_t(active)*RT_ATTENTION_HEADS;
            if(rowGroups>1u)attentionReportedRecords+=uint64_t(rowGroups)*RT_ATTENTION_HEADS;
        }
    }
    void mergeAttentionReceipts(VkCommandBuffer c,ResidentConfig cfg){
        uint32_t finalPosition=cfg.mode==0u?cfg.rows-1u:cfg.mode==1u?cfg.position:cfg.position+cfg.rows-1u;
        if(AttentionReceiptLayout{attentionBlock}.groups(finalPosition,capacity)==1u)return;
        cfg.index=0u;cfg.checkpoint=0u;
        receiptDispatch(c,cfg,1u,RT_ATTENTION_HEADS,cfg.rows);attentionMergeDispatches++;
    }
    void attentionCheck(uint32_t rows,uint32_t layer){
        uint32_t inputHeader[4];std::cin.read(reinterpret_cast<char*>(inputHeader),sizeof(inputHeader));
        uint32_t count=inputHeader[0],queryStart=inputHeader[1],historyStart=inputHeader[2];
        if(!std::cin||!rows||rows>std::min(workRows,32u)||!count||count>capacity||layer>=depth||!fullLayer(layer)||
           queryStart<historyStart||uint64_t(queryStart)+rows>uint64_t(historyStart)+count||
           uint64_t(historyStart)+count>HISTORY_SLOT_MASK+1u||inputHeader[3])
            throw std::runtime_error("Invalid full-attention diagnostic shape or causal domain");
        std::vector<float> q,k,v,gate;input(q,size_t(rows)*MODEL_Q);input(k,size_t(count)*MODEL_GQA);
        input(v,size_t(count)*MODEL_GQA);input(gate,size_t(rows)*MODEL_Q);
        auto finite=[](const std::vector<float>& values){return std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);});};
        if(!finite(q)||!finite(k)||!finite(v)||!finite(gate))throw std::runtime_error("Attention diagnostic values must be finite");
        attentionDiagnosticDirty=true;
        uint32_t initial[8]{0u,0u,1u,0u,0xffffffffu,0xffffffffu,0u,0u};e.upload(control,initial,32u);setup();
        std::vector<uint32_t> marks(size_t(std::max(1u,MODEL_FULL))*capacity*2u,0u);
        for(uint32_t slot=0;slot<std::max(1u,MODEL_FULL)*capacity;slot++)marks[slot*2u]=0xffffffffu;
        uint32_t historyBase=fullIndex(layer)*capacity;
        for(uint32_t i=0;i<count;i++)marks[(historyBase+(historyStart+i)%capacity)*2u]=historyStart+i;
        e.upload(slots,marks.data(),marks.size()*4u);
        for(uint32_t first=0;first<count;first+=workRows){
            uint32_t batch=std::min(workRows,count-first);
            std::vector<float> keyRows(size_t(batch)*MODEL_A,0.f),contentRows(size_t(batch)*MODEL_W,0.f);
            for(uint32_t row=0;row<batch;row++)for(uint32_t head=0;head<RT_ATTENTION_HEADS;head++)for(uint32_t axis=0;axis<RT_ATTENTION_HEAD_DIM;axis++){
                uint32_t compact=(head/(RT_ATTENTION_HEADS/(MODEL_GQA/RT_ATTENTION_HEAD_DIM)))*RT_ATTENTION_HEAD_DIM+axis;
                keyRows[size_t(row)*MODEL_A+head*RT_ATTENTION_HEAD_DIM+axis]=k[size_t(first+row)*MODEL_GQA+compact];
                contentRows[size_t(row)*MODEL_W+MODEL_Q+head*RT_ATTENTION_HEAD_DIM+axis]=v[size_t(first+row)*MODEL_GQA+compact];
            }
            e.upload(key,keyRows.data(),keyRows.size()*4u);e.upload(feedForward,contentRows.data(),contentRows.size()*4u);
            auto c=e.command();begin(c);ResidentConfig cfg{34u,batch,layer,historyStart+first,0u,2u,0u,capacity,depth,0u,2u,0u,0u};
            launch(c,math,cfg,1u,batch);cfg.stage=6u;launch(c,math,cfg,1u,batch);
            check(vkEndCommandBuffer(c),"end attention diagnostic history packing");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        }
        auto rebuild=e.command();begin(rebuild);history->build(rebuild,history->count,true);
        check(vkEndCommandBuffer(rebuild),"end attention diagnostic geometry build");e.submit(rebuild);vkFreeCommandBuffers(e.device,e.pool,1u,&rebuild);
        std::vector<float> queryRows(size_t(rows)*MODEL_A,0.f),gateRows(size_t(rows)*MODEL_W,0.f);
        for(uint32_t row=0;row<rows;row++){
            std::copy_n(q.data()+size_t(row)*MODEL_Q,MODEL_Q,queryRows.data()+size_t(row)*MODEL_A);
            std::copy_n(gate.data()+size_t(row)*MODEL_Q,MODEL_Q,gateRows.data()+size_t(row)*MODEL_W);
        }
        e.upload(query,queryRows.data(),queryRows.size()*4u);e.upload(feedForward,gateRows.data(),gateRows.size()*4u);
        std::vector<float> emissions(size_t(rows)*4u,0.f);for(uint32_t row=0;row<rows;row++){emissions[row*4u]=.125f;emissions[row*4u+1u]=.25f;}
        e.upload(rays,emissions.data(),emissions.size()*4u);e.upload(control,initial,32u);
        prepareAttention(rows,queryStart,historyStart);AttentionReceiptLayout receipt{attentionBlock,attentionBackend==2u};
        uint32_t records=uint32_t(receipt.records(rows,queryStart,capacity,historyStart)),stride=receipt.stride();
        size_t outputBytes=size_t(rows)*MODEL_Q*4u,physicalBytes=size_t(rows)*RT_ATTENTION_HEADS*8u;
        size_t packetBytes=size_t(count)*HISTORY_BYTES,receiptBytes=size_t(records)*stride*4u;
        size_t total=2u*(outputBytes+physicalBytes)+packetBytes+receiptBytes;
        ensureReadback(total+4u);auto c=e.command();begin(c);recordRT=false;
        ResidentConfig cfg{0u,rows,layer,queryStart,0u,2u,1u,capacity,depth,0u,2u,0u,historyStart};
        vkCmdResetQueryPool(c,e.timestamps,0u,5u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0u);
        receiptDispatch(c,cfg,2u,(rows+127u)/128u);
        uint32_t groups=receipt.groups(queryStart+rows-1u,capacity,historyStart);
        receiptDispatch(c,cfg,0u,groups,rows);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
        receiptDispatch(c,cfg,1u,RT_ATTENTION_HEADS,rows);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);
        size_t offset=0u;
        auto copyOutput=[&](){for(uint32_t row=0;row<rows;row++)copy(c,detector,MODEL_Q*4u,offset+size_t(row)*MODEL_Q*4u,size_t(row)*MODEL_A*4u);offset+=outputBytes;
            for(uint32_t row=0;row<rows;row++)copy(c,physical,RT_ATTENTION_HEADS*8u,offset+size_t(row)*RT_ATTENTION_HEADS*8u,size_t(row)*std::max(MODEL_STATE_HEADS,uint32_t(RT_ATTENTION_HEADS))*8u);offset+=physicalBytes;};
        copyOutput();
        bool legacy=historyStart==0u;
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,3u);
        if(legacy){
            vkCmdFillBuffer(c,control->handle,24u,4u,0u);vkCmdFillBuffer(c,hits->handle,0u,size_t(rows)*(capacity+1u)*4u,0u);sync(c);
            cfg.stage=0u;launch(c,ray,cfg,(rows*((capacity+15u)/16u)+7u)/8u);
            payload(c,cfg);cfg.stage=3u;bindCompute(c,ray[3u],layerSets[layer]);
            vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);vkCmdDispatch(c,RT_ATTENTION_HEADS,rows,1u);sync(c);
        }
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,4u);copyOutput();
        for(uint32_t i=0;i<count;i++)copy(c,keyGeometry,HISTORY_BYTES,offset+size_t(i)*HISTORY_BYTES,size_t(historyBase+(historyStart+i)%capacity)*HISTORY_BYTES);
        offset+=packetBytes;copy(c,attentionReceipts,receiptBytes,offset);offset+=receiptBytes;copy(c,control,4u,offset,12u);
        hostBarrier(c);check(vkEndCommandBuffer(c),"end full-attention diagnostic");auto timing=e.measure(c,4u);
        uint64_t times[5];check(vkGetQueryPoolResults(e.device,e.timestamps,0u,5u,sizeof(times),times,8u,VK_QUERY_RESULT_64_BIT),"read full-attention diagnostic timestamps");
        vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        uint32_t error=*reinterpret_cast<uint32_t*>(static_cast<char*>(readback->map)+total);
        if(error)throw std::runtime_error("Full-attention receipt error "+std::to_string(error));
        std::vector<char> output;append(output,std::vector<uint32_t>{0x41545231u,rows,count,RT_ATTENTION_HEADS,MODEL_GQA/RT_ATTENTION_HEAD_DIM,RT_ATTENTION_HEAD_DIM,HISTORY_BYTES,attentionBlock,stride,records,uint32_t(legacy),error});
        append(output,std::vector<double>{double(times[1]-times[0])*e.period/1e6,double(times[2]-times[1])*e.period/1e6,legacy?double(times[4]-times[3])*e.period/1e6:0.0,timing.wall});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+total);reply(output);
    }
    static constexpr uint32_t attentionStatCount=16u;
    uint32_t attentionStatWords()const{return attentionStatsEnabled?2u*attentionStatCount*std::max(1u,MODEL_FULL):0u;}
    void attentionStatistics(uint32_t reset){
        if(!attentionStatsEnabled||attentionBackend!=2u||reset>1u)throw std::runtime_error("Attention statistics require grouped_rt and RT_ATTENTION_STATS=1");
        size_t bytes=size_t(attentionStatWords())*4u,offset=size_t(11u+capacity+65536u)*4u;
        ensureReadback(bytes);auto c=e.command();begin(c);copy(c,control,bytes,0u,offset);
        if(reset){vkCmdFillBuffer(c,control->handle,offset,bytes,0u);sync(c);}
        hostBarrier(c);check(vkEndCommandBuffer(c),"end attention statistics");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        std::vector<char> output;append(output,std::vector<uint32_t>{0x41545331u,attentionStatCount,MODEL_FULL,attentionBlock});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);reply(output);
    }
