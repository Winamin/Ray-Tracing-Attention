    static constexpr uint32_t attentionClassDomains=2u*(MODEL_GQA/RT_ATTENTION_HEAD_DIM);
    static constexpr uint32_t attentionClassDirectoryHeaderWords=3u;
    static constexpr uint32_t attentionClassDirectoryEntryWords=2u;
    static constexpr uint32_t attentionClassTempDomainWords=4u*RT_ATTENTION_HEAD_DIM+RT_ATTENTION_HEAD_DIM/4u+1u;
    static constexpr uint32_t attentionClassMemberWords=RT_ATTENTION_HEAD_DIM/4u;
    static uint64_t attentionClassDirectoryWords(uint32_t slots){return attentionClassDirectoryHeaderWords+uint64_t(slots)*attentionClassDomains*attentionClassDirectoryEntryWords;}
    static uint32_t attentionClassDomainWords(uint32_t count){return 2u*count+attentionClassMemberWords;}
    uint64_t attentionClassBytes()const{
        uint64_t bytes=0u;for(const auto& layer:attentionClassLayers)if(layer.buffer)bytes+=layer.buffer->size;
        return bytes?bytes:(attentionClassDummy?attentionClassDummy->size:0u);
    }
    std::vector<uint64_t> attentionClassResourceDetails(){
        refreshAttentionClassMirrors();
        uint64_t directoryBytes=0u,representatives=0u,domains=0u;
        for(const auto& layer:attentionClassLayers){
            if(layer.buffer)directoryBytes+=attentionClassDirectoryWords(attentionClassSlots)*4u;
            for(size_t entry=1u;entry<layer.directory.size();entry+=2u){
                representatives+=layer.directory[entry];domains+=layer.directory[entry]!=0u;
            }
        }
        return {directoryBytes,representatives*4u,representatives*4u,
            domains*RT_ATTENTION_HEAD_DIM,attentionClassScratch->size,attentionClassCounts->size,representatives};
    }
    static void copyAttentionClassRegions(VkCommandBuffer c,Buffer* source,Buffer* target,uint32_t count,const VkBufferCopy* regions){
        if(!source||!target)throw std::runtime_error("Attention class copy requires live buffers");
        for(uint32_t i=0u;i<count;i++){
            const auto& region=regions[i];
            if(region.srcOffset>source->size||region.size>source->size-region.srcOffset||
               region.dstOffset>target->size||region.size>target->size-region.dstOffset)
                throw std::runtime_error("Attention class copy exceeds its buffer bounds");
        }
        if(count)vkCmdCopyBuffer(c,source->handle,target->handle,count,regions);
    }
    static void copyAttentionClassRegion(VkCommandBuffer c,Buffer* source,Buffer* target,VkBufferCopy region){
        copyAttentionClassRegions(c,source,target,1u,&region);
    }
    Buffer* allocateAttentionClassLayer(uint32_t slots,uint32_t arenaWords){
        uint64_t bytes=(attentionClassDirectoryWords(slots)+std::max(1u,arenaWords))*4u;
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        if(bytes>properties.limits.maxStorageBufferRange||bytes>UINT32_MAX)
            throw std::runtime_error("Local attention classes exceed the device storage-buffer range");
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        return e.buffer(bytes,usage,false,true);
    }
    void recordAttentionGap(uint64_t first){
        if(attentionSegmentEnd&&attentionProfileSlot!=~0u){
            if(attentionProfileGaps.size()<=attentionProfileSlot)attentionProfileGaps.resize(attentionProfileSlot+1u);
            attentionProfileGaps[attentionProfileSlot]+=double(first-attentionSegmentEnd)*e.period/1e6;
        }
    }
    void submitAttentionSegment(VkCommandBuffer& c){
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,attentionSubmitTimestamps,1u);
        check(vkEndCommandBuffer(c),"end attention resource segment");
        attentionSplitWait+=e.submit(c);attentionSplitSubmissions++;
        uint64_t stamps[2];check(vkGetQueryPoolResults(e.device,attentionSubmitTimestamps,0u,2u,sizeof(stamps),stamps,8u,VK_QUERY_RESULT_64_BIT),"read attention resource segment");
        recordAttentionGap(stamps[0]);attentionSegmentEnd=stamps[1];
        attentionSplitGPU+=double(stamps[1]-stamps[0])*e.period/1e6;
        vkFreeCommandBuffers(e.device,e.pool,1u,&c);c=VK_NULL_HANDLE;
    }
    Measure attentionTiming(Measure final,double previousGPU,double previousWait){
        if(attentionBackend!=4u)return final;
        uint64_t first,last;
        check(vkGetQueryPoolResults(e.device,attentionSubmitTimestamps,0u,1u,sizeof(first),&first,8u,VK_QUERY_RESULT_64_BIT),"read final attention segment start");
        check(vkGetQueryPoolResults(e.device,e.timestamps,1u,1u,sizeof(last),&last,8u,VK_QUERY_RESULT_64_BIT),"read final attention segment end");
        recordAttentionGap(first);
        final.gpu=attentionSplitGPU-previousGPU+double(last-first)*e.period/1e6;
        final.wall+=attentionSplitWait-previousWait;return final;
    }
    bool refreshAttentionClassMirrors(VkCommandBuffer& c,bool resume=true){
        if(attentionBackend!=4u)return false;
        std::vector<uint32_t> dirty;
        for(uint32_t full=0u;full<attentionClassLayers.size();full++)
            if(attentionClassLayers[full].mirrorDirty)dirty.push_back(full);
        if(dirty.empty())return false;
        uint64_t words=attentionClassDirectoryWords(attentionClassSlots),bytes=words*4u*dirty.size();
        if(attentionClassCounts->size<bytes){
            e.release(attentionClassCounts);
            attentionClassCounts=e.buffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        }
        sync(c);
        for(size_t i=0u;i<dirty.size();i++){
            VkBufferCopy directory{0u,uint64_t(i)*words*4u,words*4u};
            copyAttentionClassRegion(c,attentionClassLayers[dirty[i]].buffer,attentionClassCounts,directory);
        }
        hostBarrier(c);submitAttentionSegment(c);
        const auto* copied=static_cast<const uint32_t*>(attentionClassCounts->map);
        for(size_t i=0u;i<dirty.size();i++){
            auto& layer=attentionClassLayers[dirty[i]];const auto* source=copied+i*words;
            if(source[0]!=attentionClassSlots||source[2]!=layer.arenaWords||source[1]>source[2])
                throw std::runtime_error("Invalid GPU attention class arena metadata");
            layer.usedWords=layer.reservedWords=source[1];
            layer.directory.assign(source+attentionClassDirectoryHeaderWords,source+words);
            layer.allocations.assign(layer.directory.size()/2u,{});layer.freeBlocks.clear();
            for(size_t entry=0u;entry<layer.allocations.size();entry++){
                uint32_t count=layer.directory[2u*entry+1u];if(!count)continue;
                uint32_t offset=layer.directory[2u*entry],size=attentionClassDomainWords(count);
                if(count>RT_ATTENTION_HEAD_DIM||offset>layer.usedWords||size>layer.usedWords-offset)
                    throw std::runtime_error("Invalid GPU compact attention class directory");
                layer.allocations[entry]={offset,size};
            }
            layer.mirrorDirty=false;
        }
        if(resume){c=e.command();begin(c);}return true;
    }
    void refreshAttentionClassMirrors(){
        if(attentionBackend!=4u||std::none_of(attentionClassLayers.begin(),attentionClassLayers.end(),
            [](const auto& layer){return layer.mirrorDirty;}))return;
        auto c=e.command();begin(c);refreshAttentionClassMirrors(c,false);
    }
    uint32_t attentionClassArenaCapacity(uint64_t required,uint32_t previous,uint32_t directorySlots){
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        uint64_t limit=std::min<uint64_t>(properties.limits.maxStorageBufferRange,UINT32_MAX)/4u;
        uint64_t directory=attentionClassDirectoryWords(directorySlots);
        if(directory>=limit||required>limit-directory)
            throw std::runtime_error("Local attention classes exceed the device storage-buffer range");
        uint64_t next=std::max(required,uint64_t(previous)+previous/2u);
        next=(next+255u)&~uint64_t(255u);
        return uint32_t(std::min(next,limit-directory));
    }
    void resizeAttentionClasses(uint32_t required,bool preserve=true){
        if(attentionBackend!=4u)return;
        uint32_t next=required?std::min(capacity,(required+31u)/32u*32u):0u;
        if(next==attentionClassSlots&&preserve)return;
        if(preserve)refreshAttentionClassMirrors();
        if(!next){
            for(auto& layer:attentionClassLayers){e.release(layer.buffer);layer={};}
            attentionClassSlots=0u;
            for(uint32_t layer=0;layer<depth;layer++)bind(layerSets[layer],false,layer);bind(initialSet,true);return;
        }
        uint32_t copied=preserve?std::min(next,attentionClassSlots):0u;
        std::vector<Buffer*> retired(attentionClassLayers.size()),replacements(attentionClassLayers.size());
        auto c=e.command();begin(c);
        for(uint32_t index=0;index<attentionClassLayers.size();index++){
            auto& layer=attentionClassLayers[index];retired[index]=layer.buffer;
            uint32_t arena=preserve?layer.arenaWords:0u;
            if(!preserve||!layer.buffer){
                uint64_t batchWords=uint64_t(std::min(next,workRows))*attentionClassDomains*
                    attentionClassDomainWords(RT_ATTENTION_HEAD_DIM);
                arena=attentionClassArenaCapacity(batchWords,0u,next);
            }
            replacements[index]=allocateAttentionClassLayer(next,arena);
            vkCmdFillBuffer(c,replacements[index]->handle,0u,replacements[index]->size,0u);
            sync(c);
            uint32_t header[3]{next,preserve?layer.usedWords:0u,arena};
            vkCmdUpdateBuffer(c,replacements[index]->handle,0u,sizeof(header),header);
            if(copied&&layer.buffer){
                VkBufferCopy directory{attentionClassDirectoryHeaderWords*4u,attentionClassDirectoryHeaderWords*4u,uint64_t(copied)*attentionClassDomains*attentionClassDirectoryEntryWords*4u};
                copyAttentionClassRegion(c,layer.buffer,replacements[index],directory);
                if(layer.usedWords){VkBufferCopy payload{attentionClassDirectoryWords(attentionClassSlots)*4u,
                    attentionClassDirectoryWords(next)*4u,uint64_t(layer.usedWords)*4u};
                    copyAttentionClassRegion(c,layer.buffer,replacements[index],payload);}
            }
            if(preserve){
                std::vector<uint32_t> directory(size_t(next)*attentionClassDomains*2u,0u);
                size_t words=size_t(copied)*attentionClassDomains*2u;
                if(words)std::copy_n(layer.directory.begin(),std::min(words,layer.directory.size()),directory.begin());
                layer.directory=std::move(directory);layer.allocations.resize(size_t(next)*attentionClassDomains);
            }else{
                layer.usedWords=layer.reservedWords=0u;layer.freeBlocks.clear();layer.mirrorDirty=false;
                layer.directory.assign(size_t(next)*attentionClassDomains*2u,0u);
                layer.allocations.assign(size_t(next)*attentionClassDomains,{});
            }
            layer.arenaWords=arena;
        }
        sync(c);check(vkEndCommandBuffer(c),"end attention class directory resize");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        for(uint32_t index=0;index<attentionClassLayers.size();index++){
            e.release(retired[index]);attentionClassLayers[index].buffer=replacements[index];
        }
        attentionClassSlots=next;
        for(uint32_t layer=0;layer<depth;layer++)bind(layerSets[layer],false,layer);bind(initialSet,true);
    }
    void growAttentionClassLayer(uint32_t full,uint32_t required){
        auto& layer=attentionClassLayers[full];if(required<=layer.arenaWords)return;
        uint32_t next=attentionClassArenaCapacity(required,layer.arenaWords,attentionClassSlots);auto* replacement=allocateAttentionClassLayer(attentionClassSlots,next);
        auto c=e.command();begin(c);vkCmdFillBuffer(c,replacement->handle,0u,replacement->size,0u);
        sync(c);
        uint32_t header[3]{attentionClassSlots,layer.usedWords,next};
        vkCmdUpdateBuffer(c,replacement->handle,0u,sizeof(header),header);
        if(layer.buffer){
            VkBufferCopy directory{attentionClassDirectoryHeaderWords*4u,attentionClassDirectoryHeaderWords*4u,uint64_t(attentionClassSlots)*attentionClassDomains*2u*4u};
            copyAttentionClassRegion(c,layer.buffer,replacement,directory);
            // Allocation has already increased usedWords to the new demand.
            // The retired buffer only contains the old arena, not that tail.
            uint32_t copied=std::min(layer.usedWords,layer.arenaWords);
            if(copied){VkBufferCopy payload{attentionClassDirectoryWords(attentionClassSlots)*4u,
                attentionClassDirectoryWords(attentionClassSlots)*4u,uint64_t(copied)*4u};
                copyAttentionClassRegion(c,layer.buffer,replacement,payload);}
        }
        sync(c);check(vkEndCommandBuffer(c),"end attention class payload growth");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        e.release(layer.buffer);layer.buffer=replacement;layer.arenaWords=next;
        for(uint32_t actual=0;actual<depth;actual++)if(fullLayer(actual)&&fullIndex(actual)==full)bind(layerSets[actual],false,actual);
    }
    void compactAttentionClassLayer(uint32_t full,uint32_t appendWords){
        auto& layer=attentionClassLayers[full];
        if(layer.mirrorDirty||!attentionClassRestoreCopies.empty())
            throw std::runtime_error("Attention class compaction requires a completed live directory");
        std::vector<uint32_t> directory(layer.directory.size(),0u);
        std::vector<AttentionClassAllocation> allocations(layer.allocations.size());
        std::vector<VkBufferCopy> copies;uint64_t used=0u;
        uint64_t payloadBase=attentionClassDirectoryWords(attentionClassSlots)*4u;
        for(size_t entry=0u;entry<layer.allocations.size();entry++){
            if(!layer.directory[2u*entry+1u])continue;
            std::array<uint32_t,2> record{};
            if(!validAttentionClassEntry(full,entry,&record))
                throw std::runtime_error("Invalid compact attention class during arena growth");
            uint32_t words=attentionClassDomainWords(record[1]);
            if(used+words>UINT32_MAX)throw std::runtime_error("Attention class payload offset overflow");
            directory[2u*entry]=uint32_t(used);directory[2u*entry+1u]=record[1];
            allocations[entry]={uint32_t(used),words};
            VkBufferCopy region{payloadBase+uint64_t(record[0])*4u,payloadBase+used*4u,uint64_t(words)*4u};
            if(!copies.empty()&&copies.back().srcOffset+copies.back().size==region.srcOffset&&
               copies.back().dstOffset+copies.back().size==region.dstOffset)copies.back().size+=region.size;
            else copies.push_back(region);
            used+=words;
        }
        uint64_t required=used+appendWords;
        uint32_t next=required<=layer.arenaWords?layer.arenaWords:attentionClassArenaCapacity(required,layer.arenaWords,attentionClassSlots);
        auto* replacement=allocateAttentionClassLayer(attentionClassSlots,next);
        auto c=e.command();begin(c);vkCmdFillBuffer(c,replacement->handle,0u,replacement->size,0u);sync(c);
        uint32_t header[3]{attentionClassSlots,uint32_t(used),next};
        vkCmdUpdateBuffer(c,replacement->handle,0u,sizeof(header),header);
        for(size_t first=0u;first<directory.size();first+=16384u){
            size_t count=std::min<size_t>(16384u,directory.size()-first);
            vkCmdUpdateBuffer(c,replacement->handle,(attentionClassDirectoryHeaderWords+first)*4u,count*4u,directory.data()+first);
        }
        copyAttentionClassRegions(c,layer.buffer,replacement,uint32_t(copies.size()),copies.data());sync(c);
        check(vkEndCommandBuffer(c),"end attention class arena compaction");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        e.release(layer.buffer);layer.buffer=replacement;layer.arenaWords=next;
        layer.usedWords=layer.reservedWords=uint32_t(used);layer.directory=std::move(directory);
        layer.allocations=std::move(allocations);layer.freeBlocks.clear();
        for(uint32_t actual=0u;actual<depth;actual++)if(fullLayer(actual)&&fullIndex(actual)==full)bind(layerSets[actual],false,actual);
    }
    void releaseAttentionClassBlock(AttentionClassLayerCache& layer,AttentionClassAllocation block){
        if(!block.words)return;layer.freeBlocks.push_back(block);
        std::sort(layer.freeBlocks.begin(),layer.freeBlocks.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
        std::vector<AttentionClassAllocation> merged;
        for(auto item:layer.freeBlocks){
            if(!merged.empty()&&merged.back().offset+merged.back().words==item.offset)merged.back().words+=item.words;
            else merged.push_back(item);
        }
        layer.freeBlocks=std::move(merged);
    }
    uint32_t allocateAttentionClassBlock(AttentionClassLayerCache& layer,uint32_t words,AttentionClassAllocation& current){
        if(current.words>=words)return current.offset;
        releaseAttentionClassBlock(layer,current);current={};
        for(auto it=layer.freeBlocks.begin();it!=layer.freeBlocks.end();++it)if(it->words>=words){
            uint32_t offset=it->offset;it->offset+=words;it->words-=words;if(!it->words)layer.freeBlocks.erase(it);
            current={offset,words};return offset;
        }
        if(uint64_t(layer.usedWords)+words>UINT32_MAX)throw std::runtime_error("Attention class payload offset overflow");
        uint32_t offset=layer.usedWords;layer.usedWords+=words;layer.reservedWords=layer.usedWords;current={offset,words};return offset;
    }
    void resetAttentionClassPending(){
        for(auto& ranges:pendingAttentionClassRanges)ranges.clear();
    }
    bool validAttentionClassDomain(uint32_t full,uint32_t slot)const{
        if(full>=MODEL_FULL||full>=attentionClassLayers.size()||slot>=attentionClassSlots)return false;
        const auto& layer=attentionClassLayers[full];size_t entry=size_t(slot)*attentionClassDomains;
        if(!layer.buffer||attentionClassDirectoryWords(attentionClassSlots)*4u>layer.buffer->size)return false;
        for(uint32_t domain=0u;domain<attentionClassDomains;domain++)
            if(!validAttentionClassEntry(full,entry+domain))return false;
        return true;
    }
    bool validAttentionClassEntry(uint32_t full,size_t entry,std::array<uint32_t,2>* record=nullptr)const{
        if(full>=MODEL_FULL||full>=attentionClassLayers.size())return false;
        const auto& layer=attentionClassLayers[full];
        if(!layer.buffer||layer.mirrorDirty||entry>=layer.allocations.size()||entry>=layer.directory.size()/2u||
           entry>=uint64_t(attentionClassSlots)*attentionClassDomains)return false;
        uint32_t offset=layer.directory[2u*entry],count=layer.directory[2u*entry+1u];
        if(count<1u||count>RT_ATTENTION_HEAD_DIM)return false;
        uint32_t words=attentionClassDomainWords(count);const auto& allocation=layer.allocations[entry];
        if(offset!=allocation.offset||words>allocation.words||offset>layer.usedWords||words>layer.usedWords-offset||
           offset>layer.arenaWords||words>layer.arenaWords-offset||
           (attentionClassDirectoryWords(attentionClassSlots)+uint64_t(offset)+words)*4u>layer.buffer->size)return false;
        if(record)*record={offset,count};return true;
    }
    bool sameLiveAttentionClasses(const std::shared_ptr<SharedHistoryCache::Entry>& entry){
        if(attentionBackend!=4u||!entry||!entry->addressReady||!gpuFieldReady||attentionDiagnosticDirty)return false;
        uint32_t tokens=uint32_t(entry->input.size());
        if(!tokens||tokens>capacity||gpuFieldPosition!=tokens||gpuFieldAddress.key!=entry->address.key||
           gpuFieldAddress.chart!=entry->address.chart)return false;
        refreshAttentionClassMirrors();
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            uint32_t layer=MODEL_FULL_LAYERS[full];
            if(!pendingAttentionClassRanges[layer].empty())return false;
            for(uint32_t slot=0u;slot<tokens;slot++)if(!validAttentionClassDomain(full,slot))return false;
        }
        return true;
    }
    bool validAttentionBlockPayload(const SharedHistoryCache::Block& block)const{
        if(!block.ready||!block.attentionReady||!block.count||block.count>SharedHistoryCache::blockTokens||
           block.first%SharedHistoryCache::blockTokens||uint64_t(block.first)+block.count>capacity||
           block.attentionDomains!=attentionClassDomains||block.attentionHeadDim!=RT_ATTENTION_HEAD_DIM||
           block.attentionKVWidth!=MODEL_GQA||block.attentionPacketBytes!=HISTORY_BYTES||
           block.attentionLayers.size()!=MODEL_FULL||block.attentionFullLayers.size()!=MODEL_FULL||
           !block.packets||block.packets->size!=uint64_t(MODEL_FULL)*block.count*HISTORY_BYTES||
           !block.marks||block.marks->size!=uint64_t(MODEL_FULL)*block.count*8u)return false;
        size_t domains=size_t(block.count)*attentionClassDomains;
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            const auto& layer=block.attentionLayers[full];uint64_t words=0u;
            if(block.attentionFullLayers[full]!=MODEL_FULL_LAYERS[full]||!layer.payload||layer.directory.size()!=domains)return false;
            for(const auto& entry:layer.directory){
                uint32_t count=entry[1];if(count<1u||count>RT_ATTENTION_HEAD_DIM||entry[0]!=words)return false;
                words+=attentionClassDomainWords(count);if(words>UINT32_MAX)return false;
            }
            if(words*4u!=layer.payload->size||layer.payload->allocationSize<layer.payload->size)return false;
        }
        return true;
    }
    void saveAttentionBlock(VkCommandBuffer& c,SharedHistoryCache::Block& block,bool matchingPackets){
        if(attentionBackend!=4u||!sharedHistory||block.attentionReady||!matchingPackets||!block.ready||
           !MODEL_FULL||!block.count||block.count>SharedHistoryCache::blockTokens||
           block.first%SharedHistoryCache::blockTokens||uint64_t(block.first)+block.count>capacity)return;
        uint64_t used=sharedHistory->blockBytes(),snapshots=sharedHistory->snapshotBytes();
        if(used>sharedHistory->budget||snapshots>sharedHistory->budget-used)return;
        for(uint32_t full=0u;full<MODEL_FULL;full++)
            if(!pendingAttentionClassRanges[MODEL_FULL_LAYERS[full]].empty())return;
        refreshAttentionClassMirrors(c);
        uint64_t available=sharedHistory->budget-used-snapshots,requested=0u;
        std::vector<SharedHistoryCache::Block::AttentionLayer> saved(MODEL_FULL);
        std::vector<std::vector<VkBufferCopy>> copies(MODEL_FULL);
        std::vector<uint64_t> layerBytes(MODEL_FULL);
        std::vector<uint32_t> fullLayers(MODEL_FULL_LAYERS,MODEL_FULL_LAYERS+MODEL_FULL);
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            uint64_t words=0u;auto& directory=saved[full].directory;
            for(uint32_t token=0u;token<block.count;token++)for(uint32_t domain=0u;domain<attentionClassDomains;domain++){
                std::array<uint32_t,2> entry{};size_t index=size_t(block.first+token)*attentionClassDomains+domain;
                if(!validAttentionClassEntry(full,index,&entry))return;
                uint32_t size=attentionClassDomainWords(entry[1]);
                if(words+size>UINT32_MAX)return;
                directory.push_back({uint32_t(words),entry[1]});
                VkBufferCopy copy{(attentionClassDirectoryWords(attentionClassSlots)+entry[0])*4u,words*4u,uint64_t(size)*4u};
                auto& regions=copies[full];
                if(!regions.empty()&&regions.back().srcOffset+regions.back().size==copy.srcOffset&&
                   regions.back().dstOffset+regions.back().size==copy.dstOffset)regions.back().size+=copy.size;
                else regions.push_back(copy);
                words+=size;
            }
            layerBytes[full]=words*4u;if(!layerBytes[full]||layerBytes[full]>available-requested)return;
            requested+=layerBytes[full];
        }
        // Standalone transfer buffers make the optional cache's physical byte
        // charge explicit instead of opening a new shared memory pool. Publish
        // all layers together only after allocation rounding also fits budget.
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        uint64_t allocations=0u;
        for(const auto& buffer:e.buffers)allocations+=buffer->ownsMemory&&buffer->memory!=VK_NULL_HANDLE;
        for(const auto& pool:e.memoryPools)allocations+=pool.memory!=VK_NULL_HANDLE;
        // Keep enough allocation slots for a complete live directory resize
        // plus staging; optional cache exhaustion must only create a gap.
        uint64_t limit=properties.limits.maxMemoryAllocationCount,reserve=MODEL_FULL+8u;
        if(allocations>limit||reserve>limit-allocations||MODEL_FULL>limit-allocations-reserve)return;
        uint64_t allocated=0u;bool fits=true;
        try{
            for(uint32_t full=0u;full<MODEL_FULL;full++){
                saved[full].payload=e.buffer(layerBytes[full],VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,false,true);
                uint64_t bytes=saved[full].payload->allocationSize;
                if(bytes>available-allocated){fits=false;break;}allocated+=bytes;
            }
        }catch(...){for(auto& layer:saved)e.release(layer.payload);throw;}
        if(!fits){for(auto& layer:saved)e.release(layer.payload);return;}
        sync(c);
        for(uint32_t full=0u;full<MODEL_FULL;full++)
            copyAttentionClassRegions(c,attentionClassLayers[full].buffer,saved[full].payload,uint32_t(copies[full].size()),copies[full].data());
        block.attentionDomains=attentionClassDomains;block.attentionHeadDim=RT_ATTENTION_HEAD_DIM;
        block.attentionKVWidth=MODEL_GQA;block.attentionPacketBytes=HISTORY_BYTES;
        block.attentionFullLayers=std::move(fullLayers);
        block.attentionLayers=std::move(saved);block.attentionReady=true;
    }
    void prepareAttentionClassAttachment(const std::shared_ptr<SharedHistoryCache::Entry>& entry,bool sameLive){
        resetAttentionClassPending();attentionClassRestoreCopies.clear();
        if(attentionBackend!=4u||!entry||sameLive)return;
        uint32_t tokens=uint32_t(entry->input.size()),covered=0u;
        std::vector<AttentionClassRange> gaps;
        std::vector<std::shared_ptr<SharedHistoryCache::Block>> cachedBlocks;
        for(const auto& block:entry->blocks){
            if(!block||!block->ready||block->first!=covered||!block->count||block->count>SharedHistoryCache::blockTokens||
               block->first%SharedHistoryCache::blockTokens||uint64_t(covered)+block->count>tokens||tokens>capacity)
                throw std::runtime_error("Attention class attachment has an invalid history block domain");
            covered+=block->count;
            if(validAttentionBlockPayload(*block)){cachedBlocks.push_back(block);continue;}
            if(!gaps.empty()&&gaps.back().first+gaps.back().count==block->first)gaps.back().count+=block->count;
            else gaps.push_back({block->first,block->count});
        }
        if(!tokens||covered!=tokens)throw std::runtime_error("Attention class attachment does not cover its prefix");
        bool cached=!cachedBlocks.empty();
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            auto& layer=attentionClassLayers[full];uint32_t actual=MODEL_FULL_LAYERS[full];
            pendingAttentionClassRanges[actual]=gaps;
            for(const auto& block:cachedBlocks){
                const auto& source=block->attentionLayers[full];
                for(uint32_t token=0u;token<block->count;token++)for(uint32_t domain=0u;domain<attentionClassDomains;domain++){
                    size_t local=size_t(token)*attentionClassDomains+domain,live=size_t(block->first+token)*attentionClassDomains+domain;
                    if(live>=layer.allocations.size()||live>=layer.directory.size()/2u)
                        throw std::runtime_error("Restored attention class directory is out of range");
                    uint32_t count=source.directory[local][1],words=attentionClassDomainWords(count);
                    uint32_t offset=allocateAttentionClassBlock(layer,words,layer.allocations[live]);
                    layer.directory[2u*live]=offset;layer.directory[2u*live+1u]=count;
                    VkBufferCopy region{uint64_t(source.directory[local][0])*4u,
                        (attentionClassDirectoryWords(attentionClassSlots)+offset)*4u,uint64_t(words)*4u};
                    if(!attentionClassRestoreCopies.empty()&&attentionClassRestoreCopies.back().source==source.payload&&
                       attentionClassRestoreCopies.back().full==full&&
                       attentionClassRestoreCopies.back().region.srcOffset+attentionClassRestoreCopies.back().region.size==region.srcOffset&&
                       attentionClassRestoreCopies.back().region.dstOffset+attentionClassRestoreCopies.back().region.size==region.dstOffset)
                        attentionClassRestoreCopies.back().region.size+=region.size;
                    else attentionClassRestoreCopies.push_back({source.payload,full,region});
                }
            }
            if(cached)growAttentionClassLayer(full,layer.usedWords);
        }
        for(const auto& copy:attentionClassRestoreCopies){
            auto* target=attentionClassLayers[copy.full].buffer;
            if(!copy.source||!target||copy.region.srcOffset>copy.source->size||copy.region.size>copy.source->size-copy.region.srcOffset||
               copy.region.dstOffset>target->size||copy.region.size>target->size-copy.region.dstOffset)
                throw std::runtime_error("Restored attention class payload is out of range");
        }
    }
    void restoreAttentionClassAttachment(VkCommandBuffer c,uint32_t tokens){
        if(attentionBackend!=4u||attentionClassRestoreCopies.empty())return;
        std::vector<VkBufferCopy> regions;
        for(size_t first=0u;first<attentionClassRestoreCopies.size();){
            const auto& source=attentionClassRestoreCopies[first];regions.clear();size_t next=first;
            while(next<attentionClassRestoreCopies.size()&&attentionClassRestoreCopies[next].source==source.source&&
                  attentionClassRestoreCopies[next].full==source.full)regions.push_back(attentionClassRestoreCopies[next++].region);
            copyAttentionClassRegions(c,source.source,attentionClassLayers[source.full].buffer,uint32_t(regions.size()),regions.data());
            first=next;
        }
        uint64_t words=uint64_t(tokens)*attentionClassDomains*2u;
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            const auto& layer=attentionClassLayers[full];
            if(!layer.buffer||words>layer.directory.size()||(attentionClassDirectoryHeaderWords+words)*4u>layer.buffer->size)
                throw std::runtime_error("Restored attention class directory exceeds its buffer");
            uint32_t header[3]{attentionClassSlots,layer.usedWords,layer.arenaWords};
            vkCmdUpdateBuffer(c,layer.buffer->handle,0u,sizeof(header),header);
            // vkCmdUpdateBuffer accepts at most 64 KiB. Batch contiguous native
            // entries without changing their token/KV-head/causal ordering.
            for(uint64_t first=0u;first<words;first+=16384u){uint64_t count=std::min<uint64_t>(16384u,words-first);
                vkCmdUpdateBuffer(c,layer.buffer->handle,(attentionClassDirectoryHeaderWords+first)*4u,count*4u,layer.directory.data()+first);}
        }
        sync(c);attentionClassRestoreCopies.clear();
    }
    void prepareStaticAttentionClassScene(){
        if(attentionClassScene)return;
        uint64_t before=e.physicalDeviceAllocated;
        attentionClassScene=std::make_unique<AttentionClassGeometry>(e);
        e.release(e.uploadStaging);
        attentionClassPeakSceneBytes=std::max(attentionClassPeakSceneBytes,
            std::max(attentionClassScene->peakBuildBytes,e.physicalDeviceAllocated-before));
        attentionClassBuilds++;
        for(uint32_t layer=0;layer<depth;layer++)bind(layerSets[layer],false,layer);
        bind(initialSet,true);
    }
    void classifyAttentionHistory(VkCommandBuffer& c,ResidentConfig cfg){
        if(attentionBackend!=4u)return;
        auto ranges=std::move(pendingAttentionClassRanges[cfg.layer]);pendingAttentionClassRanges[cfg.layer].clear();
        for(const auto& range:ranges){
            if(!range.count||uint64_t(range.first)+range.count>capacity)throw std::runtime_error("Invalid missing attention class interval");
            for(uint32_t offset=0u;offset<range.count;offset+=workRows){
                auto batch=cfg;batch.position=range.first+offset;batch.rows=std::min(workRows,range.count-offset);batch.mode=2u;
                classifyAttentionHistory(c,batch);
            }
        }
        if(!attentionClassPipelines[0])for(uint32_t operation=0u;operation<5u;operation++)
            attentionClassPipelines[operation]=e.pipeline("resident_attention_classes",
                layout,operation,1u,0xffffffffu,wordTile,128u,featureTile);
        uint32_t first=cfg.position%capacity;
        uint32_t required=cfg.rows>capacity-first?capacity:first+cfg.rows;
        if(!attentionClassScene||required>attentionClassSlots){
            submitAttentionSegment(c);
            attentionProfileSlot=cfg.debug==2u?cfg.step*depth+cfg.layer:~0u;
            if(required>attentionClassSlots)resizeAttentionClasses(required);
            prepareStaticAttentionClassScene();
            c=e.command();begin(c);
        }
        if(attentionStatsEnabled)cfg.debug|=65536u;
        uint32_t full=fullIndex(cfg.layer);auto& layer=attentionClassLayers[full];
        // A batch can append no more than DH representatives per domain. This
        // reservation is only a bound; GPU records still contain their actual D.
        uint64_t reservation=uint64_t(cfg.rows)*attentionClassDomains*attentionClassDomainWords(RT_ATTENTION_HEAD_DIM);
        if(reservation>UINT32_MAX)throw std::runtime_error("Attention class batch reservation overflow");
        if(uint64_t(layer.reservedWords)+reservation>layer.arenaWords){
            attentionProfileSlot=cfg.debug==2u?cfg.step*depth+cfg.layer:~0u;
            if(!refreshAttentionClassMirrors(c,false))submitAttentionSegment(c);
            // Reclaim obsolete compact payloads only when the exact GPU cursor
            // cannot fit the batch, after preceding commands have completed.
            if(uint64_t(layer.usedWords)+reservation>layer.arenaWords)
                compactAttentionClassLayer(full,uint32_t(reservation));
            layer.reservedWords=layer.usedWords;c=e.command();begin(c);
        }
        layer.reservedWords+=uint32_t(reservation);
        uint32_t coordinates=cfg.rows*2u*MODEL_GQA;
        for(uint32_t operation=0u;operation<3u;operation++){
            bindCompute(c,attentionClassPipelines[operation],layerSets[cfg.layer]);
            vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
            uint32_t groups=operation==1u?(coordinates+127u)/128u:
                cfg.rows*2u*(MODEL_GQA/RT_ATTENTION_HEAD_DIM);
            vkCmdDispatch(c,groups,1u,1u);sync(c);
        }
        bindCompute(c,attentionClassPipelines[3],layerSets[cfg.layer]);
        vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatch(c,cfg.rows*attentionClassDomains,1u,1u);sync(c);
        layer.mirrorDirty=true;
        attentionClassRays+=coordinates;
    }
    void ensureAttentionPipelines(){
        if(attentionReceiptPipelines[0])return;
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext=&subgroup;vkGetPhysicalDeviceProperties2(e.physical,&properties);
        const VkSubgroupFeatureFlags required=VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|
            VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        const auto& limits=properties.properties.limits;
        bool tc=attentionBackend==5u,local=attentionBackend==4u;
        attentionLocalSplit=local&&RT_FULL_COUNT>0&&RT_ATTENTION_HEAD_DIM==256u&&MODEL_Q==4u*MODEL_GQA;
        if(tc&&(!e.cooperativeMatrix2||RT_ATTENTION_HEAD_DIM%16u!=0u))
            throw std::runtime_error("grouped_rt_tc requires cooperative_matrix2 and a head dimension divisible by 16");
        uint32_t sharedBytes=local?(attentionLocalSplit?48u:32u)*RT_ATTENTION_HEAD_DIM+4u*attentionBlock+attentionBlock/8u+512u:tc?(attentionBlock>256u?152u:168u)*RT_ATTENTION_HEAD_DIM+16u*attentionBlock+attentionBlock/8u+512u:
            128u*RT_ATTENTION_HEAD_DIM+((attentionBackend==2u||attentionBackend==4u)?20u:4u)*attentionBlock+attentionBlock/8u+512u;
        if(local)sharedBytes+=4u*(attentionBlock+1u); // RT's compact per-group gap queue.
        uint32_t producerWidth=local?256u:512u;
        if(subgroup.subgroupSize!=32u||(subgroup.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)==0u||
           (subgroup.supportedOperations&required)!=required||limits.maxComputeWorkGroupInvocations<producerWidth||
           limits.maxComputeWorkGroupSize[0]<producerWidth||limits.maxComputeSharedMemorySize<sharedBytes)
            throw std::runtime_error("Grouped attention requires 32-lane compute subgroups with basic, vote, ballot, arithmetic and shuffle operations and sufficient workgroup/shared memory limits");
        for(uint32_t slot=0u;slot<(attentionLocalSplit?5u:3u);slot++){
            const char* module=tc?"resident_attention_receipt_tc":local?"resident_attention_local":attentionBackend==2u?"resident_attention_receipt":"resident_attention_dense";
            uint32_t operation=slot,width=slot==0u?producerWidth:128u;
            if(attentionLocalSplit){
                if(slot==0u){module="resident_attention_local_key";width=128u;}
                else if(slot==1u)module="resident_attention_local_value";
                else if(slot==3u){module="resident_attention_local_trace";width=32u;}
                else if(slot==4u){module="resident_attention_local_key_fallback";operation=0u;width=256u;}
            }
            attentionReceiptPipelines[slot]=e.pipeline(module,layout,operation,attentionBlock,0xffffffffu,wordTile,width,featureTile,0xffffffffu,local?uint32_t(attentionVMemberDistribution):0xffffffffu,0xffffffffu,0xffffffffu,tc?1u:0xffffffffu);
        }
    }
    void prepareAttention(uint32_t rows,uint32_t position,uint32_t origin=0u){
        ensureAttentionPipelines();
        AttentionReceiptLayout receipt{attentionBlock,(attentionBackend==2u||attentionBackend==5u),attentionBackend==4u};
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
        cfg.stage=operation==4u?0u:operation;cfg.padding0=2u;std::memcpy(&cfg.padding1,&routeAperture,4u);
        bindCompute(c,attentionReceiptPipelines[operation],layerSets[cfg.layer]);
        vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatch(c,x,y,1u);sync(c);
    }
    void dispatchAttentionProducer(VkCommandBuffer c,ResidentConfig cfg,uint32_t groups){
        if(!attentionLocalSplit){receiptDispatch(c,cfg,0u,groups,cfg.rows);return;}
        uint32_t finalPosition=cfg.mode==0u?cfg.rows-1u:cfg.mode==1u?cfg.position:cfg.position+cfg.rows-1u;
        uint32_t available=finalPosition<cfg.checkpoint?0u:std::min(finalPosition-cfg.checkpoint+1u,capacity);
        // Advancing row pairs cannot share a last-row window once it fills the
        // ring. Single-row decode also avoids a separate trace dispatch.
        bool shareRows=cfg.rows>1u&&(cfg.mode==1u||available<capacity);
        if(!shareRows){receiptDispatch(c,cfg,4u,groups,cfg.rows);return;}
        receiptDispatch(c,cfg,3u,groups,cfg.rows);
        uint32_t kvHeads=(heads[cfg.layer]+3u)/4u;
        receiptDispatch(c,cfg,0u,groups*kvHeads,(cfg.rows+1u)/2u);
    }
    void traceAttentionReceipts(VkCommandBuffer c,ResidentConfig cfg){
        cfg.index=0u;cfg.checkpoint=0u;
        receiptDispatch(c,cfg,2u,(cfg.rows+127u)/128u);
        uint32_t finalPosition=cfg.mode==0u?cfg.rows-1u:cfg.mode==1u?cfg.position:cfg.position+cfg.rows-1u;
        uint32_t groups=AttentionReceiptLayout{attentionBlock}.groups(finalPosition,capacity);
        uint32_t stamp=startRT(c,0u);
        dispatchAttentionProducer(c,cfg,groups);endRT(c,stamp);attentionProducerDispatches++;
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
        if(attentionBackend!=4u&&AttentionReceiptLayout{attentionBlock}.groups(finalPosition,capacity)==1u)return;
        cfg.index=0u;cfg.checkpoint=0u;
        receiptDispatch(c,cfg,1u,attentionBackend==4u?MODEL_GQA/RT_ATTENTION_HEAD_DIM:RT_ATTENTION_HEADS,cfg.rows);attentionMergeDispatches++;
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
        attentionDiagnosticDirty=true;resetAttentionClassPending();attentionClassRestoreCopies.clear();
        uint32_t firstSlot=historyStart%capacity;
        uint32_t classExtent=count>capacity-firstSlot?capacity:firstSlot+count;
        resizeAttentionClasses(classExtent,false);if(attentionBackend==4u)prepareStaticAttentionClassScene();
        uint32_t initial[8]{0u,0u,1u,0u,0xffffffffu,0xffffffffu,0u,0u};e.upload(control,initial,32u);setup();
        std::vector<uint32_t> marks(size_t(std::max(1u,MODEL_FULL))*capacity*2u,0u);
        for(uint32_t slot=0;slot<std::max(1u,MODEL_FULL)*capacity;slot++)marks[slot*2u]=0xffffffffu;
        uint32_t historyBase=fullIndex(layer)*capacity;
        for(uint32_t i=0;i<count;i++)marks[(historyBase+(historyStart+i)%capacity)*2u]=historyStart+i;
        e.upload(slots,marks.data(),marks.size()*4u);
        e.ensureTimestamps(7u);double classMs=0.0;
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
            if(attentionBackend==4u){
                vkCmdResetQueryPool(c,e.timestamps,5u,2u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,5u);
                classifyAttentionHistory(c,cfg);
                vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,6u);
            }
            check(vkEndCommandBuffer(c),"end attention diagnostic history packing");e.submit(c);
            if(attentionBackend==4u){uint64_t stamps[2];check(vkGetQueryPoolResults(e.device,e.timestamps,5u,2u,sizeof(stamps),stamps,8u,VK_QUERY_RESULT_64_BIT),"read class diagnostic timestamps");double classCost=double(stamps[1]-stamps[0])*e.period/1e6;classMs+=classCost;attentionClassGPU+=classCost;}
            vkFreeCommandBuffers(e.device,e.pool,1u,&c);
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
        prepareAttention(rows,queryStart,historyStart);AttentionReceiptLayout receipt{attentionBlock,(attentionBackend==2u||attentionBackend==5u),attentionBackend==4u};
        uint32_t records=uint32_t(receipt.records(rows,queryStart,capacity,historyStart)),stride=receipt.stride();
        size_t outputBytes=size_t(rows)*MODEL_Q*4u,physicalBytes=size_t(rows)*RT_ATTENTION_HEADS*8u;
        size_t packetBytes=size_t(count)*HISTORY_BYTES,receiptBytes=size_t(records)*stride*4u;
        size_t classBytes=attentionBackend==4u?size_t(count)*2u*MODEL_GQA*4u:0u;
        size_t total=2u*(outputBytes+physicalBytes)+packetBytes+receiptBytes+classBytes;
        ensureReadback(total+4u);auto c=e.command();begin(c);recordRT=false;
        // Receipt diagnostics do not enable the optional control statistics tail.
        ResidentConfig cfg{0u,rows,layer,queryStart,0u,2u,1u,capacity,depth,2u|131072u,2u,0u,historyStart};
        vkCmdResetQueryPool(c,e.timestamps,0u,5u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0u);
        receiptDispatch(c,cfg,2u,(rows+127u)/128u);
        uint32_t groups=receipt.groups(queryStart+rows-1u,capacity,historyStart);
        dispatchAttentionProducer(c,cfg,groups);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
        receiptDispatch(c,cfg,1u,attentionBackend==4u?MODEL_GQA/RT_ATTENTION_HEAD_DIM:RT_ATTENTION_HEADS,rows);
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
        offset+=packetBytes;copy(c,attentionReceipts,receiptBytes,offset);offset+=receiptBytes;
        if(classBytes)for(uint32_t first=0u;first<count;first+=workRows){
            uint32_t batch=std::min(workRows,count-first);ResidentConfig classCfg=cfg;
            classCfg.rows=batch;classCfg.position=historyStart+first;classCfg.mode=2u;
            bindCompute(c,attentionClassPipelines[4],layerSets[layer]);
            vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(classCfg),&classCfg);
            vkCmdDispatch(c,batch*attentionClassDomains,1u,1u);sync(c);
            std::vector<VkBufferCopy> regions(size_t(batch)*attentionClassDomains);
            for(uint32_t row=0u;row<batch;row++)for(uint32_t domain=0u;domain<attentionClassDomains;domain++){
                size_t index=size_t(row)*attentionClassDomains+domain;
                regions[index]={index*attentionClassTempDomainWords*4u,
                    offset+((size_t(first+row)*attentionClassDomains+domain)*RT_ATTENTION_HEAD_DIM)*4u,
                    RT_ATTENTION_HEAD_DIM*4u};
            }
            Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyBuffer(c,attentionClassScratch->handle,readback->handle,uint32_t(regions.size()),regions.data());sync(c);
        }
        offset+=classBytes;copy(c,control,4u,offset,12u);
        hostBarrier(c);check(vkEndCommandBuffer(c),"end full-attention diagnostic");auto timing=e.measure(c,4u);
        uint64_t times[5];check(vkGetQueryPoolResults(e.device,e.timestamps,0u,5u,sizeof(times),times,8u,VK_QUERY_RESULT_64_BIT),"read full-attention diagnostic timestamps");
        vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        uint32_t error=*reinterpret_cast<uint32_t*>(static_cast<char*>(readback->map)+total);
        if(error)throw std::runtime_error("Full-attention receipt error "+std::to_string(error));
        std::vector<char> output;append(output,std::vector<uint32_t>{attentionBackend==4u?0x41544431u:0x41545231u,rows,count,RT_ATTENTION_HEADS,MODEL_GQA/RT_ATTENTION_HEAD_DIM,RT_ATTENTION_HEAD_DIM,HISTORY_BYTES,attentionBlock,stride,records,uint32_t(legacy),error});
        append(output,std::vector<double>{double(times[1]-times[0])*e.period/1e6,double(times[2]-times[1])*e.period/1e6,legacy?double(times[4]-times[3])*e.period/1e6:0.0,timing.wall});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+total);if(classBytes)append(output,std::vector<double>{classMs});reply(output);
    }
    static constexpr uint32_t attentionStatCount=16u;
    uint32_t attentionStatWords()const{return attentionStatsEnabled?2u*attentionStatCount*std::max(1u,MODEL_FULL):0u;}
    void attentionStatistics(uint32_t reset){
        if(!attentionStatsEnabled||(attentionBackend!=2u&&attentionBackend!=4u&&attentionBackend!=5u)||reset>1u)throw std::runtime_error("Attention statistics require grouped_rt and RT_ATTENTION_STATS=1");
        size_t bytes=size_t(attentionStatWords())*4u,offset=size_t(11u+capacity+65536u)*4u;
        ensureReadback(bytes);auto c=e.command();begin(c);copy(c,control,bytes,0u,offset);
        if(reset){vkCmdFillBuffer(c,control->handle,offset,bytes,0u);sync(c);}
        hostBarrier(c);check(vkEndCommandBuffer(c),"end attention statistics");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
        std::vector<char> output;append(output,std::vector<uint32_t>{0x41545331u,attentionStatCount,MODEL_FULL,attentionBlock});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);reply(output);
    }
