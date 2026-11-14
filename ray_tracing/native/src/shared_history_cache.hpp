#pragma once

// Shared immutable history blocks and compact detector-state markers.
// Input identity is chained content; decode identity remains OpticalPathKey.
class SharedHistoryCache {
public:
    static constexpr uint32_t limit=64u,blockTokens=16u;
    using Key=std::array<uint32_t,8>;
    struct Block {
        struct AttentionLayer {
            Buffer* payload{};
            // One [relative payload word offset, representative count] per
            // token/K-or-V/KV-head, in the existing compact directory order.
            std::vector<std::array<uint32_t,2>> directory;
        };
        Engine& e;uint32_t first,count;bool ready=false;std::unique_ptr<ShapeMesh> scene;Buffer *packets{},*marks{};
        bool attentionReady=false;uint32_t attentionDomains{},attentionHeadDim{},attentionKVWidth{},attentionPacketBytes{};
        std::vector<uint32_t> attentionFullLayers;
        std::vector<AttentionLayer> attentionLayers;
        Block(Engine& engine,uint32_t begin,uint32_t n):e(engine),first(begin),count(n) {
            scene=std::make_unique<ShapeMesh>(e,std::max(1u,MODEL_FULL),96u,1u);
            auto usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            packets=e.buffer(size_t(std::max(1u,MODEL_FULL))*n*HISTORY_BYTES,usage);
            marks=e.buffer(size_t(std::max(1u,MODEL_FULL))*n*8u,usage);
        }
        uint64_t attentionBytes()const{uint64_t bytes=0u;for(const auto& layer:attentionLayers)if(layer.payload)bytes+=layer.payload->allocationSize;return bytes;}
        uint64_t bytes()const{return packets->size+marks->size+scene->sceneBytes()+attentionBytes();}
        ~Block(){for(auto& layer:attentionLayers)e.release(layer.payload);e.release(packets);e.release(marks);}
    };
    struct Entry {
        Engine& e;uint32_t slot,kind;uint64_t touched;Key key;
        std::vector<uint32_t> input;OpticalStateMarker snapshot;OpticalPathKey::Address address{};std::array<uint32_t,2> stops{};bool addressReady=false;std::vector<std::shared_ptr<Block>> blocks;
        Entry(Engine& engine):e(engine){}
        ~Entry(){for(auto& item:snapshot.items)e.release(item.saved);}
    };
    Engine& e;ShapeMesh directory;Buffer *nodes{},*queries{},*results{},*readback{};
    VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkDescriptorPool pool{};VkDescriptorSet set{};VkPipeline pipeline{};
    std::array<std::shared_ptr<Entry>,limit> entries{};
    std::map<Key,std::weak_ptr<Block>> blocks;
    uint64_t clock{},evictions{},budget=512ull*1024u*1024u,lookupRays{};double lookupMs{};uint32_t entryLimit=limit;
    SharedHistoryCache(Engine& engine):e(engine),directory(e,limit/16u,96u,1u) {
        if(auto* value=std::getenv("RT_HISTORY_CACHE_MIB"))budget=std::max(64ull,std::stoull(value))*1024ull*1024ull;
        if(auto* value=std::getenv("RT_HISTORY_CACHE_ENTRIES"))entryLimit=std::max(1u,std::min(limit,uint32_t(std::stoul(value))));
        auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        nodes=e.buffer(limit*48u,usage);queries=e.buffer(limit*48u,usage);results=e.buffer(limit*4u,usage);
        readback=e.buffer(limit*4u,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        VkDescriptorSetLayoutBinding bindings[4]{};
        for(uint32_t i=0;i<4u;i++)bindings[i]={i,i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1u,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=4;sl.pBindings=bindings;
        check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"shared history layout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,4};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;
        check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"shared history pipeline layout");
        VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1u},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3u}};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=2;dp.pPoolSizes=sizes;
        check(vkCreateDescriptorPool(e.device,&dp,nullptr,&pool),"shared history pool");
        VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};a.descriptorPool=pool;a.descriptorSetCount=1;a.pSetLayouts=&setLayout;
        check(vkAllocateDescriptorSets(e.device,&a,&set),"shared history set");
        VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};as.accelerationStructureCount=1;as.pAccelerationStructures=directory.tops.data();
        Buffer* buffers[3]{nodes,queries,results};VkDescriptorBufferInfo infos[3]{};VkWriteDescriptorSet writes[4]{};
        for(uint32_t i=0;i<4;i++){auto& w=writes[i];w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=set;w.dstBinding=i;w.descriptorCount=1;w.descriptorType=bindings[i].descriptorType;
            if(!i)w.pNext=&as;else{infos[i-1]={buffers[i-1]->handle,0,buffers[i-1]->size};w.pBufferInfo=&infos[i-1];}}
        vkUpdateDescriptorSets(e.device,4,writes,0,nullptr);pipeline=e.pipeline("history_cache",layout);publish();
    }
    static std::vector<Key> identities(const std::vector<uint32_t>& input,uint32_t epoch) {
        Key key{};for(uint32_t k=0;k<8;k++)key[k]=0x9e3779b9u*(k+1u)^epoch;
        std::vector<Key> keys(input.size()+1u);keys[0]=key;
        for(size_t i=0;i<input.size();i++){for(uint32_t k=0;k<8;k++){uint32_t x=key[k]^input[i]^uint32_t(i)*0x85ebca6bu;
            x^=x>>16u;x*=0x7feb352du+k*2u;x^=x>>15u;key[k]=(x<<7u)|(x>>25u);}keys[i+1]=key;}return keys;
    }
    static std::array<uint32_t,12> node(const Key& key,uint32_t count,bool valid) {
        std::array<uint32_t,12> n{};std::copy(key.begin(),key.end(),n.begin());
        for(uint32_t k=0;k<2;k++){float chart=float(key[k]&0xffffffu)/float(0x1000000u);std::memcpy(&n[8+k],&chart,4);}
        n[10]=count;n[11]=valid;return n;
    }
    std::shared_ptr<Entry> find(const std::vector<uint32_t>& input,const std::vector<Key>& keys,uint32_t minimum=0u) {
        auto start=Clock::now();lookupRays=0;lookupMs=0;std::set<uint32_t> lengths;
        for(auto& entry:entries)if(entry&&entry->input.size()>=minimum&&entry->input.size()<=input.size())lengths.insert(uint32_t(entry->input.size()));
        if(lengths.empty())return {};
        std::vector<std::array<uint32_t,12>> q;for(auto length:lengths)q.push_back(node(keys[length],length,true));
        e.upload(queries,q.data(),q.size()*48u);auto c=e.command();e.begin(c);
        vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        uint32_t count=uint32_t(q.size());vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&count);vkCmdDispatch(c,(count+31u)/32u,1,1);
        Engine::barrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy{0,0,count*4u};vkCmdCopyBuffer(c,results->handle,readback->handle,1,&copy);
        Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        check(vkEndCommandBuffer(c),"end shared history lookup");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        std::shared_ptr<Entry> best;auto* hits=static_cast<uint32_t*>(readback->map);
        for(uint32_t i=0;i<count;i++)if(hits[i]<limit){auto entry=entries[hits[i]];
            if(entry&&entry->key==keys[entry->input.size()]&&std::equal(entry->input.begin(),entry->input.end(),input.begin())&&(!best||entry->input.size()>best->input.size()))best=entry;}
        if(best)best->touched=++clock;lookupRays=count;lookupMs=ms(start);return best;
    }
    uint64_t snapshotBytes()const {uint64_t n=0;for(auto& entry:entries)if(entry)n+=entry->snapshot.bytes;return n;}
    uint64_t blockBytes()const {uint64_t n=0;for(auto& pair:blocks)if(auto b=pair.second.lock())n+=b->bytes();return n;}
    uint32_t size()const {return uint32_t(std::count_if(entries.begin(),entries.end(),[](auto& x){return bool(x);}));}
    uint32_t maxEntries(uint64_t snapshotSize)const {uint64_t historyBytes=blockBytes();return uint32_t(std::max(1ull,std::min(uint64_t(entryLimit),(budget>historyBytes?budget-historyBytes:0u)/std::max(1ull,snapshotSize))));}
    static uint32_t priority(uint32_t kind){return kind&20u?3u:kind&1u?2u:kind&2u?1u:0u;}
    std::shared_ptr<Entry> reserve(const std::vector<uint32_t>& input,uint32_t count,uint32_t kind,const Key& key,uint64_t snapshotSize) {
        for(auto& x:entries)if(x&&x->key==key&&x->input.size()==count&&std::equal(x->input.begin(),x->input.end(),input.begin())){x->kind|=kind;x->touched=++clock;return {};}
        uint32_t maximum=maxEntries(snapshotSize);
        // Preserve request ends and common roots ahead of optional inner/output
        // boundaries, so a returning branch still attaches its shared history.
        while(size()>=maximum){auto oldest=entries.end();for(auto i=entries.begin();i!=entries.end();i++)if(*i&&(oldest==entries.end()||priority((*i)->kind)<priority((*oldest)->kind)||(priority((*i)->kind)==priority((*oldest)->kind)&&(*i)->touched<(*oldest)->touched)))oldest=i;
            if(priority((*oldest)->kind)>priority(kind))return {};
            oldest->reset();evictions++;}
        uint32_t slot=0;while(entries[slot])slot++;
        auto entry=std::make_shared<Entry>(e);entry->slot=slot;entry->kind=kind;entry->key=key;entry->touched=++clock;
        entry->input.assign(input.begin(),input.begin()+count);entries[slot]=entry;return entry;
    }
    void publish() {
        std::array<std::array<uint32_t,12>,limit> data{};std::vector<float> vertices(limit*18u,0.f);
        const float corners[6][2]{{-1,-1},{-1,1},{1,1},{-1,-1},{1,1},{1,-1}};
        for(uint32_t i=0;i<limit;i++)if(auto& entry=entries[i]){data[i]=node(entry->key,uint32_t(entry->input.size()),true);float xy[2];std::memcpy(xy,data[i].data()+8,8);
            for(uint32_t v=0;v<6;v++)for(uint32_t axis=0;axis<2;axis++)vertices[i*18u+v*3u+axis]=xy[axis]+corners[v][axis]*.00005f;}
        e.upload(nodes,data.data(),sizeof(data));e.upload(directory.vertices,vertices.data(),vertices.size()*4u);
        auto c=e.command();e.begin(c);directory.build(c,directory.count,false);check(vkEndCommandBuffer(c),"end shared boundary publication");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        for(auto it=blocks.begin();it!=blocks.end();)if(it->second.expired())it=blocks.erase(it);else ++it;
    }
    ~SharedHistoryCache(){entries={};blocks.clear();vkDestroyPipeline(e.device,pipeline,nullptr);vkDestroyDescriptorPool(e.device,pool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);e.release(nodes);e.release(queries);e.release(results);e.release(readback);}
};
