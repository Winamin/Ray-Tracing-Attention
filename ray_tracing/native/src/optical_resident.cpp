#include <map>
#include "resident_dimensions.hpp"
#include "reply_cache.hpp"

struct ResidentConfig {uint32_t stage,rows,layer,position,step,mode,index,capacity,depth,debug,padding0,padding1;};

static uint32_t historyInstance(uint32_t layer,uint32_t segment){
return segment*16u;
}
class ResidentHistory:public ShapeMesh {
public:
    uint32_t depth,capacity,chunks;
    struct UpdateBatch {std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pointers;};
    std::vector<UpdateBatch> incremental;
    std::vector<std::array<VkAccelerationStructureBuildGeometryInfoKHR,4>> topInfos;
    std::vector<std::array<VkAccelerationStructureBuildRangeInfoKHR,4>> topRanges;
    std::vector<std::array<const VkAccelerationStructureBuildRangeInfoKHR*,4>> topPointers;
    ResidentHistory(Engine& e,uint32_t d,uint32_t cap):ShapeMesh(e,std::max(1u,MODEL_FULL)*((cap+15)/16),96,std::max(1u,MODEL_FULL)),depth(d),capacity(cap),chunks((cap+15)/16) {
        const float corners[6][2]{{-1,-1},{-1,1},{1,1},{-1,-1},{1,1},{1,-1}};
        std::vector<float> data(size_t(count)*96*3);
        for(uint32_t l=0;l<depth;l++)if(fullLayer(l))for(uint32_t s=0;s<chunks;s++){
            uint32_t i=(fullIndex(l))*chunks+s;
            auto& instance=hostInstances[i];instance.mask=1u;instance.instanceCustomIndex=historyInstance(l,s);instance.transform.matrix[0][3]=4.f*l;
            for(uint32_t t=0;t<16;t++)for(uint32_t v=0;v<6;v++) {size_t at=(size_t(i)*96+t*6+v)*3;data[at]=s*16+t<capacity?.04f*corners[v][0]:0;data[at+1]=s*16+t<capacity?.04f*corners[v][1]:0;data[at+2]=-4.f*(s*16+t);}
        }
        e.upload(vertices,data.data(),vertices->size);e.upload(instances,hostInstances.data(),instances->size);
        auto c=e.command();e.begin(c);build(c,count,true);check(vkEndCommandBuffer(c),"end resident history initialization");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        incremental.resize(depth*chunks);for(uint32_t l=0;l<depth;l++)if(fullLayer(l))for(uint32_t s=0;s<chunks;s++)prepare(incremental[l*chunks+s],l,s,s);
        topInfos.resize(depth);topRanges.resize(depth);topPointers.resize(depth);
        for(uint32_t l=0;l<depth;l++)if(fullLayer(l)){uint32_t top=fullIndex(l);auto& bi=topInfos[l][0];bi={VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;bi.srcAccelerationStructure=bi.dstAccelerationStructure=tops[top];bi.geometryCount=1;bi.pGeometries=&topGeometry[top];bi.scratchData.deviceAddress=(topScratch[top]->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);topRanges[l][0]={chunks,0,0,0};topPointers[l][0]=&topRanges[l][0];}
    }
    void prepare(UpdateBatch& batch,uint32_t layer,uint32_t first,uint32_t last){
        auto& infos=batch.infos;auto& ranges=batch.ranges;infos.reserve(32*(last-first+1));ranges.reserve(32*(last-first+1));
        for(uint32_t group=0;group<1;group++)for(uint32_t b=0;b<1;b++)for(uint32_t segment=first;segment<=last;segment++){
            uint32_t i=(fullIndex(layer))*chunks+segment;auto& leaf=leaves[i];
            VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};bi.type=VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;bi.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;bi.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;bi.srcAccelerationStructure=bi.dstAccelerationStructure=leaf.handle;bi.geometryCount=1;bi.pGeometries=&leaf.geometry;bi.scratchData.deviceAddress=(leaf.scratch->address+e.alignment-1)&~VkDeviceAddress(e.alignment-1);infos.push_back(bi);ranges.push_back({32,0,0,0});
        }
        batch.pointers.reserve(ranges.size());for(auto& r:ranges)batch.pointers.push_back(&r);
    }
    void update(VkCommandBuffer c,uint32_t layer,uint32_t first,uint32_t last){
        UpdateBatch full;UpdateBatch* batch=&incremental[layer*chunks+first/16];if(first/16!=last/16){prepare(full,layer,first/16,last/16);batch=&full;}
        e.buildAS(c,uint32_t(batch->infos.size()),batch->infos.data(),batch->pointers.data());
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        e.buildAS(c,1,topInfos[layer].data(),topPointers[layer].data());
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    void restoreAll(VkCommandBuffer c){
        UpdateBatch batch;
        for(uint32_t l=0;l<depth;l++)if(fullLayer(l)){UpdateBatch part;prepare(part,l,0,chunks-1);
            batch.infos.insert(batch.infos.end(),part.infos.begin(),part.infos.end());batch.ranges.insert(batch.ranges.end(),part.ranges.begin(),part.ranges.end());}
        for(auto& range:batch.ranges)batch.pointers.push_back(&range);
        e.buildAS(c,uint32_t(batch.infos.size()),batch.infos.data(),batch.pointers.data());
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> tops;std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges;
        for(uint32_t l=0;l<depth;l++)if(fullLayer(l)){tops.push_back(topInfos[l][0]);ranges.push_back(topPointers[l][0]);}
        e.buildAS(c,uint32_t(tops.size()),tops.data(),ranges.data());
        Engine::barrier(c,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
};

struct OpticalSnapshot {
    struct Item {Buffer* source;Buffer* saved;VkDeviceSize offset,bytes;};
    std::vector<Item> items;uint64_t bytes{};
    void add(Engine& e,Buffer* source,VkDeviceSize count,VkDeviceSize offset=0){
        auto* saved=e.buffer(count,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);items.push_back({source,saved,offset,count});bytes+=count;
    }
    void transfer(VkCommandBuffer c,bool restore){
        for(auto& item:items){VkBufferCopy region{restore?0:item.offset,restore?item.offset:0,item.bytes};vkCmdCopyBuffer(c,restore?item.saved->handle:item.source->handle,restore?item.source->handle:item.saved->handle,1,&region);}
    }
};

class Resident {
public:
    Engine& e;uint32_t depth{},blocks{},middle{},quantization{},step{},capacity{},absolutePosition{};bool initialized=false;
    std::unique_ptr<ReplyTreeCache> replyTree;std::unique_ptr<ResidentHistory> history;std::unique_ptr<ShapeMesh> lenses,words,scalars,features;
    Buffer *parameters{},*matrices{},*basis{},*propagation{},*keyGeometry{},*state{},*normalized{},*query{},*key{},*decoded{},*geometry{},*slots{},*rays{},*hits{},*paths{},*detector{},*physical{},*wordDecoded{},*wordResult{},*baseChart{},*feedForward{},*auxiliary{},*prompt{},*control{},*audit{},*readback{},*featureWork{},*featureRays{},*featureDown{},*partialDown{};
    VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkDescriptorPool descriptorPool{};VkDescriptorSet set{},initialSet{},boundSet{};std::vector<VkDescriptorSet> layerSets;VkPipeline math[35]{},ray[14]{},matrixProjection[4][2]{},mlpBatch[2]{},quantDown[2][2]{},quantOutput[2][2]{},boundPipeline{};
    uint32_t wordTile{},wordTiles{},featureTile{};uint64_t wordSceneBytes{};bool sparseRouting{},exactRT{};float routeAperture{};bool cacheEnabled{},prefixReady{};OpticalSnapshot prefixSnapshot,replySnapshot;
    std::vector<uint32_t> cachedPrefix,currentPrefix,currentGenerated,checkpointRoot;
    ReplyTreeCache::Key pathKey{},checkpointKey{};uint32_t stateCursor{},checkpointPosition{},cacheEpoch{};bool checkpointReady=false;std::array<uint32_t,2> stops{};
    std::vector<VkDescriptorBufferInfo> layerMatrices;std::vector<uint32_t> heads,downFormats;std::vector<std::array<VkPipeline,2>> outputPipelines,gatePipelines;double setupGPU{};uint64_t inboundBytes{},outboundBytes{};
    VkQueryPool rtTimestamps{};bool recordRT=false;std::vector<uint32_t> rtKinds;
    Resident(Engine& engine,const std::filesystem::path& model,const std::filesystem::path& source,bool cache=false,bool sparse=false,float aperture=.08f,uint32_t spatialTile=32,bool forceRT=false,uint32_t featureBlock=8,uint32_t readoutUnroll=1,uint32_t readoutWords=4):e(engine),capacity(e.capacity),wordTile(spatialTile),wordTiles((MODEL_V+spatialTile-1u)/spatialTile),featureTile(featureBlock),sparseRouting(sparse),exactRT(forceRT),routeAperture(forceRT?2.f:aperture),cacheEnabled(cache) {
        std::ifstream inputFile(model,std::ios::binary);char magic[8];uint32_t header[9];inputFile.read(magic,8);inputFile.read(reinterpret_cast<char*>(header),36);
        if(!inputFile||std::memcmp(magic,"RTRES004",8)||header[5]!=4u||header[0]!=MODEL_D||header[4]!=MODEL_V||!header[1]||header[1]!=MODEL_LAYERS||!header[2]||header[2]>header[1]||header[3]%32)throw std::runtime_error("Invalid resident optical student");
        depth=header[1];blocks=header[2];middle=header[3];quantization=header[5];step=header[8];std::vector<uint32_t> p(header[6]),m(header[7]);inputFile.read(reinterpret_cast<char*>(p.data()),p.size()*4);std::ifstream matrixFile(model.parent_path()/"geometry_weights.bin",std::ios::binary);for(size_t offset=0;offset<m.size()*4;){size_t bytes=std::min<size_t>(64*1024*1024,m.size()*4-offset);matrixFile.read(reinterpret_cast<char*>(m.data())+offset,bytes);if(!matrixFile)break;offset+=bytes;}if(!matrixFile)throw std::runtime_error("Truncated geometric weights");if(!inputFile)throw std::runtime_error("Truncated resident parameters");if(p[11]!=1)throw std::runtime_error("Complete equivalent optical translation required");
        uint32_t recurrentIndex=0;for(uint32_t l=0;l<depth;l++){uint32_t base=32u+l*64u;p[base+41u]=fullIndex(l);p[base+42u]=recurrentIndex;if(!fullLayer(l))recurrentIndex++;}
        for(uint32_t i=0;i<MODEL_FULL;i++)p[32u+i*64u+45u]=MODEL_FULL_LAYERS[i];
        
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        parameters=e.buffer(p.size()*4,usage);matrices=e.buffer(m.size()*4,usage);e.upload(parameters,p.data(),parameters->size);e.upload(matrices,m.data(),matrices->size);m.clear();m.shrink_to_fit();for(uint32_t l=0;l<depth;l++){uint32_t base=32+l*64;layerMatrices.push_back({matrices->handle,size_t(p[base+27])*4,size_t(p[base+28])*4});heads.push_back(p[base+23]);downFormats.push_back(p[base+18]);}
        history=std::make_unique<ResidentHistory>(e,depth,capacity);keyGeometry=e.buffer(size_t(std::max(1u,MODEL_FULL))*capacity*HISTORY_BYTES,usage);
        lenses=std::make_unique<ShapeMesh>(e,sparseRouting?(depth+1)*32:MODEL_FULL+1,12,sparseRouting?4:1);uint64_t beforeWord=e.allocated;words=std::make_unique<ShapeMesh>(e,wordTiles*(sparseRouting?4:1),wordTile*6,1);wordSceneBytes=e.allocated-beforeWord;if(sparseRouting||exactRT)features=std::make_unique<ShapeMesh>(e,depth*(sparseRouting?4u:1u)*(middle/featureTile),featureTile*6,depth);scalars=std::make_unique<ShapeMesh>(e,1,65536*3,1);
        if(sparseRouting){for(uint32_t group=0;group<4;group++)for(uint32_t domain=0;domain<=depth;domain++)for(uint32_t b=0;b<8;b++) {uint32_t i=group*(depth+1)*8+domain*8+b;lenses->hostInstances[i].mask=1u<<b;lenses->hostInstances[i].instanceCustomIndex=domain*32+group*8+b;}}else for(uint32_t i=0;i<MODEL_FULL+1;i++){uint32_t domain=i<MODEL_FULL?MODEL_FULL_LAYERS[i]:depth;lenses->hostInstances[i].mask=1u;lenses->hostInstances[i].instanceCustomIndex=domain*32;}
        e.upload(lenses->instances,lenses->hostInstances.data(),lenses->instances->size);
        std::ifstream headFile(source,std::ios::binary);char basisMagic[8];uint32_t shape[2];headFile.read(basisMagic,8);headFile.read(reinterpret_cast<char*>(shape),8);if(!headFile||std::memcmp(basisMagic,quantization==4u?"RTBASQ01":"RTBAS001",8)||shape[0]!=MODEL_V||shape[1]!=MODEL_D)throw std::runtime_error("Complete model vocabulary basis required");size_t basisBytes=std::filesystem::file_size(source)-16u;std::vector<uint32_t> basisHost(basisBytes/4u);headFile.read(reinterpret_cast<char*>(basisHost.data()),basisBytes);if(!headFile)throw std::runtime_error("Truncated original word basis");basis=e.buffer(basisHost.size()*4,usage);e.upload(basis,basisHost.data(),basis->size);basisHost.clear();basisHost.shrink_to_fit();
        uint32_t workRows=std::min(capacity,32u);auto floats=[&](size_t n){return e.buffer(n*4,usage);};propagation=floats(size_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128*128+MODEL_S*9)+size_t(workRows)*MODEL_S*3);state=floats(size_t(workRows)*MODEL_D);normalized=floats(size_t(workRows)*MODEL_D);query=floats(size_t(workRows)*MODEL_A);key=floats(size_t(workRows)*MODEL_A);decoded=floats(1);geometry=floats(1);slots=floats(size_t(std::max(1u,MODEL_FULL))*capacity*2);rays=floats(size_t(workRows)*(sparseRouting?4u:1u)*4u);hits=floats(std::max<size_t>(size_t(workRows)*(capacity+1),256));paths=floats(1);detector=key;physical=floats(size_t(workRows)*std::max(MODEL_STATE_HEADS,uint32_t(RT_ATTENTION_HEADS))*2);wordDecoded=floats(size_t(256)*MODEL_D);wordResult=e.buffer(size_t(8192+3*MODEL_V)*4,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);baseChart=floats(1);feedForward=floats(size_t(workRows)*MODEL_W);auxiliary=floats(size_t(workRows)*MODEL_STATE_HEADS*2);prompt=floats(capacity);control=e.buffer((11+capacity+65536)*4,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);audit=floats(1);featureWork=e.buffer(size_t(workRows*(16+MODEL_FF*(sparseRouting?2:1))+16)*4,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);featureRays=floats(size_t(workRows)*(sparseRouting?4u:1u)*4u);featureDown=floats(sparseRouting?size_t(depth)*MODEL_FF*1024:1);partialDown=floats(sparseRouting?size_t(workRows)*96*MODEL_D:1);
        readback=e.buffer(4*1024*1024,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        if(cacheEnabled){initializeSnapshot(prefixSnapshot,false);initializeSnapshot(replySnapshot,true);}
        VkDescriptorSetLayoutBinding bindings[38]{};for(uint32_t i=0;i<38;i++)bindings[i]={i,((i<5&&i!=2)||i==32)?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,i==6||i==32?depth:i==0||i==1||i==3?4u:1u,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=38;sl.pBindings=bindings;check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"resident set layout");VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(ResidentConfig)};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"resident pipeline layout");VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,(13+depth)*(depth+1)},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(depth+1)*(32+depth)}};VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.maxSets=depth+1;pool.poolSizeCount=2;pool.pPoolSizes=sizes;check(vkCreateDescriptorPool(e.device,&pool,nullptr,&descriptorPool),"resident descriptor pool");std::vector<VkDescriptorSetLayout> layouts(depth+1,setLayout);VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};alloc.descriptorPool=descriptorPool;alloc.descriptorSetCount=depth+1;alloc.pSetLayouts=layouts.data();std::vector<VkDescriptorSet> sets(depth+1);check(vkAllocateDescriptorSets(e.device,&alloc,sets.data()),"resident descriptor sets");layerSets.assign(sets.begin(),sets.begin()+depth);set=sets[0];initialSet=sets[depth];for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);for(uint32_t i=0;i<35;i++)if(i!=2&&i!=4&&i!=5&&i!=10)math[i]=e.pipeline("resident_math",layout,i,1u,0xffffffffu,wordTile,i==16u?readoutWords*32u:128u,featureTile,readoutUnroll,readoutWords,quantization==4u?(i==16u?p[15]:0xffffffffu):0xffffffffu);for(uint32_t i=0;i<14;i++)if(i<4||i>7||i==5)ray[i]=e.pipeline("resident_ray",layout,i,32u,0xffffffffu,wordTile,i==0||i==12||i==13?32u:128u,featureTile);
        for(uint32_t kind=2;kind<=5;kind++)for(uint32_t batch=0;batch<2;batch++)matrixProjection[kind-2][batch]=e.pipeline("resident_math",layout,2u,batch?4u:1u,kind,wordTile,128u,featureTile);
        for(uint32_t kind=0;kind<2;kind++)mlpBatch[kind]=e.pipeline("resident_math",layout,kind==0u?28u:33u,4u,0xffffffffu,wordTile,128u,featureTile,1u,4u,0xffffffffu);
        if(quantization==4u)for(uint32_t kind=0;kind<2;kind++)for(uint32_t batch=0;batch<2;batch++)quantDown[kind][batch]=e.pipeline("resident_math",layout,33u,batch?4u:1u,0xffffffffu,wordTile,128u,featureTile,1u,4u,kind?14u:12u);
        if(quantization==4u){std::map<std::pair<uint32_t,uint32_t>,std::array<VkPipeline,2>> cached;
            for(uint32_t l=0;l<depth;l++){uint32_t base=32u+l*64u;std::array<VkPipeline,2> out{},gate{};
                for(uint32_t stage: {2u,28u}){uint32_t format=stage==2u?p[base+14u]:p[base+16u]==p[base+20u]?p[base+16u]:0xffffffffu;auto id=std::make_pair(stage,format);
                    if(!cached.count(id)){std::array<VkPipeline,2> pipes{};for(uint32_t batch=0;batch<2;batch++)pipes[batch]=e.pipeline("resident_math",layout,stage,batch?4u:1u,stage==2u?2u:0xffffffffu,wordTile,128u,featureTile,1u,4u,format);cached[id]=pipes;}
                    if(stage==2u)out=cached[id];else gate=cached[id];}outputPipelines.push_back(out);gatePipelines.push_back(gate);}}

        if(cacheEnabled)replyTree=std::make_unique<ReplyTreeCache>(e,control,wordResult);
        VkQueryPoolCreateInfo rtPool{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};rtPool.queryType=VK_QUERY_TYPE_TIMESTAMP;rtPool.queryCount=RT_TIMESTAMP_COUNT;
        check(vkCreateQueryPool(e.device,&rtPool,nullptr,&rtTimestamps),"create RT telemetry timestamps");
    }
    void initializeSnapshot(OpticalSnapshot& snapshot,bool reply){
        snapshot.add(e,state,MODEL_D*4);
        if(MODEL_RECURRENT)snapshot.add(e,propagation,size_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128*128+MODEL_S*9)*4);
        (void)reply;
    }
    void growTree(uint32_t next){
        if(next<=capacity||next>HISTORY_SLOT_MASK+1u||capacity<32u)throw std::runtime_error("Invalid tree growth request");
        uint32_t previous=capacity;auto start=Clock::now();
        
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto enlarged=std::make_unique<ResidentHistory>(e,depth,next);
        auto* newPaths=e.buffer(size_t(MODEL_FULL)*next*HISTORY_BYTES,usage);
        auto* newSlots=e.buffer(size_t(MODEL_FULL)*next*8u,usage);
        auto* newHits=e.buffer(size_t(32u)*(next+1u)*4u,usage);
        auto* newPrompt=e.buffer(size_t(next)*4u,usage);
        auto* newControl=e.buffer(size_t(11u+next+65536u)*4u,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        auto c=e.command();begin(c);vkCmdFillBuffer(c,newSlots->handle,0,newSlots->size,0u);sync(c);
        for(uint32_t layer=0;layer<MODEL_FULL;layer++){
            VkBufferCopy path{size_t(layer)*previous*HISTORY_BYTES,size_t(layer)*next*HISTORY_BYTES,size_t(previous)*HISTORY_BYTES};vkCmdCopyBuffer(c,keyGeometry->handle,newPaths->handle,1,&path);
            VkBufferCopy marks{size_t(layer)*previous*8u,size_t(layer)*next*8u,size_t(previous)*8u};vkCmdCopyBuffer(c,slots->handle,newSlots->handle,1,&marks);
            VkBufferCopy vertices{size_t(layer)*history->chunks*96u*12u,size_t(layer)*enlarged->chunks*96u*12u,size_t(history->chunks)*96u*12u};vkCmdCopyBuffer(c,history->vertices->handle,enlarged->vertices->handle,1,&vertices);
        }
        VkBufferCopy headers{0,0,32u};vkCmdCopyBuffer(c,control->handle,newControl->handle,1,&headers);
        VkBufferCopy scalars{size_t(11u+previous)*4u,size_t(11u+next)*4u,65536u*4u};vkCmdCopyBuffer(c,control->handle,newControl->handle,1,&scalars);
        sync(c);enlarged->restoreAll(c);check(vkEndCommandBuffer(c),"end GPU history tree growth");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        
        history=std::move(enlarged);e.release(keyGeometry);keyGeometry=newPaths;e.release(slots);slots=newSlots;e.release(hits);hits=newHits;e.release(prompt);prompt=newPrompt;e.release(control);control=newControl;
        capacity=e.capacity=next;
        if(cacheEnabled)replyTree->rebind(control,wordResult);
        for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);e.release(e.uploadStaging);
        std::vector<char> output;append(output,std::vector<double>{double(previous),double(next),ms(start)});reply(output);
    }
    void restoreSnapshot(VkCommandBuffer c,OpticalSnapshot& snapshot){
        snapshot.transfer(c,true);sync(c);
    }
    void bind(VkDescriptorSet target,bool initial,uint32_t layer=0) {
        VkAccelerationStructureKHR handles[5][4]{};for(uint32_t i=0;i<4;i++){handles[0][i]=history->tops[fullIndex(layer)];handles[1][i]=lenses->tops[sparseRouting?i:0u];handles[3][i]=words->tops[0];}handles[2][0]=scalars->tops[0];handles[4][0]=scalars->tops[0];VkAccelerationStructureKHR featureHandles[RT_LAYER_COUNT]{};for(uint32_t l=0;l<depth;l++)featureHandles[l]=features?features->tops[l]:scalars->tops[0];VkWriteDescriptorSetAccelerationStructureKHR as[36]{};
        Buffer* buffers[38]{nullptr,nullptr,propagation,nullptr,nullptr,parameters,matrices,basis,state,normalized,query,key,keyGeometry,initial?words->vertices:history->vertices,decoded,geometry,slots,rays,hits,paths,detector,physical,wordDecoded,scalars->vertices,wordResult,lenses->vertices,baseChart,feedForward,auxiliary,prompt,control,audit,nullptr,features?features->vertices:scalars->vertices,featureWork,featureRays,featureDown,partialDown};VkDescriptorBufferInfo infos[38]{};VkWriteDescriptorSet writes[38]{};
        for(uint32_t i=0;i<38;i++){auto& w=writes[i];w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=target;w.dstBinding=i;w.descriptorCount=i==6||i==32?depth:i==0||i==1||i==3?4:1;w.descriptorType=((i<5&&i!=2)||i==32)?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;if((i<5&&i!=2)||i==32){as[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};as[i].accelerationStructureCount=w.descriptorCount;as[i].pAccelerationStructures=i==32?featureHandles:handles[i];w.pNext=&as[i];}else{infos[i]={buffers[i]->handle,0,buffers[i]->size};w.pBufferInfo=i==6?layerMatrices.data():&infos[i];}}vkUpdateDescriptorSets(e.device,38,writes,0,nullptr);
    }
    static void sync(VkCommandBuffer c){Engine::barrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);}
    void begin(VkCommandBuffer c){e.begin(c);boundSet=VK_NULL_HANDLE;boundPipeline=VK_NULL_HANDLE;}
    void bindCompute(VkCommandBuffer c,VkPipeline pipeline,VkDescriptorSet selected){if(boundPipeline!=pipeline){vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);boundPipeline=pipeline;}if(boundSet!=selected){vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&selected,0,nullptr);boundSet=selected;}}
    uint32_t startRT(VkCommandBuffer c,uint32_t kind){
        if(!recordRT)return ~0u;uint32_t at=uint32_t(rtKinds.size());if(at>=RT_QUERY_PAIRS)throw std::runtime_error("RT telemetry query capacity exceeded");
        rtKinds.push_back(kind);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,at*2u);return at;
    }
    void endRT(VkCommandBuffer c,uint32_t at){if(at!=~0u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,at*2u+1u);}
    void launch(VkCommandBuffer c,VkPipeline* pipelines,ResidentConfig cfg,uint32_t x,uint32_t y=1,uint32_t z=1,VkDescriptorSet target=VK_NULL_HANDLE) {
        cfg.padding0=sparseRouting?1u:exactRT?2u:0u;std::memcpy(&cfg.padding1,&routeAperture,4);bool projection=pipelines==math&&cfg.stage==2u,routeBatch=pipelines==math&&(cfg.stage==28u||cfg.stage==33u)&&cfg.rows>1u&&!sparseRouting;bool batch=(projection&&cfg.rows>1u)||routeBatch;VkPipeline selected=routeBatch?mlpBatch[cfg.stage==28u?0u:1u]:projection?matrixProjection[cfg.index-2u][batch?1u:0u]:pipelines[cfg.stage];if(quantization==4u&&pipelines==math&&cfg.stage==33u&&(downFormats[cfg.layer]==12u||downFormats[cfg.layer]==14u))selected=quantDown[downFormats[cfg.layer]==14u?1u:0u][batch?1u:0u];if(quantization==4u&&projection&&cfg.index==2u)selected=outputPipelines[cfg.layer][batch?1u:0u];if(quantization==4u&&pipelines==math&&cfg.stage==28u)selected=gatePipelines[cfg.layer][batch?1u:0u];bindCompute(c,selected,target?target:layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        uint32_t query=~0u;if(pipelines==ray&&(cfg.stage==0u||cfg.stage==12u||cfg.stage==13u))query=startRT(c,cfg.stage==0u?0u:cfg.stage==12u?1u:2u);
        if(pipelines==ray&&cfg.stage==0u&&x>65535u){y=(x+65534u)/65535u;x=65535u;}
        vkCmdDispatch(c,x,batch?(y+3u)/4u:y,z);sync(c);endRT(c,query);
    }
    void payload(VkCommandBuffer c,ResidentConfig cfg){cfg.stage=8;launch(c,ray,cfg,1);}
    void mark(VkCommandBuffer c,ResidentConfig cfg,uint32_t at){if(cfg.debug==2)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2+cfg.step*(depth*7+4)+cfg.layer*7+at);}
    void setup(){if(initialized)return;ResidentConfig cfg{7,1,0,0,0,0,0,capacity,depth,0,0,0};auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        launch(c,math,cfg,sparseRouting?(depth+1)*32:MODEL_FULL+1,1,1,initialSet);lenses->build(c,lenses->count,true);cfg.stage=8;launch(c,math,cfg,512,1,1,initialSet);scalars->build(c,1,true);cfg.stage=9;launch(c,math,cfg,32768,(wordTiles*wordTile+32767u)/32768u,sparseRouting?4u:1u,initialSet);words->build(c,words->count,true);if(features){cfg.stage=25;for(uint32_t l=0;l<depth;l++){cfg.layer=l;launch(c,math,cfg,MODEL_FF,sparseRouting?4u:1u,1,initialSet);if(sparseRouting){cfg.stage=31;launch(c,math,cfg,24576);}cfg.stage=25;}features->build(c,features->count,true);}cfg.stage=9;launch(c,ray,cfg,512);if(quantization!=4u){cfg.stage=22;cfg.index=0;for(uint32_t l=0;l<depth;l++){cfg.layer=l;launch(c,math,cfg,32768,(layerMatrices[l].range/4+32768*128-1)/(32768*128));}cfg.index=1;launch(c,math,cfg,32768,(MODEL_V*1024u+32768u*128u-1)/(32768u*128u));}vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);check(vkEndCommandBuffer(c),"end resident first-forward geometry");auto time=e.measure(c,1);setupGPU=time.gpu;vkFreeCommandBuffers(e.device,e.pool,1,&c);
        auto coefficients=e.download<uint32_t>(control,11);if(coefficients[3]&512u)throw std::runtime_error("RT coefficient conversion: reference "+std::to_string(coefficients[7])+" measured "+std::to_string(coefficients[8])+" layer "+std::to_string(coefficients[9])+" basis "+std::to_string(coefficients[10])+" count "+std::to_string(coefficients[6]));initialized=true;if(!sparseRouting){wordSceneBytes-=words->scratchBytes();words->releaseScratch();lenses->releaseScratch();scalars->releaseScratch();if(features)features->releaseScratch();}e.release(e.uploadStaging);e.release(e.downloadStaging);}
    void layer(VkCommandBuffer c,ResidentConfig cfg) {
        mark(c,cfg,0);cfg.stage=1;cfg.index=0;launch(c,math,cfg,cfg.rows);cfg.stage=2;cfg.index=5;launch(c,math,cfg,fullLayer(cfg.layer)?(MODEL_Q+MODEL_GQA)/2u:quantization==4u?(2u*RT_STATE_KEY_HEADS*128u+2u*MODEL_S+3u)/4u:MODEL_D,cfg.rows);
        if(fullLayer(cfg.layer)){
            if(RT_QK_NORM){cfg.stage=14;launch(c,math,cfg,heads[cfg.layer],cfg.rows);}cfg.stage=3;launch(c,math,cfg,(MODEL_Q+255u)/256u,cfg.rows);
            cfg.stage=34;launch(c,math,cfg,1,cfg.rows);
cfg.stage=6;launch(c,math,cfg,1,cfg.rows);
            mark(c,cfg,1);uint32_t first=cfg.mode==0?0:cfg.position%capacity,last=cfg.mode==0?cfg.rows-1:cfg.mode==1?first:(cfg.position+cfg.rows-1)%capacity;history->update(c,cfg.layer,first,last);mark(c,cfg,2);
            vkCmdFillBuffer(c,control->handle,24,4,0);vkCmdFillBuffer(c,hits->handle,0,size_t(cfg.rows)*(capacity+1u)*4u,0);sync(c);cfg.stage=0;launch(c,ray,cfg,(cfg.rows*((capacity+15u)/16u)+7u)/8u);mark(c,cfg,3);payload(c,cfg);mark(c,cfg,4);
            cfg.stage=3;launch(c,ray,cfg,heads[cfg.layer],cfg.rows);mark(c,cfg,5);
        }else{
            cfg.stage=18;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);cfg.stage=19;launch(c,math,cfg,MODEL_S*3u/128u);cfg.stage=20;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);
            mark(c,cfg,1);mark(c,cfg,2);mark(c,cfg,3);mark(c,cfg,4);cfg.stage=21;launch(c,math,cfg,32,MODEL_STATE_HEADS);mark(c,cfg,5);cfg.stage=15;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);
        }
        cfg.stage=2;cfg.index=2;launch(c,math,cfg,MODEL_D/4u,cfg.rows);cfg.stage=1;cfg.index=1;launch(c,math,cfg,cfg.rows);if(features)routedMLP(c,cfg);else{cfg.stage=2;cfg.index=3;launch(c,math,cfg,(middle+3)/4,cfg.rows);cfg.index=4;launch(c,math,cfg,MODEL_D/4u,cfg.rows);}if(cfg.debug==1){cfg.stage=12;launch(c,math,cfg,MODEL_D/128u);}mark(c,cfg,6);
    }
    size_t featureGlobal()const{return size_t(std::min(capacity,32u))*(16u+MODEL_FF*(sparseRouting?2u:1u));}
    void routedMLP(VkCommandBuffer c,ResidentConfig cfg){
        vkCmdFillBuffer(c,featureWork->handle,0,size_t(cfg.rows)*(16u+MODEL_FF*(sparseRouting?2u:1u))*4u,0u);sync(c);
        if(sparseRouting||!exactRT||cfg.index!=1u){cfg.stage=26;launch(c,math,cfg,sparseRouting?4u:1u,cfg.rows);}cfg.stage=13;launch(c,ray,cfg,((middle/featureTile)*(sparseRouting?4u:1u)+31u)/32u,cfg.rows);if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);if(sparseRouting){cfg.stage=27;launch(c,math,cfg,1);
        cfg.stage=28;cfg.padding0=1u;std::memcpy(&cfg.padding1,&routeAperture,4);bindCompute(c,math[28],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatchIndirect(c,featureWork->handle,featureGlobal()*4u);sync(c);if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);cfg.stage=29;bindCompute(c,math[29],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);vkCmdDispatchIndirect(c,featureWork->handle,(featureGlobal()+4u)*4u);sync(c);
        }else{cfg.stage=28;launch(c,math,cfg,(middle+3u)/4u,cfg.rows);if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);cfg.stage=33;launch(c,math,cfg,MODEL_D/4u,cfg.rows);}
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,3u);
        if(sparseRouting){cfg.stage=32;launch(c,math,cfg,8,cfg.rows);}
    }
    void forward(VkCommandBuffer c,ResidentConfig cfg){cfg.stage=0;launch(c,math,cfg,cfg.rows);for(uint32_t l=0;l<depth;l++){cfg.layer=l;layer(c,cfg);}cfg.stage=1;cfg.index=2;launch(c,math,cfg,cfg.rows);}
    void wordMark(VkCommandBuffer c,ResidentConfig cfg,uint32_t at){if(cfg.debug==2u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u+cfg.step*(depth*7u+4u)+depth*7u+at);}
    void wordTrace(VkCommandBuffer c,ResidentConfig cfg){
        vkCmdFillBuffer(c,wordResult->handle,4096u*4u,4u,0u);
        vkCmdFillBuffer(c,wordResult->handle,(8192u+MODEL_V)*4u,MODEL_V*4u,0u);sync(c);
        cfg.stage=11;launch(c,math,cfg,sparseRouting?4:1);cfg.stage=12;launch(c,ray,cfg,((wordTiles+31)/32)*(sparseRouting?4:1));
    }
    void word(VkCommandBuffer c,ResidentConfig cfg){wordMark(c,cfg,0);wordTrace(c,cfg);wordMark(c,cfg,1);
        cfg.stage=24;launch(c,math,cfg,1);cfg.stage=16;cfg.padding0=sparseRouting?1u:exactRT?2u:0u;std::memcpy(&cfg.padding1,&routeAperture,4);bindCompute(c,math[16],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);vkCmdDispatchIndirect(c,wordResult->handle,6144u*4u);sync(c);wordMark(c,cfg,2);
        cfg.stage=17;launch(c,math,cfg,128);cfg.stage=23;launch(c,math,cfg,1);wordMark(c,cfg,3);}
    void copy(VkCommandBuffer c,Buffer* source,size_t bytes,size_t destination,size_t offset=0){VkBufferCopy region{offset,destination,bytes};vkCmdCopyBuffer(c,source->handle,readback->handle,1,&region);}
    void ensureReadback(size_t bytes){if(readback->size>=bytes)return;e.release(readback);readback=e.buffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);}
    void hostBarrier(VkCommandBuffer c){Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);}
    void reply(const std::vector<char>& bytes){uint32_t header[2]{0,uint32_t(bytes.size())};std::cout.write(reinterpret_cast<char*>(header),8);std::cout.write(bytes.data(),bytes.size());std::cout.flush();outboundBytes+=bytes.size();}
    void prefill(uint32_t count,uint32_t debug){
        if(!count||count>capacity)throw std::runtime_error("Invalid resident prefix");
        ensureReadback(debug?size_t(depth+1)*count*MODEL_D*4+1024:4*1024*1024);
        if(debug&&audit->size< size_t(depth)*capacity*MODEL_D*4){e.release(audit);audit=e.buffer(size_t(depth)*capacity*MODEL_D*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);}
        std::vector<uint32_t> ids;input(ids,count);std::cin.read(reinterpret_cast<char*>(stops.data()),8);
        if(!std::cin||*std::max_element(ids.begin(),ids.end())>=MODEL_V)throw std::runtime_error("Invalid prefix word IDs");
        // A matching prefix preserves its immutable historical light packets.
        // Token-address nodes remain valid when the active context changes.
        if(debug||ids!=checkpointRoot)checkpointReady=false;
        if(debug)prefixReady=false;
        auto start=Clock::now();bool cached=cacheEnabled&&prefixReady&&!debug&&cachedPrefix.size()<=ids.size()&&std::equal(cachedPrefix.begin(),cachedPrefix.end(),ids.begin());
        uint32_t reused=cached?uint32_t(cachedPrefix.size()):0,rows=count-reused;uint32_t initial[8]{0,0,1,0,stops[0],stops[1],0,0};
        if(rows)e.upload(prompt,ids.data()+reused,rows*4);e.upload(control,initial,32);inboundBytes+=rows*4+40;setup();e.release(e.uploadStaging);
        auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        if(cached)restoreSnapshot(c,prefixSnapshot);else{vkCmdFillBuffer(c,slots->handle,0,slots->size,0);vkCmdFillBuffer(c,propagation->handle,0,propagation->size,0);sync(c);}
        uint32_t lastRows=0;
        for(uint32_t offset=0;offset<rows;offset+=32u){lastRows=std::min(32u,rows-offset);uint32_t position=reused+offset;
            ResidentConfig cfg{0,lastRows,0,position,0,position?2u:0u,offset,capacity,depth,debug,0,0};forward(c,cfg);
            if(debug){copy(c,state,size_t(lastRows)*MODEL_D*4,32+size_t(offset)*MODEL_D*4);sync(c);}
        }
        if(lastRows>1u){VkBufferCopy last{size_t(lastRows-1u)*MODEL_D*4,0,MODEL_D*4};vkCmdCopyBuffer(c,state->handle,state->handle,1,&last);sync(c);}
        if(cacheEnabled&&!debug&&(!cached||rows)){prefixSnapshot.transfer(c,false);sync(c);}
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);copy(c,control,32,0);size_t bytes=32;
        if(debug){bytes+=size_t(count)*MODEL_D*4;for(uint32_t l=0;l<depth;l++){copy(c,audit,size_t(count)*MODEL_D*4,bytes,size_t(l)*capacity*MODEL_D*4);bytes+=size_t(count)*MODEL_D*4;}}
        
        hostBarrier(c);check(vkEndCommandBuffer(c),"end resident prefill");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);absolutePosition=count;
        auto* stats=static_cast<uint32_t*>(readback->map);if(stats[3])throw std::runtime_error("Resident prefix optical error "+std::to_string(stats[3]));
        currentPrefix=ids;currentGenerated.clear();stateCursor=0;pathKey=ReplyTreeCache::address(ids,stops,cacheEpoch);if(cacheEnabled&&!debug&&(!cached||rows)){cachedPrefix=ids;prefixReady=true;}
        std::vector<char> output;append(output,std::vector<double>{setupGPU,time.gpu,time.wall,ms(start),double(cached),double(reused)});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);reply(output);
        if(debug){e.release(audit);audit=e.buffer(4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);if(readback->size>4*1024*1024){e.release(readback);readback=e.buffer(4*1024*1024,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);}}
    }
    void materializeCachedState(VkCommandBuffer c){
        if(stateCursor==currentGenerated.size())return;
        if(checkpointReady&&checkpointRoot==currentPrefix&&checkpointKey==pathKey&&checkpointPosition==absolutePosition){
            restoreSnapshot(c,replySnapshot);stateCursor=uint32_t(currentGenerated.size());return;
        }
        // Only a cache miss needs the numerical state. A truncated replay can
        // continue along its cached path without materializing intermediate states.
        for(uint32_t i=stateCursor;i<currentGenerated.size();i++){
            uint32_t token=currentGenerated[i];vkCmdUpdateBuffer(c,control->handle,0,4,&token);sync(c);
            ResidentConfig cfg{0,1,0,uint32_t(currentPrefix.size())+i,0,1,0,capacity,depth,0,0,0};forward(c,cfg);
        }
        stateCursor=uint32_t(currentGenerated.size());
    }
    void acceptTokens(const uint32_t* ids,uint32_t count){
        for(uint32_t i=0;i<count;i++){pathKey=ReplyTreeCache::advance(pathKey,ids[i],absolutePosition);currentGenerated.push_back(ids[i]);absolutePosition++;}
    }
    bool generate(uint32_t count,uint32_t debug){
        if(!initialized||!count||count>64)throw std::runtime_error("Resident generation expects 1..64 tokens after prefill");
        uint32_t cachedCount=0,cacheQueries=0,cacheLeaf=0xffffffffu,submissions=0;
        double gpu=0,wall=0,cacheGPU=0,cacheWall=0,cacheRT=0,restoreGPU=0;
        uint32_t restoreTokens=0;bool restoredCheckpoint=false;
        std::array<uint32_t,8> header{};std::array<uint32_t,64> ids{};
        std::array<uint32_t,5> counts{};std::array<double,10> parts{};std::array<double,4> rtParts{};
        bool probe=cacheEnabled&&debug!=1u&&debug!=2u;
        if(probe){
            auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdResetQueryPool(c,rtTimestamps,0,2);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,0);
            replyTree->probe(c,pathKey,absolutePosition,count);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,1);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);
            copy(c,control,32,0);copy(c,wordResult,256,32,528u*4u);copy(c,replyTree->receipt,32,288);hostBarrier(c);
            check(vkEndCommandBuffer(c),"end RT cached path query");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
            auto* result=static_cast<uint32_t*>(readback->map);std::copy_n(result,8,header.begin());std::copy_n(result+8,64,ids.begin());
            cachedCount=result[72];cacheLeaf=result[73];cacheQueries=result[74];
            if(cachedCount>count||header[1]!=cachedCount)throw std::runtime_error("Invalid cached path receipt");
            uint64_t stamps[2];check(vkGetQueryPoolResults(e.device,rtTimestamps,0,2,16,stamps,8,VK_QUERY_RESULT_64_BIT),"read cache RT timing");
            cacheRT=double(stamps[1]-stamps[0])*e.period/1e6;
            cacheGPU=gpu=time.gpu;cacheWall=wall=time.wall;submissions++;
            acceptTokens(ids.data(),cachedCount);
            if(!std::equal(pathKey.begin(),pathKey.end(),result+76))throw std::runtime_error("Cached context address mismatch");
        }
        uint32_t remaining=count-cachedCount;
        if(remaining&&(!probe||header[2])){
            auto c=e.command();begin(c);recordRT=debug==2u||debug==3u;rtKinds.clear();
            if(recordRT)vkCmdResetQueryPool(c,rtTimestamps,0,RT_TIMESTAMP_COUNT);
            vkCmdResetQueryPool(c,e.timestamps,0,debug==2u?2u+remaining*(depth*7u+4u):2u);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
            vkCmdFillBuffer(c,featureWork->handle,(featureGlobal()+8u)*4u,8u,0u);
            vkCmdFillBuffer(c,wordResult->handle,6148u*4u,4u,0u);sync(c);
            restoreTokens=uint32_t(currentGenerated.size())-stateCursor;
            if(restoreTokens){
                restoredCheckpoint=checkpointReady&&checkpointRoot==currentPrefix&&checkpointKey==pathKey&&checkpointPosition==absolutePosition;
                vkCmdResetQueryPool(c,rtTimestamps,RT_RECOVERY_QUERY,2u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,RT_RECOVERY_QUERY);
                bool recorded=recordRT;recordRT=false;materializeCachedState(c);recordRT=recorded;
                vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,RT_RECOVERY_QUERY+1u);
            }
            vkCmdFillBuffer(c,control->handle,4,4,cachedCount);sync(c);
            auto startKey=pathKey;uint32_t firstPosition=absolutePosition;
            for(uint32_t i=0;i<remaining;i++){
                ResidentConfig cfg{0,1,0,firstPosition+i,i,1,0,capacity,depth,debug==2u?2u:0u,0,0};word(c,cfg);forward(c,cfg);
            }
            if(cacheEnabled){replySnapshot.transfer(c,false);sync(c);replyTree->record(c,startKey,firstPosition,remaining,cachedCount);}
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);sync(c);
            copy(c,control,32,0);copy(c,wordResult,256,32,528u*4u);copy(c,featureWork,8,288,(featureGlobal()+8u)*4u);copy(c,wordResult,4,296,6148u*4u);
            if(debug==1u)copy(c,state,MODEL_D*4u,308);hostBarrier(c);
            check(vkEndCommandBuffer(c),"end resident generated path");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
            gpu+=time.gpu;wall+=time.wall;submissions++;
            auto* result=static_cast<uint32_t*>(readback->map);std::copy_n(result,8,header.begin());std::copy_n(result+8,64,ids.begin());std::copy_n(result+72,3,counts.begin());
            if(header[3])throw std::runtime_error("Resident decode optical error "+std::to_string(header[3]));
            if(header[1]<cachedCount||header[1]>count)throw std::runtime_error("Invalid generated path length");
            uint32_t emitted=header[1]-cachedCount;
            if(!features){uint32_t computed=emitted-uint32_t(header[2]==0u&&emitted>0u);counts[0]=counts[1]=computed*depth*middle;}
            acceptTokens(ids.data()+cachedCount,emitted);stateCursor=uint32_t(currentGenerated.size());
            if(cacheEnabled){replyTree->commit(emitted);checkpointReady=true;checkpointRoot=currentPrefix;checkpointKey=pathKey;checkpointPosition=absolutePosition;}
            if(debug==2u){
                std::vector<uint64_t> times(remaining*(depth*7u+4u));check(vkGetQueryPoolResults(e.device,e.timestamps,2,uint32_t(times.size()),times.size()*8,times.data(),8,VK_QUERY_RESULT_64_BIT),"read resident profile");
                for(uint32_t i=0;i<remaining;i++){uint32_t b=i*(depth*7u+4u);for(uint32_t l=0;l<depth;l++)for(uint32_t j=0;j<6;j++)parts[j]+=double(times[b+l*7+j+1]-times[b+l*7+j])*e.period/1e6;
                    parts[6]+=double(times[b+depth*7+3]-times[b+depth*7])*e.period/1e6;for(uint32_t j=0;j<3;j++)parts[7+j]+=double(times[b+depth*7+j+1]-times[b+depth*7+j])*e.period/1e6;}
            }
            if(!rtKinds.empty()){
                std::vector<uint64_t> times(rtKinds.size()*2u);check(vkGetQueryPoolResults(e.device,rtTimestamps,0,uint32_t(times.size()),times.size()*8,times.data(),8,VK_QUERY_RESULT_64_BIT),"read RT telemetry");
                for(uint32_t i=0;i<rtKinds.size();i++)rtParts[rtKinds[i]]+=double(times[i*2u+1u]-times[i*2u])*e.period/1e6;
            }
            if(restoreTokens){uint64_t times[2];check(vkGetQueryPoolResults(e.device,rtTimestamps,RT_RECOVERY_QUERY,2u,16,times,8,VK_QUERY_RESULT_64_BIT),"read cached state recovery");restoreGPU=double(times[1]-times[0])*e.period/1e6;}
        }
        recordRT=false;rtParts[3]+=cacheRT;counts[3]=cachedCount;counts[4]=cacheLeaf;
        bool allCached=cachedCount>0u&&cachedCount==header[1];
        std::vector<char> output;append(output,std::vector<double>{gpu,wall,double(submissions),double(submissions),0,double(allCached),double(absolutePosition),double(header[7]),parts[0],parts[1],parts[2],parts[3],parts[4],parts[5],parts[6],parts[7],parts[8],parts[9]});
        append(output,std::vector<uint32_t>(header.begin(),header.end()));append(output,std::vector<uint32_t>(ids.begin(),ids.end()));append(output,std::vector<uint32_t>(counts.begin(),counts.end()));
        if(debug==1u){auto* stateBytes=static_cast<char*>(readback->map)+308;output.insert(output.end(),stateBytes,stateBytes+MODEL_D*4u);}
        append(output,std::vector<double>(rtParts.begin(),rtParts.end()));
        append(output,std::vector<uint32_t>{cachedCount,cacheQueries,replyTree?replyTree->size:0u,ReplyTreeCache::nodeCapacity});
        append(output,std::vector<double>{cacheGPU,cacheWall});
        append(output,std::vector<double>{restoreGPU,double(restoreTokens),double(restoredCheckpoint)});reply(output);return header[2]!=0u;
    }
    void streamGenerate(uint32_t count,uint32_t debug){if(!count||count>64)throw std::runtime_error("Resident stream expects 1..64 tokens");for(uint32_t i=0;i<count;i++)if(!generate(1u,debug))break;}
    void wordCheck(){
        ensureReadback(64u+MODEL_V*4u+256u*MODEL_D*4u);
        std::vector<float> q;input(q,MODEL_D);uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(state,q.data(),q.size()*4);e.upload(control,initial,32);setup();
        auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2+depth*7+4);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        ResidentConfig cfg{0,1,0,0,0,0,0,capacity,depth,2,0,0};word(c,cfg);cfg.stage=5;launch(c,ray,cfg,MODEL_D/256u,256);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);sync(c);size_t bytes=32;copy(c,control,32,0);
        copy(c,wordResult,16,bytes);bytes+=16;copy(c,wordResult,4,bytes,4096*4);bytes+=4;
        copy(c,wordResult,MODEL_V*4,bytes,(8192+2*MODEL_V)*4);bytes+=MODEL_V*4;
        copy(c,wordDecoded,256*MODEL_D*4,bytes);bytes+=256*MODEL_D*4;hostBarrier(c);check(vkEndCommandBuffer(c),"end resident all-contribution check");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        auto* stats=static_cast<uint32_t*>(readback->map);if(stats[3])throw std::runtime_error("Resident word optical error "+std::to_string(stats[3]));
        uint64_t stamps[4];check(vkGetQueryPoolResults(e.device,e.timestamps,2+depth*7,4,sizeof(stamps),stamps,8,VK_QUERY_RESULT_64_BIT),"read word timings");
        std::vector<char> output;append(output,std::vector<double>{time.gpu,time.wall,double(stamps[1]-stamps[0])*e.period/1e6,double(stamps[2]-stamps[1])*e.period/1e6,double(stamps[3]-stamps[2])*e.period/1e6});
        output.insert(output.end(),static_cast<char*>(readback->map)+32,static_cast<char*>(readback->map)+bytes);reply(output);
    }
    void resources(){uint64_t deviceBytes=0;for(auto& b:e.buffers)if(!b->map)deviceBytes+=b->allocationSize;
        std::vector<char> output;append(output,std::vector<uint64_t>{e.physicalAllocated,e.physicalDeviceAllocated,wordSceneBytes,words->vertices->size,wordTiles,wordTile,featureTile,replyTree?replyTree->sceneBytes:0u,prefixSnapshot.bytes+replySnapshot.bytes,parameters->size+matrices->size+basis->size,featureDown->size,deviceBytes,uint64_t(std::count_if(e.memoryPools.begin(),e.memoryPools.end(),[](const auto& p){return p.memory!=VK_NULL_HANDLE;})),keyGeometry->size,decoded->size,propagation->size,audit->size,partialDown->size,
            e.uploadStaging?e.uploadStaging->size:0u,readback->size,history->sceneBytes(),history->count,uint64_t(history->tops.size()),
            words->scratchBytes()+lenses->scratchBytes()+scalars->scratchBytes()+(features?features->scratchBytes():0u),
            hits->size,featureWork->size,std::min(capacity,32u),uint64_t(detector==key),geometry->size+paths->size+baseChart->size});reply(output);
    }
    void apertureUpdate(){float aperture;std::cin.read(reinterpret_cast<char*>(&aperture),4);if(!std::cin||!(aperture>0&&aperture<=2)||!sparseRouting)throw std::runtime_error("Sparse aperture update requires (0,2]");
        routeAperture=aperture;prefixReady=false;checkpointReady=false;cacheEpoch++;setup();auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        ResidentConfig cfg{30,1,0,0,0,0,0,capacity,depth,0,0,0};launch(c,math,cfg,32768,(MODEL_V*4u+32767u)/32768u,1,initialSet);words->build(c,words->count,false);
        cfg.index=1;launch(c,math,cfg,32768,(depth*4u*MODEL_FF+32767u)/32768u,1,initialSet);features->build(c,features->count,false);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);check(vkEndCommandBuffer(c),"end route aperture update");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        std::vector<char> output;append(output,std::vector<double>{time.gpu,time.wall});reply(output);
    }
    void codecCheck(uint32_t count){if(!count||count%2||count>std::min(capacity,32u)*MODEL_D)throw std::runtime_error("Invalid coordinate codec input");std::vector<float> values;input(values,count);uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(state,values.data(),count*4);e.upload(control,initial,32);setup();auto c=e.command();begin(c);ResidentConfig cfg{13,count,0,0,0,0,0,capacity,depth,0,0,0};launch(c,math,cfg,(count/2+127)/128);copy(c,detector,count*4,0);hostBarrier(c);check(vkEndCommandBuffer(c),"end resident coordinate codec check");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);std::vector<char> output(static_cast<char*>(readback->map),static_cast<char*>(readback->map)+count*4);reply(output);}
    void mlpCheck(uint32_t count,uint32_t layer){if(!count||count>std::min(capacity,32u)||layer>=depth)throw std::runtime_error("Invalid optical MLP input");std::vector<float> values;input(values,size_t(count)*MODEL_D);uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(normalized,values.data(),values.size()*4);e.upload(control,initial,32);setup();auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,5u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0u);vkCmdFillBuffer(c,state->handle,0,size_t(count)*MODEL_D*4,0);sync(c);ResidentConfig cfg{2,count,layer,0,0,0,3,capacity,depth,4,0,0};vkCmdFillBuffer(c,feedForward->handle,0,size_t(count)*middle*4,0);sync(c);if(features)routedMLP(c,cfg);else{vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);launch(c,math,cfg,middle/4,count);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);cfg.index=4;launch(c,math,cfg,MODEL_D/4u,count);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,3u);}vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,4u);copy(c,state,size_t(count)*MODEL_D*4,0);for(uint32_t r=0;r<count;r++)copy(c,feedForward,size_t(middle)*4,size_t(count)*MODEL_D*4+size_t(r)*middle*4,size_t(r)*MODEL_W*4);if(features)for(uint32_t r=0;r<count;r++)copy(c,featureWork,4,size_t(count)*(MODEL_D+middle)*4+r*4,size_t(r)*(16u+middle*(sparseRouting?2u:1u))*4);hostBarrier(c);check(vkEndCommandBuffer(c),"end optical MLP check");e.submit(c);uint64_t times[5];check(vkGetQueryPoolResults(e.device,e.timestamps,0u,5u,sizeof(times),times,8u,VK_QUERY_RESULT_64_BIT),"MLP profile");vkFreeCommandBuffers(e.device,e.pool,1,&c);size_t bytes=size_t(count)*(MODEL_D+middle)*4;if(!features)for(uint32_t r=0;r<count;r++)static_cast<uint32_t*>(readback->map)[bytes/4+r]=middle;bytes+=count*4;std::vector<char> output(static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);if(features&&!sparseRouting)append(output,std::vector<double>{double(times[1]-times[0])*e.period/1e6,double(times[2]-times[1])*e.period/1e6,double(times[3]-times[2])*e.period/1e6});reply(output);}
    ~Resident(){vkDeviceWaitIdle(e.device);if(rtTimestamps)vkDestroyQueryPool(e.device,rtTimestamps,nullptr);for(auto p:math)if(p)vkDestroyPipeline(e.device,p,nullptr);for(auto p:ray)if(p)vkDestroyPipeline(e.device,p,nullptr);for(auto& pair:matrixProjection)for(auto p:pair)vkDestroyPipeline(e.device,p,nullptr);for(auto p:mlpBatch)vkDestroyPipeline(e.device,p,nullptr);vkDestroyDescriptorPool(e.device,descriptorPool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);}
};

int main(int argc,char** argv){
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    try{if(argc<3)throw std::runtime_error("Usage: rt_optical_resident student.bin original_head.bin [capacity]");uint32_t capacity=argc>3?uint32_t(std::stoul(argv[3])):300;auto root=std::filesystem::absolute(argv[0]).parent_path();Engine engine;engine.initialize(root,capacity);bool sparse=argc>5&&std::string(argv[5])=="sparse";float aperture=argc>6?std::stof(argv[6]):.08f;if(!(aperture>0&&aperture<=2))throw std::runtime_error("Route aperture must be in (0,2]");uint32_t wordTile=argc>7?uint32_t(std::stoul(argv[7])):32;if(wordTile<8||wordTile>256||(wordTile&(wordTile-1u)))throw std::runtime_error("Word tile requires a dividing power of two in [8,256]");uint32_t featureTile=argc>8?uint32_t(std::stoul(argv[8])):8;if(featureTile<8||featureTile>256||(featureTile&(featureTile-1u)))throw std::runtime_error("Feature tile requires a power of two in [8,256]");uint32_t readoutUnroll=argc>9?uint32_t(std::stoul(argv[9])):1u,readoutWords=argc>10?uint32_t(std::stoul(argv[10])):4u;if((readoutUnroll!=1u&&readoutUnroll!=2u&&readoutUnroll!=4u&&readoutUnroll!=8u)||(readoutWords!=4u&&readoutWords!=8u))throw std::runtime_error("Readout requires unroll 1/2/4/8 and words per group 4/8");Resident model(engine,argv[1],argv[2],argc>4&&std::stoul(argv[4])!=0,sparse,aperture,wordTile,argc>5&&std::string(argv[5])=="rt_full",featureTile,readoutUnroll,readoutWords);uint32_t hello[5]{0x5254523a,model.depth,model.blocks,model.middle,model.quantization};std::cout.write(reinterpret_cast<char*>(hello),20);std::cout.flush();uint32_t op;
        while(std::cin.read(reinterpret_cast<char*>(&op),4)){if(!op)break;uint32_t count,debug;if(!std::cin.read(reinterpret_cast<char*>(&count),4)||!std::cin.read(reinterpret_cast<char*>(&debug),4))throw std::runtime_error("Incomplete resident request");if(op==4||op==5||op==6){model.prefixReady=false;model.checkpointReady=false;}if(op==1)model.prefill(count,debug);else if(op==2)model.generate(count,debug);else if(op==4)model.wordCheck();else if(op==5)model.codecCheck(count);else if(op==6)model.mlpCheck(count,debug);else if(op==7)model.apertureUpdate();else if(op==8)model.resources();else if(op==9)model.streamGenerate(count,debug);else if(op==10)model.growTree(count);else throw std::runtime_error("Unknown resident operation");}return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
