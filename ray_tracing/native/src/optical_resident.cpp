#include <map>
#include <set>
#include "resident_dimensions.hpp"
#include "reply_cache.hpp"
#include "mlp_geometry.hpp"
#include "attention_receipts.hpp"
#include "attention_class_geometry.hpp"

struct ResidentConfig {uint32_t stage,rows,layer,position,step,mode,index,capacity,depth,debug,padding0,padding1,checkpoint;};

static uint32_t historyInstance(uint32_t layer,uint32_t segment){
return segment*16u;
}
class ResidentHistory:public ShapeMesh {
public:
    uint32_t depth,capacity,chunks;std::vector<bool> migrated;bool rebuildMoved=false;Buffer* originalInstances{};
    ResidentHistory(Engine& e,uint32_t d,uint32_t cap):ShapeMesh(e,std::max(1u,MODEL_FULL)*((cap+15)/16),96,std::max(1u,MODEL_FULL),false,true),depth(d),capacity(cap),chunks((cap+15)/16) {
        const float corners[6][2]{{-1,-1},{-1,1},{1,1},{-1,-1},{1,1},{1,-1}};
        std::vector<float> data(size_t(count)*96*3);
        for(uint32_t l=0;l<depth;l++)if(fullLayer(l))for(uint32_t s=0;s<chunks;s++){
            uint32_t i=(fullIndex(l))*chunks+s;
            auto& instance=hostInstances[i];instance.mask=1u;instance.instanceCustomIndex=historyInstance(l,s);instance.transform.matrix[0][3]=4.f*l;
            for(uint32_t t=0;t<16;t++)for(uint32_t v=0;v<6;v++) {size_t at=(size_t(i)*96+t*6+v)*3;data[at]=s*16+t<capacity?.04f*corners[v][0]:0;data[at+1]=s*16+t<capacity?.04f*corners[v][1]:0;data[at+2]=-4.f*(s*16+t);}
        }
        e.upload(vertices,data.data(),vertices->size);e.upload(instances,hostInstances.data(),instances->size);
        originalInstances=e.buffer(instances->size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        e.upload(originalInstances,hostInstances.data(),instances->size);
        auto c=e.command();e.begin(c);build(c,count,true);check(vkEndCommandBuffer(c),"end resident history initialization");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);
    }
    void update(VkCommandBuffer c,uint32_t layer,uint32_t first,uint32_t last){
        uint32_t group=fullIndex(layer),offset=group*chunks;
        VkBufferCopy mutableChunks{size_t(offset+first/16u)*sizeof(hostInstances[0]),size_t(offset+first/16u)*sizeof(hostInstances[0]),size_t(last/16u-first/16u+1u)*sizeof(hostInstances[0])};
        vkCmdCopyBuffer(c,originalInstances->handle,instances->handle,1,&mutableChunks);
        Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        bool initial=!migrated.empty()&&migrated[group];
        // Rebased geometry uses BUILD for changed chunks and their TLAS. The
        // migrated scene's UPDATE path caused device loss on the native GPU.
        build(c,initial?offset+chunks:offset+last/16u+1u,initial||rebuildMoved,false,initial?offset:offset+first/16u,group);
        if(initial)migrated[group]=false;
    }
    ~ResidentHistory(){e.release(originalInstances);}
};

struct OpticalStateMarker {
    struct Item {Buffer* source;Buffer* saved;VkDeviceSize offset,bytes;};
    std::vector<Item> items;uint64_t bytes{};
    void add(Engine& e,Buffer* source,VkDeviceSize count,VkDeviceSize offset=0){
        auto* saved=e.buffer(count,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);items.push_back({source,saved,offset,count});bytes+=count;
    }
    void transfer(VkCommandBuffer c,bool restore){
        for(auto& item:items){VkBufferCopy region{restore?0:item.offset,restore?item.offset:0,item.bytes};vkCmdCopyBuffer(c,restore?item.saved->handle:item.source->handle,restore?item.source->handle:item.saved->handle,1,&region);}
    }
};

#include "shared_history_cache.hpp"

struct GenerationPacket {
    std::array<double,18> timing{};std::array<uint32_t,8> control{};std::array<uint32_t,64> ids{};
    std::array<uint32_t,5> hits{};std::array<double,4> rt{};std::array<uint32_t,4> cache{};
    std::array<double,2> cacheTime{};std::array<double,3> recovery{};
    uint32_t candidates{},rejected{};uint64_t routeTraceRows{},routeReuseRows{};std::vector<char> state;
    bool profile=false;std::array<double,2> detectorTiming{};
    std::vector<char> bytes()const{
        std::vector<char> out;
        append(out,std::vector<double>(timing.begin(),timing.end()));append(out,std::vector<uint32_t>(control.begin(),control.end()));
        append(out,std::vector<uint32_t>(ids.begin(),ids.end()));append(out,std::vector<uint32_t>(hits.begin(),hits.end()));out.insert(out.end(),state.begin(),state.end());
        append(out,std::vector<double>(rt.begin(),rt.end()));append(out,std::vector<uint32_t>(cache.begin(),cache.end()));
        append(out,std::vector<double>(cacheTime.begin(),cacheTime.end()));append(out,std::vector<double>(recovery.begin(),recovery.end()));
        append(out,std::vector<uint32_t>{candidates,rejected});append(out,std::vector<uint64_t>{routeTraceRows,routeReuseRows});
        if(profile)append(out,std::vector<double>(detectorTiming.begin(),detectorTiming.end()));return out;
    }
};

struct AttentionClassAllocation {uint32_t offset{},words{};};
struct AttentionClassLayerCache {
    Buffer* buffer{};uint32_t arenaWords{},usedWords{},reservedWords{};bool mirrorDirty=false;
    std::vector<uint32_t> directory;
    std::vector<AttentionClassAllocation> allocations,freeBlocks;
};
struct AttentionClassRange {uint32_t first,count;};
struct AttentionClassRestoreCopy {Buffer* source;uint32_t full;VkBufferCopy region;};

class Resident {
public:
    Engine& e;uint32_t depth{},blocks{},middle{},quantization{},step{},capacity{},absolutePosition{},workRows{},matrixBatchRows{};bool initialized=false;
    uint32_t captureRows{};
    uint32_t attentionBackend=0u,attentionBlock=128u;bool rtTransmissionMLP=false,attentionDiagnosticDirty=false,attentionStatsEnabled=false,attentionVMemberDistribution=false,attentionLocalSplit=false;Buffer* attentionReceipts{};VkPipeline attentionReceiptPipelines[5]{};
    bool detectorOrdered=false;static constexpr uint32_t detectorBlock=256u,detectorThreshold=512u;VkPipeline detectorPipelines[3]{};
    Buffer *pathScoreDummy{},*pathScoreWork{};std::vector<Buffer*> pathScoreLayers;
    uint32_t pathScorePages{};uint64_t pathScoreSource=1u;bool pathScoreExhausted=false;
    static constexpr uint32_t pathScoreMagic=0x52545053u,pathScoreSourceLimit=0x3fffffffu;
    std::unique_ptr<AttentionClassGeometry> attentionClassScene;
    Buffer *attentionClassDummy{},*attentionClassScratch{},*attentionClassCounts{};
    std::vector<AttentionClassLayerCache> attentionClassLayers;VkPipeline attentionClassPipelines[5]{};
    uint64_t attentionClassRays{},attentionClassBuilds{};
    std::array<std::vector<AttentionClassRange>,RT_LAYER_COUNT> pendingAttentionClassRanges;
    std::vector<AttentionClassRestoreCopy> attentionClassRestoreCopies;
    uint32_t attentionClassSlots{};
    uint64_t attentionClassPeakSceneBytes{},attentionClassReleases{},attentionSplitSubmissions{};
    double attentionSplitGPU{},attentionSplitWait{},attentionClassGPU{};
    uint64_t attentionSegmentEnd{};uint32_t attentionProfileSlot=~0u;
    std::vector<double> attentionProfileGaps;
    VkQueryPool attentionSubmitTimestamps{};
    uint64_t attentionProducerDispatches{},attentionMergeDispatches{},attentionPlannedRays{},attentionCoveredTerms{},attentionReportedRecords{};
    std::unique_ptr<ReplyTreeCache> replyTree;std::unique_ptr<OpticalPathKey> opticalPath;std::unique_ptr<ResidentHistory> history;std::unique_ptr<ShapeMesh> lenses,scalars,features;std::unique_ptr<VocabularyGeometry> words;std::unique_ptr<MLPTransmissionGeometry> mlpSurfaces;
    Buffer *parameters{},*matrices{},*basis{},*propagation{},*keyGeometry{},*state{},*normalized{},*query{},*key{},*decoded{},*geometry{},*slots{},*rays{},*hits{},*paths{},*detector{},*physical{},*wordDecoded{},*wordResult{},*baseChart{},*feedForward{},*auxiliary{},*prompt{},*control{},*audit{},*readback{},*featureWork{},*featureRays{},*featureDown{},*partialDown{};uint64_t featureTraceRows{},featureReuseRows{};
    VkDescriptorSetLayout setLayout{};VkPipelineLayout layout{};VkDescriptorPool descriptorPool{};VkDescriptorSet set{},initialSet{},boundSet{};std::vector<VkDescriptorSet> layerSets;VkPipeline math[35]{},ray[14]{},matrixProjection[4][2]{},opticalMlp[2][2]{},quantOutput[2][2]{},boundPipeline{};
    uint32_t wordTile{},wordTiles{},featureTile{};static constexpr uint32_t wordBlocks=MODEL_D/256u,wordMaskWords=(wordBlocks+31u)/32u,wordQueue=8192u+MODEL_V*(1u+wordMaskWords);uint64_t wordSceneBytes{};bool sparseRouting{},exactRT{};float routeAperture{};bool cacheEnabled{},prefixReady{},stableReady{};OpticalStateMarker liveMarker;
    std::vector<uint32_t> cachedPrefix,currentGenerated;uint32_t stableTokens{};
    std::unique_ptr<SharedHistoryCache> sharedHistory;std::vector<std::shared_ptr<SharedHistoryCache::Block>> attachedBlocks;
    OpticalPathKey::Address pathAddress{};
    uint32_t prefixTokens{},stateCursor{},cacheEpoch{};
    bool gpuFieldReady=false;OpticalPathKey::Address gpuFieldAddress{};uint32_t gpuFieldPosition{};std::array<uint32_t,2> stops{};
    std::vector<VkDescriptorBufferInfo> layerMatrices;std::vector<uint32_t> heads;std::vector<std::array<VkPipeline,2>> outputPipelines;double setupGPU{};uint64_t inboundBytes{},outboundBytes{};
    VkQueryPool rtTimestamps{};bool recordRT=false;std::vector<uint32_t> rtKinds;VkPipeline matrixPack{},matrixAttention{},fastState{};bool wideCooperative{};
    std::vector<VkPipeline> cooperativeOutputs,cooperativeGates,cooperativeDowns;std::vector<std::array<VkPipeline,4>> cooperativeQKV,decodeQKV;std::vector<std::array<VkPipeline,2>> cooperativeAuxiliary,cooperativeControlPack;std::vector<VkPipeline> ownedCooperative;
    Resident(Engine& engine,const std::filesystem::path& model,const std::filesystem::path& source,bool cache=true,bool sparse=false,float aperture=.08f,uint32_t spatialTile=32,bool forceRT=false,uint32_t featureBlock=8,uint32_t readoutUnroll=1,uint32_t readoutWords=4,uint32_t prefillBatch=512,uint32_t matrixRows=8):e(engine),capacity(e.capacity),workRows(prefillBatch),matrixBatchRows(matrixRows),wordTile(spatialTile),wordTiles((MODEL_V+spatialTile-1u)/spatialTile),featureTile(featureBlock),sparseRouting(sparse),exactRT(forceRT),routeAperture(forceRT?2.f:aperture),cacheEnabled(cache) {
        attentionBackend=attentionBackendCode(std::getenv("RT_ATTENTION_BACKEND"));
        std::string requestedDetector=std::getenv("RT_DETECTOR_MODE")?std::getenv("RT_DETECTOR_MODE"):"ordered";
        if(requestedDetector!="legacy"&&requestedDetector!="ordered")throw std::runtime_error("Detector mode must be legacy or ordered");
        detectorOrdered=MODEL_FULL&&RT_ATTENTION_HEAD_DIM<=256u&&attentionBackend<2u&&requestedDetector=="ordered";
        if(attentionBackend==4u&&RT_ATTENTION_HEAD_DIM!=256u)
            throw std::runtime_error("grouped_rt_local requires a 256-dimensional attention head");
        std::string mlpBackend=std::getenv("RT_MLP_BACKEND")?std::getenv("RT_MLP_BACKEND"):"standard";
        if(mlpBackend!="standard"&&mlpBackend!="rt_transmission")throw std::runtime_error("MLP backend must be standard or rt_transmission");
        rtTransmissionMLP=mlpBackend=="rt_transmission";
        attentionStatsEnabled=std::getenv("RT_ATTENTION_STATS")&&std::string(std::getenv("RT_ATTENTION_STATS"))=="1";
        attentionVMemberDistribution=std::getenv("RT_ATTENTION_V_MEMBERS")&&std::string(std::getenv("RT_ATTENTION_V_MEMBERS"))=="1";
        if(const char* block=std::getenv("RT_ATTENTION_BLOCK"))attentionBlock=uint32_t(std::stoul(block));
        if(attentionBlock!=128u&&attentionBlock!=256u&&attentionBlock!=512u)throw std::runtime_error("Invalid attention history block");
        if(attentionBackend>=2u&&sparse)throw std::runtime_error("Grouped attention requires complete contribution routing");
        std::ifstream inputFile(model,std::ios::binary);char magic[8];uint32_t header[9];inputFile.read(magic,8);inputFile.read(reinterpret_cast<char*>(header),36);
        if(!inputFile||std::memcmp(magic,"RTRES004",8)||header[5]!=4u||header[0]!=MODEL_D||header[4]!=MODEL_V||!header[1]||header[1]!=MODEL_LAYERS||!header[2]||header[2]>header[1]||header[3]%32)throw std::runtime_error("Invalid resident optical student");
        depth=header[1];blocks=header[2];middle=header[3];quantization=header[5];step=header[8];std::vector<uint32_t> p(header[6]),m(header[7]);inputFile.read(reinterpret_cast<char*>(p.data()),p.size()*4);std::ifstream matrixFile(model.parent_path()/"geometry_weights.bin",std::ios::binary);for(size_t offset=0;offset<m.size()*4;){size_t bytes=std::min<size_t>(64*1024*1024,m.size()*4-offset);matrixFile.read(reinterpret_cast<char*>(m.data())+offset,bytes);if(!matrixFile)break;offset+=bytes;}if(!matrixFile)throw std::runtime_error("Truncated geometric weights");if(!inputFile)throw std::runtime_error("Truncated resident parameters");if(p[11]!=1)throw std::runtime_error("Complete equivalent optical translation required");
        uint32_t recurrentIndex=0;for(uint32_t l=0;l<depth;l++){uint32_t base=32u+l*64u;p[base+41u]=fullIndex(l);p[base+42u]=recurrentIndex;if(!fullLayer(l))recurrentIndex++;}
        for(uint32_t i=0;i<MODEL_FULL;i++)p[32u+i*64u+45u]=MODEL_FULL_LAYERS[i];

        wideCooperative=e.cooperativeMatrix2&&RT_ATTENTION_HEAD_DIM%32u==0u&&!(std::getenv("RT_PREFILL_COOPERATIVE")&&std::string(std::getenv("RT_PREFILL_COOPERATIVE"))!="2");
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        parameters=e.buffer(p.size()*4,usage);matrices=e.buffer(m.size()*4,usage);e.upload(parameters,p.data(),parameters->size);e.upload(matrices,m.data(),matrices->size);m.clear();m.shrink_to_fit();for(uint32_t l=0;l<depth;l++){uint32_t base=32+l*64;layerMatrices.push_back({matrices->handle,size_t(p[base+27])*4,size_t(p[base+28])*4});heads.push_back(p[base+23]);}
        history=std::make_unique<ResidentHistory>(e,depth,capacity);keyGeometry=e.buffer(size_t(std::max(1u,MODEL_FULL))*capacity*HISTORY_BYTES,usage);
        lenses=std::make_unique<ShapeMesh>(e,sparseRouting?(depth+1)*32:MODEL_FULL+1,12,sparseRouting?4:1,false,true);uint64_t beforeWord=e.allocated;words=std::make_unique<VocabularyGeometry>(e,MODEL_V,wordBlocks);wordSceneBytes=e.allocated-beforeWord;if(rtTransmissionMLP)mlpSurfaces=std::make_unique<MLPTransmissionGeometry>(e,MODEL_D,MODEL_FF);if(sparseRouting)features=std::make_unique<ShapeMesh>(e,depth*(sparseRouting?4u:1u)*(middle/featureTile),featureTile*6,depth,false,true);scalars=std::make_unique<ShapeMesh>(e,1,65536*3,1,false,true);
        if(sparseRouting){for(uint32_t group=0;group<4;group++)for(uint32_t domain=0;domain<=depth;domain++)for(uint32_t b=0;b<8;b++) {uint32_t i=group*(depth+1)*8+domain*8+b;lenses->hostInstances[i].mask=1u<<b;lenses->hostInstances[i].instanceCustomIndex=domain*32+group*8+b;}}else for(uint32_t i=0;i<MODEL_FULL+1;i++){uint32_t domain=i<MODEL_FULL?MODEL_FULL_LAYERS[i]:depth;lenses->hostInstances[i].mask=1u;lenses->hostInstances[i].instanceCustomIndex=domain*32;}
        e.upload(lenses->instances,lenses->hostInstances.data(),lenses->instances->size);
        std::ifstream headFile(source,std::ios::binary);char basisMagic[8];uint32_t shape[2];headFile.read(basisMagic,8);headFile.read(reinterpret_cast<char*>(shape),8);if(!headFile||std::memcmp(basisMagic,quantization==4u?"RTBASQ01":"RTBAS001",8)||shape[0]!=MODEL_V||shape[1]!=MODEL_D)throw std::runtime_error("Complete model vocabulary basis required");size_t basisBytes=std::filesystem::file_size(source)-16u;std::vector<uint32_t> basisHost(basisBytes/4u);headFile.read(reinterpret_cast<char*>(basisHost.data()),basisBytes);if(!headFile)throw std::runtime_error("Truncated original word basis");basis=e.buffer(basisHost.size()*4,usage);e.upload(basis,basisHost.data(),basis->size);basisHost.clear();basisHost.shrink_to_fit();
        auto floats=[&](size_t n){return e.buffer(n*4,usage);};propagation=floats(size_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128*128+MODEL_S*9)+size_t(workRows)*MODEL_S*3);state=floats(size_t(workRows)*MODEL_D);normalized=floats(size_t(workRows)*MODEL_D);query=floats(size_t(workRows)*MODEL_A);key=floats(size_t(workRows)*MODEL_A);decoded=floats(1);geometry=floats(1);slots=floats(size_t(std::max(1u,MODEL_FULL))*capacity*2);rays=floats(std::max(size_t(workRows)*(sparseRouting?4u:1u),size_t(wordBlocks))*4u);hits=floats(std::max<size_t>(size_t(workRows)*(capacity+1),256));paths=floats(1);detector=key;physical=floats(size_t(workRows)*std::max(MODEL_STATE_HEADS,uint32_t(RT_ATTENTION_HEADS))*2);wordDecoded=floats(std::max(size_t(256)*MODEL_D,size_t(std::max(256u,workRows))*std::max(MODEL_D,std::max(MODEL_A,MODEL_W))/2u));wordResult=e.buffer(size_t(wordQueue+MODEL_V)*4,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);baseChart=floats(MODEL_D);feedForward=floats(size_t(workRows)*MODEL_W);auxiliary=floats(size_t(workRows)*MODEL_STATE_HEADS*2);prompt=floats(capacity);control=e.buffer(size_t(11u+capacity+65536u+attentionStatWords())*4u,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);audit=floats(1);featureWork=e.buffer(size_t(workRows*(16+MODEL_FF*(sparseRouting?2:1))+16)*4,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);featureRays=floats(size_t(workRows)*(sparseRouting?4u:1u)*4u);featureDown=floats(sparseRouting?size_t(depth)*MODEL_FF*1024:(wideCooperative&&MODEL_RECURRENT&&MODEL_STATE_HEADS%32u==0u?size_t(MODEL_RECURRENT)*MODEL_STATE_HEADS*MODEL_D:1));partialDown=floats(sparseRouting?size_t(workRows)*96*MODEL_D:1);
        attentionReceipts=e.buffer(4u,usage);
        pathScoreDummy=e.buffer(32u,usage);uint32_t noPathScores[8]{};e.upload(pathScoreDummy,noPathScores,sizeof(noPathScores));
        resizePathScores();
        attentionClassDummy=e.buffer(12u,usage);uint32_t emptyClassDirectory[3]{};e.upload(attentionClassDummy,emptyClassDirectory,12u);
        constexpr uint32_t classDomains=2u*(MODEL_GQA/RT_ATTENTION_HEAD_DIM);
        constexpr uint32_t classTempDomainWords=4u*RT_ATTENTION_HEAD_DIM+RT_ATTENTION_HEAD_DIM/4u+1u;
        attentionClassScratch=e.buffer(attentionBackend==4u?uint64_t(workRows)*classDomains*classTempDomainWords*4u:4u,usage,true);
        attentionClassCounts=e.buffer(4u,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        attentionClassLayers.resize(std::max(1u,MODEL_FULL));
        readback=e.buffer(4*1024*1024,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);
        if(cacheEnabled){initializeMarker(liveMarker,true);}
        VkDescriptorSetLayoutBinding bindings[46]{};uint32_t bindingCount=0;for(uint32_t i=0;i<46;i++){bindings[bindingCount++]={i,((i<5&&i!=2)||i==32||i==39||i==43)?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,i==6||i==32?depth:i==0||i==1||i==3?4u:1u,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};}
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sl.bindingCount=bindingCount;sl.pBindings=bindings;check(vkCreateDescriptorSetLayout(e.device,&sl,nullptr,&setLayout),"resident set layout");VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(ResidentConfig)};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&setLayout;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&pc;check(vkCreatePipelineLayout(e.device,&pl,nullptr,&layout),"resident pipeline layout");VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,(15+depth)*(depth+1)},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(depth+1)*(38+depth)}};VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.maxSets=depth+1;pool.poolSizeCount=2;pool.pPoolSizes=sizes;check(vkCreateDescriptorPool(e.device,&pool,nullptr,&descriptorPool),"resident descriptor pool");std::vector<VkDescriptorSetLayout> layouts(depth+1,setLayout);VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};alloc.descriptorPool=descriptorPool;alloc.descriptorSetCount=depth+1;alloc.pSetLayouts=layouts.data();std::vector<VkDescriptorSet> sets(depth+1);check(vkAllocateDescriptorSets(e.device,&alloc,sets.data()),"resident descriptor sets");layerSets.assign(sets.begin(),sets.begin()+depth);set=sets[0];initialSet=sets[depth];for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);for(uint32_t i=0;i<35;i++)if(i!=2&&i!=4&&i!=5&&i!=10)math[i]=e.pipeline("resident_math",layout,i,1u,0xffffffffu,wordTile,i==16u?readoutWords*32u:128u,featureTile,readoutUnroll,readoutWords,quantization==4u?(i==16u?p[15]:0xffffffffu):0xffffffffu);for(uint32_t i=0;i<14;i++)if(i<4||i>7||i==5)ray[i]=e.pipeline("resident_ray",layout,i,32u,0xffffffffu,wordTile,i==0||i==12||i==13?32u:128u,featureTile);
        for(uint32_t kind=2;kind<=5;kind++)if(!rtTransmissionMLP||kind==2u||kind==5u)for(uint32_t batch=0;batch<2;batch++)matrixProjection[kind-2][batch]=e.pipeline("resident_math",layout,2u,batch?matrixBatchRows:1u,kind,wordTile,128u,featureTile);
        if(rtTransmissionMLP)for(uint32_t kind=0;kind<2;kind++)for(uint32_t batch=0;batch<2;batch++)opticalMlp[kind][batch]=e.pipeline("resident_mlp",layout,kind,batch?matrixBatchRows:1u,0xffffffffu,wordTile,128u,featureTile);
        {
            std::map<std::pair<uint32_t,uint32_t>,VkPipeline> cached;
            const uint32_t projections[4]{0u,1u,6u,8u},formats[4]{10u,12u,22u,33u};
            for(uint32_t l=0;l<depth;l++){
                std::array<VkPipeline,4> handles{};
                for(uint32_t part=0;part<4u;part++){
                    uint32_t format=p[32u+l*64u+formats[part]];auto key=std::make_pair(projections[part],format);
                    if(!cached.count(key)){auto handle=e.pipeline("resident_math",layout,2u,1u,projections[part],wordTile,128u,featureTile,1u,4u,format);cached[key]=handle;ownedCooperative.push_back(handle);}
                    handles[part]=cached[key];
                }
                decodeQKV.push_back(handles);
            }
        }
        if(e.cooperativeMatrix2&&!(std::getenv("RT_PREFILL_COOPERATIVE")&&std::string(std::getenv("RT_PREFILL_COOPERATIVE"))=="0")){
            if(wideCooperative){
                matrixPack=e.pipeline("resident_pack",layout,0u,1u,0xffffffffu,wordTile,128u,featureTile);
                matrixAttention=e.pipeline("resident_attention",layout,0u,1u,0xffffffffu,wordTile,256u,featureTile);fastState=e.pipeline("resident_state",layout,21u,1u,0xffffffffu,wordTile,128u,featureTile);
                std::map<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>,VkPipeline> cache;
                auto pipe=[&](uint32_t stage,uint32_t pass,uint32_t format,uint32_t projection=0xffffffffu,uint32_t up=0xffffffffu){auto id=std::make_tuple(stage,pass,format,projection,up);if(!cache.count(id)){auto handle=e.pipeline("resident_batch2",layout,stage,pass,projection,wordTile,256u,featureTile,1u,4u,format,up);cache[id]=handle;ownedCooperative.push_back(handle);}return cache[id];};
                for(uint32_t l=0;l<depth;l++){uint32_t base=32u+l*64u;cooperativeOutputs.push_back(pipe(2u,1u,p[base+14u],2u));if(!rtTransmissionMLP){cooperativeGates.push_back(pipe(28u,1u,p[base+16u],0xffffffffu,p[base+20u]));cooperativeDowns.push_back(pipe(33u,1u,p[base+18u]));}cooperativeQKV.push_back({pipe(2u,1u,p[base+10u],0u),pipe(2u,1u,p[base+12u],1u),pipe(2u,1u,p[base+22u],6u),pipe(2u,1u,p[base+33u],8u)});
                    std::array<VkPipeline,2> auxiliaryPipes{},controlPack{};if(!fullLayer(l)&&MODEL_STATE_HEADS%32u==0u)for(uint32_t b=0;b<2;b++){
                        auxiliaryPipes[b]=e.pipeline("resident_auxiliary",layout,0u,b,0xffffffffu,wordTile,128u,featureTile,1u,4u,p[base+(b?39u:40u)]);controlPack[b]=e.pipeline("resident_auxiliary",layout,1u,b,0xffffffffu,wordTile,128u,featureTile,1u,4u,p[base+(b?39u:40u)]);ownedCooperative.push_back(auxiliaryPipes[b]);ownedCooperative.push_back(controlPack[b]);}cooperativeAuxiliary.push_back(auxiliaryPipes);cooperativeControlPack.push_back(controlPack);}
            }
        }
        if(quantization==4u){
            std::map<uint32_t,std::array<VkPipeline,2>> cached;
            for(uint32_t l=0;l<depth;l++){
                uint32_t format=p[32u+l*64u+14u];
                if(!cached.count(format)){
                    std::array<VkPipeline,2> pipes{};
                    for(uint32_t batch=0;batch<2;batch++)pipes[batch]=e.pipeline("resident_math",layout,2u,batch?matrixBatchRows:1u,2u,wordTile,128u,featureTile,1u,4u,format);
                    cached[format]=pipes;
                }
                outputPipelines.push_back(cached[format]);
            }
        }

        if(cacheEnabled){
            opticalPath=std::make_unique<OpticalPathKey>(e,state,propagation,keyGeometry,control,capacity);
            replyTree=std::make_unique<ReplyTreeCache>(e,control,wordResult,opticalPath->addresses);
            sharedHistory=std::make_unique<SharedHistoryCache>(e);
        }
        VkQueryPoolCreateInfo rtPool{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};rtPool.queryType=VK_QUERY_TYPE_TIMESTAMP;rtPool.queryCount=RT_TIMESTAMP_COUNT;
        check(vkCreateQueryPool(e.device,&rtPool,nullptr,&rtTimestamps),"create RT telemetry timestamps");
        if(attentionBackend==4u){rtPool.queryCount=2u;check(vkCreateQueryPool(e.device,&rtPool,nullptr,&attentionSubmitTimestamps),"create attention submission timestamps");}
    }
    void initializeMarker(OpticalStateMarker& snapshot,bool reply){
        snapshot.add(e,state,MODEL_D*4);
        // Boundary markers retain the resident optical encoding, including the
        // recurrent field and convolution history. Batch scratch is transient.
        // The live reply marker still needs only its detector row: its matching
        // field remains resident and is validated by gpuFieldAddress.
        if(!reply&&MODEL_RECURRENT){
            uint64_t bytes=uint64_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128u*128u+MODEL_S*9u)*4u;
            snapshot.add(e,propagation,bytes);
        }
    }
    void growTree(uint32_t next){
        if(next<=capacity||next>HISTORY_SLOT_MASK+1u||capacity<32u)throw std::runtime_error("Invalid tree growth request");
        if(!initialized){uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(control,initial,32);setup();}
        uint32_t previous=capacity;auto start=Clock::now();

        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto enlarged=std::make_unique<ResidentHistory>(e,depth,next);
        auto* newPaths=e.buffer(size_t(MODEL_FULL)*next*HISTORY_BYTES,usage);
        auto* newSlots=e.buffer(size_t(MODEL_FULL)*next*8u,usage);
        auto* newHits=e.buffer(size_t(workRows)*(next+1u)*4u,usage);
        auto* newPrompt=e.buffer(size_t(next)*4u,usage);
        auto* newControl=e.buffer(size_t(11u+next+65536u+attentionStatWords())*4u,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        auto c=e.command();begin(c);vkCmdFillBuffer(c,newSlots->handle,0,newSlots->size,0u);sync(c);
        if(absolutePosition)for(uint32_t layer=0;layer<MODEL_FULL;layer++){
            VkBufferCopy path{size_t(layer)*previous*HISTORY_BYTES,size_t(layer)*next*HISTORY_BYTES,size_t(previous)*HISTORY_BYTES};vkCmdCopyBuffer(c,keyGeometry->handle,newPaths->handle,1,&path);
            VkBufferCopy marks{size_t(layer)*previous*8u,size_t(layer)*next*8u,size_t(previous)*8u};vkCmdCopyBuffer(c,slots->handle,newSlots->handle,1,&marks);
            VkBufferCopy vertices{size_t(layer)*history->chunks*96u*12u,size_t(layer)*enlarged->chunks*96u*12u,size_t(history->chunks)*96u*12u};vkCmdCopyBuffer(c,history->vertices->handle,enlarged->vertices->handle,1,&vertices);
        }
        if(initialized){
            VkBufferCopy headers{0,0,32u};vkCmdCopyBuffer(c,control->handle,newControl->handle,1,&headers);
            VkBufferCopy scalars{size_t(11u+previous)*4u,size_t(11u+next)*4u,size_t(65536u+attentionStatWords())*4u};vkCmdCopyBuffer(c,control->handle,newControl->handle,1,&scalars);
            sync(c);if(absolutePosition)enlarged->build(c,enlarged->count,true);
        }
        check(vkEndCommandBuffer(c),"end GPU history tree growth");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);

        auto retiredHistory=std::move(history);history=std::move(enlarged);history->migrated.assign(MODEL_FULL,true);history->rebuildMoved=true;
        Buffer *retiredPaths=keyGeometry,*retiredSlots=slots,*retiredHits=hits,*retiredPrompt=prompt,*retiredControl=control;
        keyGeometry=newPaths;slots=newSlots;hits=newHits;prompt=newPrompt;control=newControl;
        // Class maps have their own compact layer stride; enlarging the history
        // address space does not allocate unused derived records.
        capacity=e.capacity=next;
        resizePathScores();
        if(cacheEnabled){opticalPath->rebind(state,propagation,keyGeometry,control,next);replyTree->rebind(control,wordResult);}
        for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);
        retiredHistory.reset();e.release(retiredPaths);e.release(retiredSlots);e.release(retiredHits);e.release(retiredPrompt);e.release(retiredControl);e.release(e.uploadStaging);
        std::vector<char> output;append(output,std::vector<double>{double(previous),double(next),ms(start)});reply(output);
    }
    void restoreSnapshot(VkCommandBuffer c,OpticalStateMarker& snapshot){
        snapshot.transfer(c,true);sync(c);
    }
    void capturePath(VkCommandBuffer c,uint32_t position,uint32_t index,const OpticalPathKey::Key* expected=nullptr){
        opticalPath->capture(c,position,index,cacheEpoch,expected);boundPipeline=VK_NULL_HANDLE;boundSet=VK_NULL_HANDLE;
    }
    void bind(VkDescriptorSet target,bool initial,uint32_t layer=0) {
        VkAccelerationStructureKHR handles[5][4]{};for(uint32_t i=0;i<4;i++){handles[0][i]=history->tops[fullIndex(layer)];handles[1][i]=lenses->tops[sparseRouting?i:0u];handles[3][i]=words->tops[0];}handles[2][0]=scalars->tops[0];handles[4][0]=scalars->tops[0];VkAccelerationStructureKHR featureHandles[RT_LAYER_COUNT]{};for(uint32_t l=0;l<depth;l++)featureHandles[l]=features?features->tops[l]:scalars->tops[0];VkWriteDescriptorSetAccelerationStructureKHR as[46]{};
        Buffer* classPayload=attentionClassDummy;
        Buffer* pathPayload=pathScoreDummy;
        if(!initial&&fullLayer(layer)&&!pathScoreLayers.empty())pathPayload=pathScoreLayers[fullIndex(layer)];
        if(!initial&&attentionBackend==4u&&fullLayer(layer)){
            auto* candidate=attentionClassLayers[fullIndex(layer)].buffer;if(candidate)classPayload=candidate;
        }
        Buffer* buffers[46]{nullptr,nullptr,propagation,nullptr,nullptr,parameters,matrices,basis,state,normalized,query,key,keyGeometry,initial?words->vertices:history->vertices,decoded,geometry,slots,rays,hits,paths,detector,physical,wordDecoded,scalars->vertices,wordResult,lenses->vertices,baseChart,feedForward,auxiliary,prompt,control,audit,nullptr,features?features->vertices:scalars->vertices,featureWork,featureRays,featureDown,partialDown,words->vertices,nullptr,pathPayload,pathScoreWork?pathScoreWork:pathScoreDummy,attentionReceipts,nullptr,attentionClassScratch,classPayload};VkDescriptorBufferInfo infos[46]{};VkWriteDescriptorSet writes[46]{};
        uint32_t writeCount=0;for(uint32_t i=0;i<46;i++){auto& w=writes[writeCount++];w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=target;w.dstBinding=i;w.descriptorCount=i==6||i==32?depth:i==0||i==1||i==3?4:1;w.descriptorType=((i<5&&i!=2)||i==32||i==39||i==43)?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;if((i<5&&i!=2)||i==32||i==39||i==43){as[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};as[i].accelerationStructureCount=w.descriptorCount;as[i].pAccelerationStructures=i==32?featureHandles:i==43?(attentionClassScene?attentionClassScene->tops.data():handles[0]):i==39?(mlpSurfaces?mlpSurfaces->tops.data():handles[0]):handles[i];w.pNext=&as[i];}else{infos[i]={buffers[i]->handle,0,buffers[i]->size};w.pBufferInfo=i==6?layerMatrices.data():&infos[i];}}vkUpdateDescriptorSets(e.device,writeCount,writes,0,nullptr);
    }
    static void sync(VkCommandBuffer c){Engine::barrier(c,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);}
    void begin(VkCommandBuffer c){e.begin(c);boundSet=VK_NULL_HANDLE;boundPipeline=VK_NULL_HANDLE;
        if(attentionSubmitTimestamps){vkCmdResetQueryPool(c,attentionSubmitTimestamps,0u,2u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,attentionSubmitTimestamps,0u);}}
    void bindCompute(VkCommandBuffer c,VkPipeline pipeline,VkDescriptorSet selected){if(boundPipeline!=pipeline){vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);boundPipeline=pipeline;}if(boundSet!=selected){vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&selected,0,nullptr);boundSet=selected;}}
    uint32_t startRT(VkCommandBuffer c,uint32_t kind){
        if(!recordRT)return ~0u;uint32_t at=uint32_t(rtKinds.size());if(at>=RT_QUERY_PAIRS)throw std::runtime_error("RT telemetry query capacity exceeded");
        rtKinds.push_back(kind);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,at*2u);return at;
    }
    void endRT(VkCommandBuffer c,uint32_t at){if(at!=~0u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,at*2u+1u);}
    void packMatrixInput(VkCommandBuffer c,ResidentConfig cfg,uint32_t source,VkDescriptorSet selected){
        cfg.stage=0u;cfg.index=source;uint32_t cols=source==0u?MODEL_D:source==1u?(fullLayer(cfg.layer)?MODEL_Q:MODEL_S):MODEL_FF;
        uint32_t groups=(((cfg.rows+255u)/256u)*256u*cols+255u)/256u;bindCompute(c,matrixPack,selected);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatch(c,std::min(groups,65535u),(groups+65534u)/65535u,1u);sync(c);
    }
    void launch(VkCommandBuffer c,VkPipeline* pipelines,ResidentConfig cfg,uint32_t x,uint32_t y=1,uint32_t z=1,VkDescriptorSet target=VK_NULL_HANDLE) {
        cfg.padding0=sparseRouting?1u:exactRT?2u:0u;std::memcpy(&cfg.padding1,&routeAperture,4);
        bool projection=pipelines==math&&cfg.stage==2u,batch=projection&&cfg.rows>1u;
        VkPipeline selected=projection?matrixProjection[cfg.index-2u][batch?1u:0u]:pipelines[cfg.stage];
        if(quantization==4u&&projection&&cfg.index==2u)selected=outputPipelines[cfg.layer][batch?1u:0u];
        if(cfg.rows>=8u&&wideCooperative&&pipelines==math&&cfg.stage==21u)selected=fastState;
        bindCompute(c,selected,target?target:layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        if(cfg.rows>=8u&&wideCooperative&&pipelines==math&&cfg.stage==18u&&cooperativeAuxiliary[cfg.layer][0]){
            for(auto handle:cooperativeAuxiliary[cfg.layer]){bindCompute(c,handle,target?target:layerSets[cfg.layer]);vkCmdDispatch(c,MODEL_STATE_HEADS/32u,(cfg.rows+127u)/128u,1u);}sync(c);return;
        }
        // Narrow heads use the subgroup detector; its reduction is reliable for
        // the 32-channel shape, while matrix projections still share weights.
        if(attentionBackend==0u&&cfg.rows>=8u&&wideCooperative&&RT_ATTENTION_HEAD_DIM>=64u&&pipelines==ray&&cfg.stage==3u){
            bindCompute(c,matrixAttention,target?target:layerSets[cfg.layer]);vkCmdDispatch(c,x,(cfg.rows+63u)/64u,1u);sync(c);return;
        }
        if(cfg.rows>=8u&&wideCooperative&&pipelines==math&&cfg.stage==21u){vkCmdDispatch(c,4u,MODEL_STATE_HEADS,1u);sync(c);return;}
        if(cfg.rows==1u&&projection&&cfg.index==5u){
            const uint32_t widths[4]{fullLayer(cfg.layer)?MODEL_Q:RT_STATE_KEY_HEADS*128u,fullLayer(cfg.layer)?MODEL_GQA:RT_STATE_KEY_HEADS*128u,fullLayer(cfg.layer)?RT_ATTENTION_GATE*MODEL_Q:MODEL_S,fullLayer(cfg.layer)?MODEL_GQA:MODEL_S};
            for(uint32_t part=0;part<4u;part++)if(widths[part]){
                bindCompute(c,decodeQKV[cfg.layer][part],target?target:layerSets[cfg.layer]);
                vkCmdDispatch(c,(widths[part]+3u)/4u,1u,1u);
            }
            sync(c);return;
        }
        if(cfg.rows>=8u&&wideCooperative&&projection&&cfg.index==5u){
            packMatrixInput(c,cfg,0u,target?target:layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
            const uint32_t widths[4]{fullLayer(cfg.layer)?MODEL_Q:RT_STATE_KEY_HEADS*128u,fullLayer(cfg.layer)?MODEL_GQA:RT_STATE_KEY_HEADS*128u,fullLayer(cfg.layer)?RT_ATTENTION_GATE*MODEL_Q:MODEL_S,fullLayer(cfg.layer)?MODEL_GQA:MODEL_S};
            for(uint32_t part=0;part<4;part++)if(widths[part]){vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);bindCompute(c,cooperativeQKV[cfg.layer][part],target?target:layerSets[cfg.layer]);vkCmdDispatch(c,(widths[part]+std::min(64u,uint32_t(RT_ATTENTION_HEAD_DIM))-1u)/std::min(64u,uint32_t(RT_ATTENTION_HEAD_DIM)),(y+127u)/128u,1u);}sync(c);return;
        }
        bool cooperative=cfg.rows>=8u&&wideCooperative&&!sparseRouting&&projection&&cfg.index>=2u&&cfg.index<=4u;
        if(cooperative){selected=cfg.index==2u?cooperativeOutputs[cfg.layer]:cfg.index==3u?cooperativeGates[cfg.layer]:cooperativeDowns[cfg.layer];packMatrixInput(c,cfg,cfg.index==2u?1u:cfg.index==3u?0u:2u,target?target:layerSets[cfg.layer]);bindCompute(c,selected,target?target:layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);}
        uint32_t query=~0u;if(pipelines==ray&&(cfg.stage==0u||cfg.stage==12u||cfg.stage==13u))query=startRT(c,cfg.stage==0u?0u:cfg.stage==12u?1u:2u);
        if(pipelines==ray&&cfg.stage==0u&&x>65535u){y=(x+65534u)/65535u;x=65535u;}
        uint32_t matrixChannelTile=std::min(64u,uint32_t(RT_ATTENTION_HEAD_DIM));
        uint32_t dx=cooperative?(x+matrixChannelTile/4u-1u)/(matrixChannelTile/4u):x,dy=cooperative?(y+255u)/256u:batch?(y+matrixBatchRows-1u)/matrixBatchRows:y;
        vkCmdDispatch(c,dx,dy,z);sync(c);

        endRT(c,query);
    }
    void payload(VkCommandBuffer c,ResidentConfig cfg){cfg.stage=8;launch(c,ray,cfg,1);}
    void resizePathScores(){
        for(auto* buffer:pathScoreLayers)e.release(buffer);pathScoreLayers.clear();
        e.release(pathScoreWork);pathScoreWork=nullptr;pathScorePages=0u;
        // Per-occurrence score arithmetic is stable in standard/ordered and
        // local-class decode. Other grouped producers have different policies.
        if(!cacheEnabled||pathScoreExhausted||!MODEL_FULL||
           (attentionBackend!=0u&&attentionBackend!=1u&&attentionBackend!=4u))return;
        uint64_t budget=128ull*1024u*1024u;
        if(auto* value=std::getenv("RT_PATH_SCORE_CACHE_MIB"))budget=std::stoull(value)*1024ull*1024ull;
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        // One complete target and one occurrence stamp cover every head. The
        // RT receipt contains only a slot flag and a compact list of gaps.
        uint64_t workBytes=(16u+2ull*capacity)*4u;
        uint64_t bankWords=4u+uint64_t(RT_ATTENTION_HEADS)*RT_ATTENTION_HEAD_DIM+
            uint64_t(capacity)*(2u+RT_ATTENTION_HEADS),layerBytes=0u;
        for(uint32_t pages=2u;pages;pages--){
            uint64_t bytes=(8u+pages*bankWords)*4u;
            if(bytes<=properties.limits.maxStorageBufferRange&&bytes<=UINT32_MAX&&
               workBytes<=properties.limits.maxStorageBufferRange&&workBytes<=UINT32_MAX&&
               workBytes<=budget&&uint64_t(MODEL_FULL)*bytes<=budget-workBytes){pathScorePages=pages;layerBytes=bytes;break;}
        }
        if(!pathScorePages)return;
        auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        uint64_t allocated=0u;pathScoreWork=e.buffer(workBytes,usage|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,false,true);allocated+=pathScoreWork->allocationSize;
        for(uint32_t full=0u;full<MODEL_FULL;full++){
            auto* buffer=e.buffer(layerBytes,usage,false,true);pathScoreLayers.push_back(buffer);allocated+=buffer->allocationSize;
        }
        if(allocated>budget){
            for(auto* buffer:pathScoreLayers)e.release(buffer);pathScoreLayers.clear();e.release(pathScoreWork);pathScoreWork=nullptr;pathScorePages=0u;return;
        }
        auto c=e.command();e.begin(c);resetPathScores(c);check(vkEndCommandBuffer(c),"end path score cache allocation");
        e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1u,&c);
    }
    void resetPathScores(VkCommandBuffer c){
        if(!pathScorePages)return;
        uint32_t header[8]{capacity,pathScorePages,RT_ATTENTION_HEADS,RT_ATTENTION_HEAD_DIM,pathScoreExhausted?0u:pathScoreMagic,attentionBackend,0u,0u};
        for(auto* buffer:pathScoreLayers){
            vkCmdFillBuffer(c,buffer->handle,0u,buffer->size,0u);sync(c);
            vkCmdUpdateBuffer(c,buffer->handle,0u,sizeof(header),header);
        }
        vkCmdFillBuffer(c,pathScoreWork->handle,0u,pathScoreWork->size,0u);sync(c);
    }
    uint32_t reservePathScoreSources(VkCommandBuffer c,uint32_t rows){
        if(!pathScorePages||pathScoreExhausted)return 0u;
        if(pathScoreSource+rows-1u>pathScoreSourceLimit){
            // Source identities must never alias a still-restorable block.
            // Disable numerical reuse for this resident instead of wrapping.
            for(auto* buffer:pathScoreLayers)vkCmdFillBuffer(c,buffer->handle,16u,4u,0u);
            sync(c);pathScoreExhausted=true;return 0u;
        }
        uint32_t first=uint32_t(pathScoreSource);pathScoreSource+=rows;return first;
    }
    void prepareDetector(){
        if(!detectorOrdered)return;
        const char* orderedModules[3]{"resident_detector_ordered_scores","resident_detector_ordered_normalize","resident_detector_ordered_value"};
        if(!detectorPipelines[0])for(uint32_t i=0;i<3u;i++)detectorPipelines[i]=e.pipeline(orderedModules[i],layout,0u,detectorBlock,0xffffffffu,wordTile,i==2u?32u:128u,featureTile);
        uint64_t stride=4u+detectorBlock;
        uint64_t bytes=uint64_t((capacity+detectorBlock-1u)/detectorBlock)*RT_ATTENTION_HEADS*stride*4u;
        if(bytes<=attentionReceipts->size)return;
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
        if(bytes>properties.limits.maxStorageBufferRange||bytes>UINT32_MAX){detectorOrdered=false;return;}
        e.release(attentionReceipts);attentionReceipts=e.buffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);
    }
    bool orderedDetector(const ResidentConfig& cfg)const{
        uint32_t visible=std::min(cfg.position+1u,capacity),kvHeads=RT_GQA_WIDTH/RT_ATTENTION_HEAD_DIM;
        return detectorOrdered&&cfg.rows==1u&&cfg.mode==1u&&visible>=detectorThreshold&&
            uint64_t((visible+detectorBlock-1u)/detectorBlock)*kvHeads<=65535u&&
            uint64_t(heads[cfg.layer])*((RT_ATTENTION_HEAD_DIM+31u)/32u)<=65535u;
    }
    void detectHistory(VkCommandBuffer c,ResidentConfig cfg){
        uint32_t visible=std::min(cfg.position+1u,capacity),kvHeads=RT_GQA_WIDTH/RT_ATTENTION_HEAD_DIM;
        uint64_t groups=uint64_t((visible+detectorBlock-1u)/detectorBlock)*kvHeads;
        uint64_t valueGroups=uint64_t(heads[cfg.layer])*((RT_ATTENTION_HEAD_DIM+31u)/32u);
        bool pathScores=pathScorePages&&!pathScoreExhausted&&cfg.rows==1u&&cfg.mode==1u&&(cfg.debug&131072u)==0u;
        if(!orderedDetector(cfg)){
            cfg.stage=3u;launch(c,ray,cfg,heads[cfg.layer],cfg.rows);
            // The fused legacy pass has no separate normalization dispatch.
            // Publish the one-per-slot stamp after every head has completed.
            if(pathScores){cfg.stage=10u;launch(c,ray,cfg,(visible+127u)/128u);}
            return;
        }
        cfg.stage=0u;bindCompute(c,detectorPipelines[0],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        // The existing RT payload pass publishes the command. sync() includes
        // DRAW_INDIRECT / INDIRECT_COMMAND_READ as well as shader consumers.
        if(pathScores)vkCmdDispatchIndirect(c,pathScoreWork->handle,8u*sizeof(uint32_t));
        else vkCmdDispatch(c,uint32_t(groups),1u,1u);
        sync(c);
        bindCompute(c,detectorPipelines[1],layerSets[cfg.layer]);vkCmdDispatch(c,heads[cfg.layer],1u,1u);sync(c);
        bindCompute(c,detectorPipelines[2],layerSets[cfg.layer]);vkCmdDispatch(c,uint32_t(valueGroups),1u,1u);sync(c);
    }
    void mark(VkCommandBuffer c,ResidentConfig cfg,uint32_t at){if(cfg.debug==2)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2+cfg.step*(depth*7+4)+cfg.layer*7+at);}
    void setup(){if(initialized)return;ResidentConfig cfg{7,1,0,0,0,0,0,capacity,depth,0,0,0};auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);vkCmdFillBuffer(c,featureRays->handle,0,featureRays->size,0u);sync(c);
        if(mlpSurfaces)mlpSurfaces->build(c,mlpSurfaces->count,true);launch(c,math,cfg,sparseRouting?(depth+1)*32:MODEL_FULL+1,1,1,initialSet);lenses->build(c,lenses->count,true);cfg.stage=8;launch(c,math,cfg,512,1,1,initialSet);scalars->build(c,1,true);cfg.stage=9;launch(c,math,cfg,std::min(MODEL_V,32768u),(MODEL_V+32767u)/32768u,wordBlocks,initialSet);words->build(c,words->count,true);cfg.index=cfg.position=0;if(features){cfg.stage=25;for(uint32_t l=0;l<depth;l++){cfg.layer=l;launch(c,math,cfg,MODEL_FF,sparseRouting?4u:1u,1,initialSet);if(sparseRouting){cfg.stage=31;launch(c,math,cfg,24576);}cfg.stage=25;}features->build(c,features->count,true);}cfg.stage=9;launch(c,ray,cfg,512);if(quantization!=4u){cfg.stage=22;cfg.index=0;for(uint32_t l=0;l<depth;l++){cfg.layer=l;launch(c,math,cfg,32768,(layerMatrices[l].range/4+32768*128-1)/(32768*128));}cfg.index=1;launch(c,math,cfg,32768,(MODEL_V*1024u+32768u*128u-1)/(32768u*128u));}if(wideCooperative)for(uint32_t l=0;l<depth;l++)if(cooperativeControlPack[l][0]){cfg.layer=l;vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);for(auto handle:cooperativeControlPack[l]){bindCompute(c,handle,layerSets[l]);vkCmdDispatch(c,MODEL_STATE_HEADS*MODEL_D/256u,1u,1u);sync(c);}}
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);check(vkEndCommandBuffer(c),"end resident first-forward geometry");auto time=e.measure(c,1);setupGPU=time.gpu;vkFreeCommandBuffers(e.device,e.pool,1,&c);
        auto coefficients=e.download<uint32_t>(control,11);if(coefficients[3]&512u)throw std::runtime_error("RT coefficient conversion: reference "+std::to_string(coefficients[7])+" measured "+std::to_string(coefficients[8])+" layer "+std::to_string(coefficients[9])+" basis "+std::to_string(coefficients[10])+" count "+std::to_string(coefficients[6]));initialized=true;if(!sparseRouting){wordSceneBytes-=words->scratchBytes();words->releaseScratch();lenses->releaseScratch();scalars->releaseScratch();if(mlpSurfaces)mlpSurfaces->releaseScratch();if(features)features->releaseScratch();}e.release(e.uploadStaging);e.release(e.downloadStaging);}
    void layer(VkCommandBuffer& c,ResidentConfig cfg) {
        mark(c,cfg,0);cfg.stage=1;cfg.index=0;launch(c,math,cfg,cfg.rows);cfg.stage=2;cfg.index=5;launch(c,math,cfg,fullLayer(cfg.layer)?(MODEL_Q+MODEL_GQA)/2u:quantization==4u?(2u*RT_STATE_KEY_HEADS*128u+2u*MODEL_S+3u)/4u:MODEL_D,cfg.rows);
        if(fullLayer(cfg.layer)){
            if(RT_QK_NORM){cfg.stage=14;launch(c,math,cfg,heads[cfg.layer],cfg.rows);}
            uint32_t historyLower=cfg.checkpoint;cfg.checkpoint=reservePathScoreSources(c,cfg.rows);
            cfg.stage=3;launch(c,math,cfg,(MODEL_Q+255u)/256u,cfg.rows);cfg.checkpoint=historyLower;
            cfg.stage=34;launch(c,math,cfg,1,cfg.rows);
cfg.stage=6;launch(c,math,cfg,1,cfg.rows);
            if(opticalPath){opticalPath->updateHistory(c,fullIndex(cfg.layer),cfg.mode==0u?0u:cfg.position,cfg.rows,cfg.mode==1u);boundPipeline=VK_NULL_HANDLE;boundSet=VK_NULL_HANDLE;}
            classifyAttentionHistory(c,cfg);
            mark(c,cfg,1);uint32_t first=cfg.mode==0?0:cfg.position%capacity,last=cfg.mode==0?cfg.rows-1:cfg.mode==1?first:(cfg.position+cfg.rows-1)%capacity;history->update(c,cfg.layer,first,last);mark(c,cfg,2);
            if(pathScorePages&&!pathScoreExhausted&&cfg.rows==1u&&cfg.mode==1u){
                uint32_t previousIndex=cfg.index;
                cfg.index=attentionBackend==4u?2u:orderedDetector(cfg)?1u:0u;
                cfg.stage=1u;launch(c,ray,cfg,1u);cfg.index=previousIndex;
            }
            if(attentionBackend>=2u){
                traceAttentionReceipts(c,cfg);mark(c,cfg,3);payload(c,cfg);mark(c,cfg,4);mergeAttentionReceipts(c,cfg);mark(c,cfg,5);
            }else{
            vkCmdFillBuffer(c,control->handle,24,4,0);vkCmdFillBuffer(c,hits->handle,0,size_t(cfg.rows)*(capacity+1u)*4u,0);sync(c);cfg.stage=0;launch(c,ray,cfg,(cfg.rows*((capacity+15u)/16u)+7u)/8u);mark(c,cfg,3);payload(c,cfg);mark(c,cfg,4);
            detectHistory(c,cfg);mark(c,cfg,5);
            }
        }else{
            cfg.stage=18;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);cfg.stage=19;launch(c,math,cfg,(MODEL_S*3u+127u)/128u,cfg.rows);cfg.stage=20;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);
            mark(c,cfg,1);mark(c,cfg,2);mark(c,cfg,3);mark(c,cfg,4);cfg.stage=21;launch(c,math,cfg,32,MODEL_STATE_HEADS);mark(c,cfg,5);cfg.stage=15;launch(c,math,cfg,MODEL_STATE_HEADS,cfg.rows);
        }
        cfg.stage=2;cfg.index=2;launch(c,math,cfg,MODEL_D/4u,cfg.rows);cfg.stage=1;cfg.index=1;launch(c,math,cfg,cfg.rows);executeMLP(c,cfg);if(cfg.debug==1){VkBufferCopy snapshot{0,(size_t(cfg.layer)*captureRows+(cfg.mode==0u?0u:cfg.position))*MODEL_D*4u,size_t(cfg.rows)*MODEL_D*4u};vkCmdCopyBuffer(c,state->handle,audit->handle,1,&snapshot);sync(c);}mark(c,cfg,6);
    }
    size_t featureGlobal()const{return size_t(workRows)*(16u+MODEL_FF*(sparseRouting?2u:1u));}
    void executeMLP(VkCommandBuffer& c,ResidentConfig cfg){
        if(rtTransmissionMLP||sparseRouting){routedMLP(c,cfg);return;}
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
        cfg.stage=2u;cfg.index=3u;launch(c,math,cfg,(middle+3u)/4u,cfg.rows);
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);
        cfg.index=4u;launch(c,math,cfg,MODEL_D/4u,cfg.rows);
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,3u);
    }
    void opticalMLP(VkCommandBuffer c,ResidentConfig cfg){
        cfg.padding0=2u;
        uint32_t tile=cfg.rows>1u?matrixBatchRows:1u,groups=(cfg.rows+tile-1u)/tile;
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
        for(uint32_t pass=0;pass<2u;pass++){
            cfg.stage=pass;bindCompute(c,opticalMlp[pass][cfg.rows>1u?1u:0u],layerSets[cfg.layer]);
            vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
            uint32_t stamp=startRT(c,2u);
            vkCmdDispatch(c,(pass==0u?MODEL_FF:MODEL_D)/4u,groups,1u);sync(c);endRT(c,stamp);
            if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u+pass);
        }
        featureTraceRows+=groups;featureReuseRows+=cfg.rows-groups;
    }
    void routedMLP(VkCommandBuffer& c,ResidentConfig cfg){
        vkCmdFillBuffer(c,featureWork->handle,0,size_t(cfg.rows)*(16u+MODEL_FF*(sparseRouting?2u:1u))*4u,0u);sync(c);
        if(!sparseRouting){opticalMLP(c,cfg);return;}
        cfg.stage=26;launch(c,math,cfg,4u,cfg.rows);
        cfg.stage=13;launch(c,ray,cfg,((middle/featureTile)*4u+31u)/32u,cfg.rows);featureTraceRows+=cfg.rows;
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1u);
        cfg.stage=27;launch(c,math,cfg,1);
        cfg.stage=28;cfg.padding0=1u;std::memcpy(&cfg.padding1,&routeAperture,4);
        bindCompute(c,math[28],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatchIndirect(c,featureWork->handle,featureGlobal()*4u);sync(c);
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u);
        cfg.stage=29;bindCompute(c,math[29],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);
        vkCmdDispatchIndirect(c,featureWork->handle,(featureGlobal()+4u)*4u);sync(c);
        if(cfg.debug==4u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,3u);
        cfg.stage=32;launch(c,math,cfg,8,cfg.rows);
    }
    void forward(VkCommandBuffer& c,ResidentConfig cfg){cfg.stage=0;launch(c,math,cfg,cfg.rows);for(uint32_t l=0;l<depth;l++){cfg.layer=l;layer(c,cfg);}cfg.stage=1;cfg.index=2;launch(c,math,cfg,cfg.rows);}
    void wordMark(VkCommandBuffer c,ResidentConfig cfg,uint32_t at){if(cfg.debug==2u)vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,2u+cfg.step*(depth*7u+4u)+depth*7u+at);}
    void wordTrace(VkCommandBuffer c,ResidentConfig cfg){
        vkCmdFillBuffer(c,wordResult->handle,4096u*4u,8u,0u);
        vkCmdFillBuffer(c,wordResult->handle,(8192u+MODEL_V)*4u,size_t(MODEL_V)*wordMaskWords*4u,0u);sync(c);
        cfg.stage=11;launch(c,math,cfg,wordBlocks);cfg.stage=12;launch(c,ray,cfg,((wordTiles+31)/32)*wordBlocks);
    }
    void word(VkCommandBuffer c,ResidentConfig cfg){wordMark(c,cfg,0);wordTrace(c,cfg);wordMark(c,cfg,1);
        cfg.stage=24;launch(c,math,cfg,1);cfg.stage=16;cfg.padding0=sparseRouting?1u:exactRT?2u:0u;std::memcpy(&cfg.padding1,&routeAperture,4);bindCompute(c,math[16],layerSets[cfg.layer]);vkCmdPushConstants(c,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cfg),&cfg);vkCmdDispatchIndirect(c,wordResult->handle,6144u*4u);sync(c);wordMark(c,cfg,2);
        cfg.stage=17;launch(c,math,cfg,128);cfg.stage=23;launch(c,math,cfg,1);wordMark(c,cfg,3);}
    void copy(VkCommandBuffer c,Buffer* source,size_t bytes,size_t destination,size_t offset=0){VkBufferCopy region{offset,destination,bytes};vkCmdCopyBuffer(c,source->handle,readback->handle,1,&region);}
    void ensureReadback(size_t bytes){if(readback->size>=bytes)return;e.release(readback);readback=e.buffer(bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);}
    void hostBarrier(VkCommandBuffer c){Engine::barrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);}
    void reply(const std::vector<char>& bytes){uint32_t header[2]{0,uint32_t(bytes.size())};std::cout.write(reinterpret_cast<char*>(header),8);std::cout.write(bytes.data(),bytes.size());std::cout.flush();outboundBytes+=bytes.size();}
    void prepareBoundary(const std::shared_ptr<SharedHistoryCache::Entry>& entry,const std::vector<SharedHistoryCache::Key>& keys){
        initializeMarker(entry->snapshot,false);
        for(uint32_t first=0;first<entry->input.size();first+=16u){
            uint32_t n=std::min(16u,uint32_t(entry->input.size())-first);auto identity=keys[first+n];
            auto block=sharedHistory->blocks[identity].lock();
            if(!block){block=std::make_shared<SharedHistoryCache::Block>(e,first,n);sharedHistory->blocks[identity]=block;}
            entry->blocks.push_back(block);
        }
    }
    void saveBoundary(VkCommandBuffer& c,const std::shared_ptr<SharedHistoryCache::Entry>& entry,uint32_t row,
                      const std::vector<std::shared_ptr<SharedHistoryCache::Block>>* matchingBlocks=nullptr){
        for(auto& item:entry->snapshot.items){VkBufferCopy copy{item.offset+(item.source==state?size_t(row)*MODEL_D*4u:0u),0,item.bytes};vkCmdCopyBuffer(c,item.source->handle,item.saved->handle,1,&copy);}
        for(auto& b:entry->blocks){
            bool newPackets=!b->ready;
            if(newPackets){
                for(uint32_t layer=0;layer<MODEL_FULL;layer++){
                    VkBufferCopy packet{(size_t(layer)*capacity+b->first)*HISTORY_BYTES,size_t(layer)*b->count*HISTORY_BYTES,size_t(b->count)*HISTORY_BYTES};
                    vkCmdCopyBuffer(c,keyGeometry->handle,b->packets->handle,1,&packet);
                    VkBufferCopy mark{(size_t(layer)*capacity+b->first)*8u,size_t(layer)*b->count*8u,size_t(b->count)*8u};
                    vkCmdCopyBuffer(c,slots->handle,b->marks->handle,1,&mark);
                    VkBufferCopy shape{size_t(layer)*history->chunks*96u*12u+size_t(b->first)*72u,size_t(layer)*96u*12u,size_t(b->count)*72u};
                    vkCmdCopyBuffer(c,history->vertices->handle,b->scene->vertices->handle,1,&shape);
                }
                sync(c);b->scene->build(c,b->scene->count,false);b->ready=true;
            }
            // An older immutable packet can gain its derived compact payload
            // after restoration. Only that exact restored block, or the packet
            // frozen above in this command, establishes the source provenance.
            bool matching=newPackets||(matchingBlocks&&std::find(matchingBlocks->begin(),matchingBlocks->end(),b)!=matchingBlocks->end());
            saveAttentionBlock(c,*b,matching);
        }
        sync(c);
    }
    void prepareAttachment(const std::shared_ptr<SharedHistoryCache::Entry>& entry){
        auto instances=history->hostInstances;
        if(entry)for(auto& b:entry->blocks)if(b->count==16u)for(uint32_t layer=0;layer<MODEL_FULL;layer++){
            VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};address.accelerationStructure=b->scene->leaves[layer].handle;
            instances[layer*history->chunks+b->first/16u].accelerationStructureReference=e.asAddress(e.device,&address);
        }
        e.upload(history->instances,instances.data(),history->instances->size);
    }
    void attachBoundary(VkCommandBuffer c,const std::shared_ptr<SharedHistoryCache::Entry>& entry){
        if(opticalPath)opticalPath->invalidateHistory();
        vkCmdFillBuffer(c,slots->handle,0,slots->size,0u);sync(c);
        restoreSnapshot(c,entry->snapshot);
        for(auto& b:entry->blocks)for(uint32_t layer=0;layer<MODEL_FULL;layer++){
            VkBufferCopy packet{size_t(layer)*b->count*HISTORY_BYTES,(size_t(layer)*capacity+b->first)*HISTORY_BYTES,size_t(b->count)*HISTORY_BYTES};
            vkCmdCopyBuffer(c,b->packets->handle,keyGeometry->handle,1,&packet);
            VkBufferCopy mark{size_t(layer)*b->count*8u,(size_t(layer)*capacity+b->first)*8u,size_t(b->count)*8u};
            vkCmdCopyBuffer(c,b->marks->handle,slots->handle,1,&mark);
            VkBufferCopy shape{size_t(layer)*96u*12u,size_t(layer)*history->chunks*96u*12u+size_t(b->first)*72u,size_t(b->count)*72u};
            vkCmdCopyBuffer(c,b->scene->vertices->handle,history->vertices->handle,1,&shape);
        }
        sync(c);
        // Complete chunks attach existing BLAS references. Only the appendable
        // partial chunk needs a mutable BLAS; every layer attaches via its TLAS.
        uint32_t tokens=uint32_t(entry->input.size());
        // prepareAttentionClassAttachment allocated and relocated compact
        // entries before this command began. Reuse their existing shader ABI;
        // only blocks without a saved payload remain pending classification.
        restoreAttentionClassAttachment(c,tokens);
        for(uint32_t layer=0;layer<MODEL_FULL;layer++){
            uint32_t offset=layer*history->chunks;
            if(tokens%16u)history->build(c,offset+tokens/16u+1u,true,false,offset+tokens/16u,layer);
            else history->build(c,history->count,false,true,0u,layer);
        }
    }
    void prefill(uint32_t count,uint32_t debug,uint32_t checkpointMode=0u){
        bool capture=debug==1u,profile=debug==2u;
        if(debug>2u||!count||count>capacity)throw std::runtime_error("Invalid resident prefix");
        ensureReadback(capture?size_t(depth+1)*count*MODEL_D*4+1024:4*1024*1024);
        if(capture){
            captureRows=count;uint64_t auditBytes=uint64_t(depth)*count*MODEL_D*4u;
            VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(e.physical,&properties);
            if(auditBytes>properties.limits.maxStorageBufferRange)throw std::runtime_error("Layer capture exceeds the device storage-buffer range; use a shorter prefix");
            if(audit->size<auditBytes){
                e.release(audit);audit=e.buffer(auditBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                // Transfer detector bits to the capture buffer.
                VkDescriptorBufferInfo info{audit->handle,0,audit->size};std::vector<VkWriteDescriptorSet> writes(depth+1u);
                for(uint32_t l=0;l<=depth;l++){auto& w=writes[l];w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=l==depth?initialSet:layerSets[l];w.dstBinding=31u;w.descriptorCount=1u;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&info;}
                vkUpdateDescriptorSets(e.device,uint32_t(writes.size()),writes.data(),0u,nullptr);
            }
        }
        std::vector<uint32_t> ids;input(ids,count);std::cin.read(reinterpret_cast<char*>(stops.data()),8);
        uint32_t boundary=count;
        if(checkpointMode==1u)std::cin.read(reinterpret_cast<char*>(&boundary),4);
        if(checkpointMode==2u){uint32_t n;std::cin.read(reinterpret_cast<char*>(&n),4);if(n>4096u)throw std::runtime_error("Too many history boundaries");
            uint32_t stable=0u;
            for(uint32_t i=0;i<n;i++){uint32_t value[2];std::cin.read(reinterpret_cast<char*>(value),8);if(!value[0]||value[0]>count||!(value[1]&15u)||value[1]>15u)throw std::runtime_error("Invalid history boundary");if(value[1]&1u)stable=std::max(stable,value[0]);}
            if(stable)boundary=stable;
        }
        if(!std::cin||!boundary||boundary>count||*std::max_element(ids.begin(),ids.end())>=MODEL_V)throw std::runtime_error("Invalid prefix word IDs or boundary");
        uint64_t traceStart=featureTraceRows,reuseStart=featureReuseRows;auto start=Clock::now();
        std::vector<SharedHistoryCache::Key> keys;std::shared_ptr<SharedHistoryCache::Entry> hit;
        bool useCache=cacheEnabled&&!capture;
        if(useCache){keys=SharedHistoryCache::identities(ids,cacheEpoch);hit=sharedHistory->find(ids,keys);}
        uint32_t reused=hit?uint32_t(hit->input.size()):0u;bool addressHit=hit&&reused==count&&hit->addressReady&&hit->stops==stops;
        // A matched boundary carries both its shared history Tree and encoded
        // optical field, so only the suffix is new computation.
        uint32_t rows=count-reused,rebuildTokens=0u;
        uint64_t boundaryBytes=uint64_t(MODEL_D)*4u+uint64_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128u*128u+MODEL_S*9u)*4u;
        uint32_t snapshotKind=hit?(reused==count?1u:reused==boundary?2u:3u):0u;
        // Plan bounded storage before recording GPU commands. Entries and blocks
        // stay pinned until the command buffer has completed.
        std::map<uint32_t,std::shared_ptr<SharedHistoryCache::Entry>> pending;
        if(useCache){uint32_t maximum=sharedHistory->maxEntries(boundaryBytes),selected=0;
            std::set<uint32_t> wanted;
            // A recurrent batch exposes its terminal field, not intermediate
            // row fields. Save only actual batch ends without changing the
            // source operator's batching or allocating a field per token.
            for(uint32_t end=count;end>reused&&selected<maximum;){wanted.insert(end);selected++;
                uint32_t previous=reused+((end-reused-1u)/workRows)*workRows;
                if(previous<=reused)break;end=previous;}
            for(auto position:wanted){uint32_t kind=position==count?17u:4u;
                auto entry=sharedHistory->reserve(ids,position,kind,keys[position],boundaryBytes);if(entry){prepareBoundary(entry,keys);pending[position]=entry;}}
        }
        uint32_t initial[8]{0,0,1,0,stops[0],stops[1],0,0};
        if(rows)e.upload(prompt,ids.data(),count*4u);e.upload(control,initial,32);inboundBytes+=(rows?count*4u:0u)+40u;setup();
        prepareAttachment(hit);e.release(e.uploadStaging);
        std::vector<std::pair<uint32_t,uint32_t>> spans;
        for(uint32_t offset=reused;offset<count;){uint32_t end=std::min(count,offset+workRows);
            uint32_t length=end-offset;spans.push_back({offset,length});offset+=length;}
        uint32_t batches=uint32_t(spans.size()),queryCount=profile?2u+batches*(depth*7u+4u):2u;e.ensureTimestamps(queryCount);
        double splitGPU=attentionSplitGPU,splitWait=attentionSplitWait;
        attentionSegmentEnd=0u;attentionProfileSlot=~0u;attentionProfileGaps.clear();
        bool sameLiveClasses=sameLiveAttentionClasses(hit);
        if(attentionBackend==4u){
            // Test the live prefix before a resize can discard its directory.
            resizeAttentionClasses(count,sameLiveClasses);prepareStaticAttentionClassScene();
        }
        prepareAttentionClassAttachment(hit,sameLiveClasses);
        if(attentionBackend>=2u&&rows){uint32_t activeRows=std::min(count,workRows);prepareAttention(activeRows,count-activeRows);}
        auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,queryCount);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        if(hit)attachBoundary(c,hit);else{resetAttentionClassPending();if(opticalPath)opticalPath->resetHistory(c);vkCmdFillBuffer(c,slots->handle,0,slots->size,0u);vkCmdFillBuffer(c,propagation->handle,0,propagation->size,0u);sync(c);}
        uint32_t lastRows=0;
        for(uint32_t batch=0;batch<batches;batch++){uint32_t offset=spans[batch].first;lastRows=spans[batch].second;uint32_t position=offset;
            ResidentConfig cfg{0,lastRows,0,position,batch,position?2u:0u,offset,capacity,depth,debug,0,0,0};forward(c,cfg);
            for(auto it=pending.upper_bound(position);it!=pending.end()&&it->first<=position+lastRows;it++)
                saveBoundary(c,it->second,it->first-position-1u,hit?&hit->blocks:nullptr);
            if(capture){copy(c,state,size_t(lastRows)*MODEL_D*4,32+size_t(offset)*MODEL_D*4);sync(c);}
        }
        // A restored block may predate optional compact capture. After the
        // suffix classified every missing interval, fill that derived cache
        // even when no new boundary marker was reserved for this request.
        if(hit&&rows){for(const auto& block:hit->blocks)saveAttentionBlock(c,*block,true);sync(c);}
        if(lastRows>1u){VkBufferCopy last{size_t(lastRows-1u)*MODEL_D*4,0,MODEL_D*4};vkCmdCopyBuffer(c,state->handle,state->handle,1,&last);sync(c);}
        if(cacheEnabled){if(!addressHit)liveMarker.transfer(c,false);sync(c);}
        if(cacheEnabled&&!addressHit)capturePath(c,count,64u);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);copy(c,control,32,0);size_t bytes=32;
        if(capture){bytes+=size_t(count)*MODEL_D*4;for(uint32_t l=0;l<depth;l++){copy(c,audit,size_t(count)*MODEL_D*4,bytes,size_t(l)*captureRows*MODEL_D*4);bytes+=size_t(count)*MODEL_D*4;}}
        if(cacheEnabled&&!addressHit)copy(c,opticalPath->addresses,sizeof(pathAddress),bytes,64u*sizeof(pathAddress));
        hostBarrier(c);check(vkEndCommandBuffer(c),"end resident prefill");auto time=attentionTiming(e.measure(c,1),splitGPU,splitWait);vkFreeCommandBuffers(e.device,e.pool,1,&c);absolutePosition=count;
        auto* stats=static_cast<uint32_t*>(readback->map);if(stats[3])throw std::runtime_error("Resident prefix optical error "+std::to_string(stats[3]));
        attentionClassRestoreCopies.clear();attachedBlocks.clear();if(hit)attachedBlocks=hit->blocks;
        prefixTokens=count;currentGenerated.clear();stateCursor=0;attentionDiagnosticDirty=false;
        if(cacheEnabled){if(addressHit)pathAddress=hit->address;else std::memcpy(&pathAddress,static_cast<char*>(readback->map)+bytes,sizeof(pathAddress));}
        if(cacheEnabled){gpuFieldAddress=pathAddress;gpuFieldPosition=count;gpuFieldReady=true;}
        if(useCache){for(auto& entry:sharedHistory->entries)if(entry&&entry->input==ids){entry->address=pathAddress;entry->addressReady=true;entry->stops=stops;}}

        if(useCache){cachedPrefix=ids;prefixReady=true;stableTokens=boundary;stableReady=true;if(!pending.empty())sharedHistory->publish();}
        else{prefixReady=false;stableReady=false;}
        std::vector<char> output;append(output,std::vector<double>{setupGPU,time.gpu,time.wall,ms(start),double(bool(hit)),double(reused)});
        output.insert(output.end(),static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);
        append(output,std::vector<uint64_t>{workRows,batches,batches*MODEL_FULL,uint64_t(rows)*MODEL_FULL});
        std::array<double,2> detectorTiming{};
        if(profile){std::vector<double> phases(6,0);for(uint32_t batch=0;batch<batches;batch++)for(uint32_t l=0;l<depth;l++){uint64_t stamps[7];check(vkGetQueryPoolResults(e.device,e.timestamps,2u+batch*(depth*7u+4u)+l*7u,7u,sizeof(stamps),stamps,8u,VK_QUERY_RESULT_64_BIT),"read prefill layer timing");for(uint32_t i=0;i<6u;i++)phases[i]+=double(stamps[i+1u]-stamps[i])*e.period/1e6;detectorTiming[fullLayer(l)?0u:1u]+=double(stamps[5]-stamps[4])*e.period/1e6;uint32_t at=batch*depth+l;if(at<attentionProfileGaps.size())phases[0]-=attentionProfileGaps[at];}append(output,phases);}
        append(output,std::vector<uint64_t>{featureTraceRows-traceStart,featureReuseRows-reuseStart});
        append(output,std::vector<uint64_t>{stableReady?stableTokens:0u,snapshotKind});
        append(output,std::vector<uint64_t>{hit?hit->blocks.size():0u,pending.size(),sharedHistory?sharedHistory->size():0u,sharedHistory?sharedHistory->lookupRays:0u,uint64_t(rows),sharedHistory?sharedHistory->evictions:0u});
        append(output,std::vector<double>{useCache?sharedHistory->lookupMs:0.});append(output,std::vector<uint64_t>{rebuildTokens,uint64_t(addressHit)});
        if(profile)append(output,std::vector<double>(detectorTiming.begin(),detectorTiming.end()));reply(output);
        if(capture){e.release(audit);audit=e.buffer(4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT);for(uint32_t l=0;l<depth;l++)bind(layerSets[l],false,l);bind(initialSet,true);if(readback->size>4*1024*1024){e.release(readback);readback=e.buffer(4*1024*1024,VK_BUFFER_USAGE_TRANSFER_DST_BIT,true);}}
    }
    bool currentFieldMatches()const{
        return gpuFieldReady&&gpuFieldPosition==absolutePosition&&gpuFieldAddress.key==pathAddress.key;
    }
    std::shared_ptr<SharedHistoryCache::Entry> outputBoundary(){
        if(!prefixReady||currentFieldMatches()||stateCursor==currentGenerated.size())return {};
        uint32_t minimum=prefixTokens+stateCursor+1u;
        // Query the existing history directory only when a saved terminal state
        // could replace a replay. Ordinary new-token decode does no lookup.
        bool available=false;
        for(const auto& entry:sharedHistory->entries)if(entry&&entry->input.size()>=minimum&&entry->input.size()<=absolutePosition&&entry->input.size()<=capacity){available=true;break;}
        if(!available)return {};
        auto input=cachedPrefix;input.insert(input.end(),currentGenerated.begin(),currentGenerated.end());
        auto keys=SharedHistoryCache::identities(input,cacheEpoch);
        return sharedHistory->find(input,keys,minimum);
    }
    void materializeCachedState(VkCommandBuffer& c,const std::shared_ptr<SharedHistoryCache::Entry>& boundary){
        if(boundary){
            attachBoundary(c,boundary);stateCursor=uint32_t(boundary->input.size())-prefixTokens;
        }else if(currentFieldMatches()){
            restoreSnapshot(c,liveMarker);stateCursor=uint32_t(currentGenerated.size());return;
        }
        for(uint32_t i=stateCursor;i<currentGenerated.size();i++){
            uint32_t token=currentGenerated[i];vkCmdUpdateBuffer(c,control->handle,0,4,&token);sync(c);
            ResidentConfig cfg{0,1,0,prefixTokens+i,0,1,0,capacity,depth,0,0,0,0};forward(c,cfg);
        }
        stateCursor=uint32_t(currentGenerated.size());
    }
    void acceptTokens(const uint32_t* ids,uint32_t count){
        for(uint32_t i=0;i<count;i++){currentGenerated.push_back(ids[i]);absolutePosition++;}
    }
    void finishHistory(){
        auto start=Clock::now();uint32_t saved=0,tokens=0;
        if(cacheEnabled&&prefixReady&&!currentGenerated.empty()&&stateCursor==currentGenerated.size()&&absolutePosition<=capacity){
            auto ids=cachedPrefix;ids.insert(ids.end(),currentGenerated.begin(),currentGenerated.end());
            // A stopping token selects completion but has no forward state.
            // Persist precisely the prefix whose state has been computed.
            if(ids.back()==stops[0]||ids.back()==stops[1])ids.pop_back();
            tokens=uint32_t(ids.size());auto keys=SharedHistoryCache::identities(ids,cacheEpoch);
            uint64_t boundaryBytes=uint64_t(MODEL_D)*4u+uint64_t(MODEL_RECURRENT)*(MODEL_STATE_HEADS*128u*128u+MODEL_S*9u)*4u;
            auto entry=sharedHistory->reserve(ids,tokens,2u,keys[tokens],boundaryBytes);
            if(entry){prepareBoundary(entry,keys);auto c=e.command();begin(c);saveBoundary(c,entry,0u,&attachedBlocks);
                check(vkEndCommandBuffer(c),"end output boundary capture");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);sharedHistory->publish();saved=1;}
        }
        // Pending gaps belong to the attached history and are consumed by
        // classification. Finishing before decode must retain those intervals.
        attentionClassRestoreCopies.clear();
        std::vector<char> output;append(output,std::vector<uint64_t>{saved,tokens,sharedHistory?sharedHistory->size():0u});append(output,std::vector<double>{ms(start)});reply(output);
    }
    GenerationPacket generatePacket(uint32_t count,uint32_t debug){
        if(attentionDiagnosticDirty)throw std::runtime_error("Attention diagnostics replace history; prefill again before generation");
        if(!initialized||!count||count>64)throw std::runtime_error("Resident generation expects 1..64 tokens after prefill");
        uint64_t traceStart=featureTraceRows,reuseStart=featureReuseRows;uint32_t skippedRows=0;
        uint32_t cachedCount=0,cacheQueries=0,cacheLeaf=0xffffffffu,submissions=0;
        double gpu=0,wall=0,cacheGPU=0,cacheWall=0,cacheRT=0,restoreGPU=0;
        uint32_t restoreTokens=0,candidates=0,rejected=0;bool restoredCheckpoint=false,recoveryRecorded=false;
        std::array<uint32_t,8> header{};std::array<uint32_t,64> ids{};
        std::array<uint32_t,5> counts{};std::array<double,10> parts{};std::array<double,4> rtParts{};std::array<double,2> detectorTiming{};
        bool probe=cacheEnabled&&debug!=1u&&debug!=2u;
        if(probe){
            auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdResetQueryPool(c,rtTimestamps,0,2);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,0);
            replyTree->probe(c,pathAddress,absolutePosition,count);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,1);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);
            copy(c,control,32,0);copy(c,wordResult,256,32,528u*4u);copy(c,replyTree->receipt,64,288);hostBarrier(c);
            check(vkEndCommandBuffer(c),"end RT cached path query");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
            auto* result=static_cast<uint32_t*>(readback->map);std::copy_n(result,8,header.begin());std::copy_n(result+8,64,ids.begin());
            cachedCount=result[72];cacheLeaf=result[73];cacheQueries=result[74];
            candidates=result[86];rejected=result[87];
            if(cachedCount>count||header[1]!=cachedCount)throw std::runtime_error("Invalid cached path receipt");
            uint64_t stamps[2];check(vkGetQueryPoolResults(e.device,rtTimestamps,0,2,16,stamps,8,VK_QUERY_RESULT_64_BIT),"read cache RT timing");
            cacheRT=double(stamps[1]-stamps[0])*e.period/1e6;
            cacheGPU=gpu=time.gpu;cacheWall=wall=time.wall;submissions++;
            acceptTokens(ids.data(),cachedCount);
            if(!cachedCount&&!std::equal(pathAddress.key.begin(),pathAddress.key.end(),result+76))throw std::runtime_error("Cached optical address mismatch");
            std::copy_n(result+76,8,pathAddress.key.begin());std::copy_n(result+84,2,pathAddress.chart.begin());
        }
        uint32_t remaining=std::min(count-cachedCount,probe?1u:count);
        if(remaining&&(!probe||header[2])){
            // A validated result-path hit consumes its cached output and
            // successor address; only a miss needs detector resources.
            prepareDetector();
            double splitGPU=attentionSplitGPU,splitWait=attentionSplitWait;uint64_t splitSubmissions=attentionSplitSubmissions;
            attentionSegmentEnd=0u;attentionProfileSlot=~0u;attentionProfileGaps.clear();
            auto boundary=cacheEnabled?outputBoundary():std::shared_ptr<SharedHistoryCache::Entry>{};
            if(boundary){
                prepareAttachment(boundary);
                bool sameLiveClasses=sameLiveAttentionClasses(boundary);
                if(attentionBackend==4u){resizeAttentionClasses(absolutePosition+remaining,sameLiveClasses);prepareStaticAttentionClassScene();}
                prepareAttentionClassAttachment(boundary,sameLiveClasses);
            }
            if(attentionBackend>=2u){
                prepareAttention(1u,absolutePosition+remaining-1u);
            }
            auto c=e.command();begin(c);recordRT=debug==2u||debug==3u;rtKinds.clear();
            if(recordRT)vkCmdResetQueryPool(c,rtTimestamps,0,RT_TIMESTAMP_COUNT);
            vkCmdResetQueryPool(c,e.timestamps,0,debug==2u?2u+remaining*(depth*7u+4u):2u);
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
            vkCmdFillBuffer(c,featureWork->handle,(featureGlobal()+8u)*4u,8u,0u);
            vkCmdFillBuffer(c,wordResult->handle,6148u*4u,4u,0u);sync(c);
            uint32_t recoverCursor=boundary?uint32_t(boundary->input.size())-prefixTokens:stateCursor;
            restoreTokens=currentFieldMatches()?0u:uint32_t(currentGenerated.size())-recoverCursor;
            if(stateCursor!=currentGenerated.size()){
                recoveryRecorded=true;
                restoredCheckpoint=bool(boundary)||currentFieldMatches();
                vkCmdResetQueryPool(c,rtTimestamps,RT_RECOVERY_QUERY,2u);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,RT_RECOVERY_QUERY);
                bool recorded=recordRT;recordRT=false;materializeCachedState(c,boundary);recordRT=recorded;
                capturePath(c,absolutePosition,64u,&pathAddress.key);
                vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,rtTimestamps,RT_RECOVERY_QUERY+1u);
            }
            vkCmdFillBuffer(c,control->handle,4,4,cachedCount);sync(c);
            auto startAddress=pathAddress;uint32_t firstPosition=absolutePosition;
            for(uint32_t i=0;i<remaining;i++){
                ResidentConfig cfg{0,1,0,firstPosition+i,i,1,0,capacity,depth,debug==2u?2u:0u,0,0};word(c,cfg);forward(c,cfg);
                if(cacheEnabled)capturePath(c,firstPosition+i+1u,i);
            }
            if(cacheEnabled){liveMarker.transfer(c,false);sync(c);replyTree->record(c,startAddress,firstPosition,remaining,cachedCount);}
            vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);sync(c);
            copy(c,control,32,0);copy(c,wordResult,256,32,528u*4u);copy(c,featureWork,8,288,(featureGlobal()+8u)*4u);copy(c,wordResult,4,296,6148u*4u);
            if(debug==1u)copy(c,state,MODEL_D*4u,308);
            if(cacheEnabled)copy(c,opticalPath->addresses,remaining*sizeof(pathAddress),308u+(debug==1u?MODEL_D*4u:0u));
            hostBarrier(c);
            check(vkEndCommandBuffer(c),"end resident generated path");auto time=attentionTiming(e.measure(c,1),splitGPU,splitWait);vkFreeCommandBuffers(e.device,e.pool,1,&c);
            gpu+=time.gpu;wall+=time.wall;submissions+=1u+uint32_t(attentionSplitSubmissions-splitSubmissions);
            auto* result=static_cast<uint32_t*>(readback->map);std::copy_n(result,8,header.begin());std::copy_n(result+8,64,ids.begin());std::copy_n(result+72,3,counts.begin());
            if(header[3])throw std::runtime_error("Resident decode optical error "+std::to_string(header[3]));
            if(boundary){attachedBlocks=boundary->blocks;attentionClassRestoreCopies.clear();}
            if(header[1]<cachedCount||header[1]>count)throw std::runtime_error("Invalid generated path length");
            uint32_t emitted=header[1]-cachedCount;
            skippedRows=remaining-emitted+uint32_t(header[2]==0u&&emitted>0u);
            if(!features){uint32_t computed=emitted-uint32_t(header[2]==0u&&emitted>0u);counts[0]=counts[1]=computed*depth*middle;}
            acceptTokens(ids.data()+cachedCount,emitted);stateCursor=uint32_t(currentGenerated.size());
            if(cacheEnabled){
                if(emitted)std::memcpy(&pathAddress,static_cast<char*>(readback->map)+308u+(debug==1u?MODEL_D*4u:0u)+(emitted-1u)*sizeof(pathAddress),sizeof(pathAddress));
                gpuFieldAddress=pathAddress;gpuFieldPosition=absolutePosition;gpuFieldReady=true;replyTree->commit(emitted);
            }
            if(debug==2u){
                std::vector<uint64_t> times(remaining*(depth*7u+4u));check(vkGetQueryPoolResults(e.device,e.timestamps,2,uint32_t(times.size()),times.size()*8,times.data(),8,VK_QUERY_RESULT_64_BIT),"read resident profile");
                for(uint32_t i=0;i<remaining;i++){uint32_t b=i*(depth*7u+4u);for(uint32_t l=0;l<depth;l++){for(uint32_t j=0;j<6;j++)parts[j]+=double(times[b+l*7+j+1]-times[b+l*7+j])*e.period/1e6;detectorTiming[fullLayer(l)?0u:1u]+=double(times[b+l*7+5]-times[b+l*7+4])*e.period/1e6;uint32_t at=i*depth+l;if(at<attentionProfileGaps.size())parts[0]-=attentionProfileGaps[at];}
                    parts[6]+=double(times[b+depth*7+3]-times[b+depth*7])*e.period/1e6;for(uint32_t j=0;j<3;j++)parts[7+j]+=double(times[b+depth*7+j+1]-times[b+depth*7+j])*e.period/1e6;}
            }
            if(!rtKinds.empty()){
                std::vector<uint64_t> times(rtKinds.size()*2u);check(vkGetQueryPoolResults(e.device,rtTimestamps,0,uint32_t(times.size()),times.size()*8,times.data(),8,VK_QUERY_RESULT_64_BIT),"read RT telemetry");
                for(uint32_t i=0;i<rtKinds.size();i++)rtParts[rtKinds[i]]+=double(times[i*2u+1u]-times[i*2u])*e.period/1e6;
            }
            if(recoveryRecorded){uint64_t times[2];check(vkGetQueryPoolResults(e.device,rtTimestamps,RT_RECOVERY_QUERY,2u,16,times,8,VK_QUERY_RESULT_64_BIT),"read cached state recovery");restoreGPU=double(times[1]-times[0])*e.period/1e6;}
        }
        recordRT=false;rtParts[3]+=cacheRT;counts[3]=cachedCount;counts[4]=cacheLeaf;
        bool allCached=cachedCount>0u&&cachedCount==header[1];
        GenerationPacket packet;packet.timing={gpu,wall,double(submissions),double(submissions),0,double(allCached),double(absolutePosition),double(header[7]),parts[0],parts[1],parts[2],parts[3],parts[4],parts[5],parts[6],parts[7],parts[8],parts[9]};
        packet.control=header;packet.ids=ids;packet.hits=counts;packet.rt=rtParts;packet.profile=debug==2u;packet.detectorTiming=detectorTiming;
        packet.cache={cachedCount,cacheQueries,replyTree?replyTree->size:0u,ReplyTreeCache::nodeCapacity};packet.cacheTime={cacheGPU,cacheWall};packet.recovery={restoreGPU,double(restoreTokens),double(restoredCheckpoint)};
        packet.candidates=candidates;packet.rejected=rejected;
        packet.routeTraceRows=featureTraceRows-traceStart;packet.routeReuseRows=featureReuseRows-reuseStart;
        uint64_t ignored=uint64_t(skippedRows)*depth;
        uint64_t ignoredReuse=std::min(packet.routeReuseRows,ignored);packet.routeReuseRows-=ignoredReuse;
        packet.routeTraceRows-=std::min(packet.routeTraceRows,ignored-ignoredReuse);
        if(debug==1u){auto* data=static_cast<char*>(readback->map)+308;packet.state.assign(data,data+MODEL_D*4u);}return packet;
    }
    bool generate(uint32_t count,uint32_t debug){
        GenerationPacket total;total.profile=debug==2u;uint32_t generated=0;
        do{
            auto part=generatePacket(count-generated,debug);uint32_t emitted=part.control[1];
            if(!emitted&&part.control[2])throw std::runtime_error("Optical cache made no generation progress");
            std::copy_n(part.ids.begin(),emitted,total.ids.begin()+generated);generated+=emitted;total.control=part.control;total.control[1]=generated;
            for(uint32_t i=0;i<18u;i++)if(i!=5u&&i!=6u&&i!=7u)total.timing[i]+=part.timing[i];
            total.timing[6]=part.timing[6];total.timing[7]=part.timing[7];
            for(uint32_t i=0;i<4u;i++){total.hits[i]+=part.hits[i];total.rt[i]+=part.rt[i];}
            if(part.cache[0])total.hits[4]=part.hits[4];
            total.cache[0]+=part.cache[0];total.cache[1]+=part.cache[1];total.cache[2]=part.cache[2];total.cache[3]=part.cache[3];
            for(uint32_t i=0;i<2u;i++){total.cacheTime[i]+=part.cacheTime[i];total.recovery[i]+=part.recovery[i];total.detectorTiming[i]+=part.detectorTiming[i];}
            total.recovery[2]=std::max(total.recovery[2],part.recovery[2]);total.candidates+=part.candidates;total.rejected+=part.rejected;total.routeTraceRows+=part.routeTraceRows;total.routeReuseRows+=part.routeReuseRows;total.state=std::move(part.state);
        }while(generated<count&&total.control[2]);
        total.timing[5]=double(total.cache[0]>0u&&total.cache[0]==generated);reply(total.bytes());return total.control[2]!=0u;
    }
    void streamGenerate(uint32_t count,uint32_t debug){if(!count||count>64)throw std::runtime_error("Resident stream expects 1..64 tokens");for(uint32_t i=0;i<count;i++)if(!generate(1u,debug))break;}
    void wordCheck(){
        ensureReadback(96u+MODEL_V*4u+257u*MODEL_D*4u+wordBlocks*16u);
        std::vector<float> q;input(q,MODEL_D);uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(state,q.data(),q.size()*4);e.upload(control,initial,32);setup();
        auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,2+depth*7+4);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        ResidentConfig cfg{0,1,0,0,0,0,0,capacity,depth,2,0,0};word(c,cfg);cfg.stage=5;launch(c,ray,cfg,MODEL_D/256u,256);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);sync(c);size_t bytes=32;copy(c,control,32,0);
        copy(c,wordResult,16,bytes);bytes+=16;copy(c,wordResult,4,bytes,4096*4);bytes+=4;
        copy(c,wordResult,MODEL_V*4,bytes,size_t(wordQueue)*4);bytes+=MODEL_V*4;
        copy(c,wordDecoded,256*MODEL_D*4,bytes);bytes+=256*MODEL_D*4;
        copy(c,baseChart,MODEL_D*4u,bytes);bytes+=MODEL_D*4u;
        copy(c,rays,wordBlocks*16u,bytes);bytes+=wordBlocks*16u;
        copy(c,wordResult,4u,bytes,4097u*4u);bytes+=4u;
        hostBarrier(c);check(vkEndCommandBuffer(c),"end resident all-contribution check");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        auto* stats=static_cast<uint32_t*>(readback->map);if(stats[3])throw std::runtime_error("Resident word optical error "+std::to_string(stats[3]));
        uint64_t stamps[4];check(vkGetQueryPoolResults(e.device,e.timestamps,2+depth*7,4,sizeof(stamps),stamps,8,VK_QUERY_RESULT_64_BIT),"read word timings");
        std::vector<char> output;append(output,std::vector<double>{time.gpu,time.wall,double(stamps[1]-stamps[0])*e.period/1e6,double(stamps[2]-stamps[1])*e.period/1e6,double(stamps[3]-stamps[2])*e.period/1e6});
        output.insert(output.end(),static_cast<char*>(readback->map)+32,static_cast<char*>(readback->map)+bytes);reply(output);
    }
    void resources(){refreshAttentionClassMirrors();uint64_t deviceBytes=0;for(auto& b:e.buffers)if(!b->map)deviceBytes+=b->allocationSize;
        std::vector<char> output;append(output,std::vector<uint64_t>{e.physicalAllocated,e.physicalDeviceAllocated,wordSceneBytes,words->vertices->size,uint64_t(wordTiles)*wordBlocks,wordTile,featureTile,replyTree?replyTree->sceneBytes:0u,liveMarker.bytes+(sharedHistory?sharedHistory->snapshotBytes():0u),parameters->size+matrices->size+basis->size,featureDown->size,deviceBytes,uint64_t(std::count_if(e.memoryPools.begin(),e.memoryPools.end(),[](const auto& p){return p.memory!=VK_NULL_HANDLE;})),keyGeometry->size,decoded->size,propagation->size,audit->size,partialDown->size,
            e.uploadStaging?e.uploadStaging->size:0u,readback->size,history->sceneBytes(),history->count,uint64_t(history->tops.size()),
            words->scratchBytes()+lenses->scratchBytes()+scalars->scratchBytes()+(mlpSurfaces?mlpSurfaces->scratchBytes():0u)+(features?features->scratchBytes():0u),
            hits->size,featureWork->size,workRows,uint64_t(detector==key),geometry->size+paths->size+baseChart->size,0u,sharedHistory?sharedHistory->snapshotBytes():0u,sharedHistory?sharedHistory->blockBytes():0u,sharedHistory?sharedHistory->directory.sceneBytes():0u,sharedHistory?sharedHistory->size():0u,sharedHistory?sharedHistory->evictions:0u,mlpSurfaces?mlpSurfaces->sceneBytes():0u,mlpSurfaces?mlpSurfaces->count:0u,attentionBackend,attentionBlock,attentionReceipts->size,attentionProducerDispatches,attentionMergeDispatches,attentionPlannedRays,attentionCoveredTerms,attentionReportedRecords,attentionClassRays,attentionClassBuilds,attentionClassBytes(),attentionClassScene?attentionClassScene->sceneBytes():0u,attentionClassSlots,attentionClassPeakSceneBytes,attentionClassReleases,attentionSplitSubmissions});
        append(output,attentionClassResourceDetails());reply(output);
    }
    void apertureUpdate(){float aperture;std::cin.read(reinterpret_cast<char*>(&aperture),4);if(!std::cin||!(aperture>0&&aperture<=2)||!sparseRouting)throw std::runtime_error("Sparse aperture update requires (0,2]");
        routeAperture=aperture;if(sharedHistory){sharedHistory->entries={};sharedHistory->publish();}if(opticalPath)opticalPath->invalidateHistory();prefixReady=false;stableReady=false;resetAttentionClassPending();attentionClassRestoreCopies.clear();cacheEpoch++;setup();auto c=e.command();begin(c);resetPathScores(c);vkCmdResetQueryPool(c,e.timestamps,0,2);vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0);
        ResidentConfig cfg{30,1,0,0,0,0,0,capacity,depth,0,0,0};launch(c,math,cfg,32768,(MODEL_V*4u+32767u)/32768u,1,initialSet);words->build(c,words->count,false);
        cfg.index=1;launch(c,math,cfg,32768,(depth*4u*MODEL_FF+32767u)/32768u,1,initialSet);features->build(c,features->count,false);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,1);check(vkEndCommandBuffer(c),"end route aperture update");auto time=e.measure(c,1);vkFreeCommandBuffers(e.device,e.pool,1,&c);
        std::vector<char> output;append(output,std::vector<double>{time.gpu,time.wall});reply(output);
    }
    void codecCheck(uint32_t count){if(!count||count%2||count>std::min(capacity,32u)*MODEL_D)throw std::runtime_error("Invalid coordinate codec input");std::vector<float> values;input(values,count);uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};e.upload(state,values.data(),count*4);e.upload(control,initial,32);setup();auto c=e.command();begin(c);resetPathScores(c);ResidentConfig cfg{13,count,0,0,0,0,0,capacity,depth,0,0,0};launch(c,math,cfg,(count/2+127)/128);copy(c,detector,count*4,0);hostBarrier(c);check(vkEndCommandBuffer(c),"end resident coordinate codec check");e.submit(c);vkFreeCommandBuffers(e.device,e.pool,1,&c);std::vector<char> output(static_cast<char*>(readback->map),static_cast<char*>(readback->map)+count*4);reply(output);}
    void mlpCheck(uint32_t count,uint32_t layer){
        if(!count||count>std::min({capacity,workRows,32u})||layer>=depth)throw std::runtime_error("Invalid optical MLP input");
        std::vector<float> values;input(values,size_t(count)*MODEL_D);
        if(!std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Optical MLP input must be finite");
        size_t bytes=size_t(count)*(MODEL_D+middle)*4u+count*4u;
        ensureReadback(bytes+4u);
        uint32_t initial[8]{0,0,1,0,0xffffffffu,0xffffffffu,0,0};
        e.upload(normalized,values.data(),values.size()*4u);e.upload(control,initial,32);setup();
        auto c=e.command();begin(c);vkCmdResetQueryPool(c,e.timestamps,0,5u);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,e.timestamps,0u);
        vkCmdFillBuffer(c,state->handle,0,size_t(count)*MODEL_D*4u,0u);sync(c);
        ResidentConfig cfg{2,count,layer,0,0,0,3,capacity,depth,4,0,0};executeMLP(c,cfg);
        vkCmdWriteTimestamp(c,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,e.timestamps,4u);
        copy(c,state,size_t(count)*MODEL_D*4u,0);
        for(uint32_t r=0;r<count;r++){
            copy(c,feedForward,size_t(middle)*4u,size_t(count)*MODEL_D*4u+size_t(r)*middle*4u,size_t(r)*MODEL_W*4u);
            copy(c,featureWork,4u,size_t(count)*(MODEL_D+middle)*4u+r*4u,size_t(r)*(16u+middle*(sparseRouting?2u:1u))*4u);
        }
        copy(c,control,4u,bytes,12u);hostBarrier(c);
        check(vkEndCommandBuffer(c),"end optical MLP check");e.submit(c);
        uint64_t times[5];check(vkGetQueryPoolResults(e.device,e.timestamps,0u,5u,sizeof(times),times,8u,VK_QUERY_RESULT_64_BIT),"MLP profile");
        vkFreeCommandBuffers(e.device,e.pool,1,&c);
        uint32_t error=static_cast<uint32_t*>(readback->map)[bytes/4u];
        if(error)throw std::runtime_error("RT MLP transmission error "+std::to_string(error));
        std::vector<char> output(static_cast<char*>(readback->map),static_cast<char*>(readback->map)+bytes);
        if(!sparseRouting)append(output,std::vector<double>{double(times[1]-times[0])*e.period/1e6,double(times[2]-times[1])*e.period/1e6,double(times[3]-times[2])*e.period/1e6});
        reply(output);
    }
    #include "resident_attention_runtime.inl"
    ~Resident(){
        vkDeviceWaitIdle(e.device);if(rtTimestamps)vkDestroyQueryPool(e.device,rtTimestamps,nullptr);if(attentionSubmitTimestamps)vkDestroyQueryPool(e.device,attentionSubmitTimestamps,nullptr);
        std::set<VkPipeline> handles;auto add=[&](const auto& pipes){for(auto p:pipes)if(p)handles.insert(p);};
        add(math);add(ray);add(ownedCooperative);add(attentionReceiptPipelines);add(attentionClassPipelines);add(detectorPipelines);for(auto& pair:opticalMlp)add(pair);
        for(auto& pair:matrixProjection)add(pair);for(auto& pair:quantOutput)add(pair);
        for(auto& pair:outputPipelines)add(pair);
        for(auto p:{matrixPack,matrixAttention,fastState})if(p)handles.insert(p);
        for(auto p:handles)vkDestroyPipeline(e.device,p,nullptr);
        vkDestroyDescriptorPool(e.device,descriptorPool,nullptr);vkDestroyPipelineLayout(e.device,layout,nullptr);vkDestroyDescriptorSetLayout(e.device,setLayout,nullptr);
    }
};

int main(int argc,char** argv){
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    try{if(argc<3)throw std::runtime_error("Usage: rt_optical_resident student.bin original_head.bin [capacity]");uint32_t capacity=argc>3?uint32_t(std::stoul(argv[3])):300;auto root=std::filesystem::absolute(argv[0]).parent_path();Engine engine;engine.initialize(root,capacity);bool sparse=argc>5&&std::string(argv[5])=="sparse";float aperture=argc>6?std::stof(argv[6]):.08f;if(!(aperture>0&&aperture<=2))throw std::runtime_error("Route aperture must be in (0,2]");uint32_t wordTile=argc>7?uint32_t(std::stoul(argv[7])):32;if(wordTile<8||wordTile>256||(wordTile&(wordTile-1u)))throw std::runtime_error("Word tile requires a dividing power of two in [8,256]");uint32_t featureTile=argc>8?uint32_t(std::stoul(argv[8])):8;if(featureTile<8||featureTile>256||(featureTile&(featureTile-1u)))throw std::runtime_error("Feature tile requires a power of two in [8,256]");uint32_t readoutUnroll=argc>9?uint32_t(std::stoul(argv[9])):1u,readoutWords=argc>10?uint32_t(std::stoul(argv[10])):4u;if((readoutUnroll!=1u&&readoutUnroll!=2u&&readoutUnroll!=4u&&readoutUnroll!=8u)||(readoutWords!=4u&&readoutWords!=8u))throw std::runtime_error("Readout requires unroll 1/2/4/8 and words per group 4/8");uint32_t prefillBatch=argc>11?uint32_t(std::stoul(argv[11])):512u;if(prefillBatch<32u||prefillBatch>1024u||(prefillBatch&(prefillBatch-1u)))throw std::runtime_error("Prefill batch requires a power of two in [32,1024]");uint32_t matrixRows=argc>12?uint32_t(std::stoul(argv[12])):8u;if(matrixRows!=4u&&matrixRows!=8u&&matrixRows!=16u)throw std::runtime_error("Matrix row tile requires 4, 8 or 16 tokens");Resident model(engine,argv[1],argv[2],argc<=4||std::stoul(argv[4])!=0,sparse,aperture,wordTile,argc>5&&std::string(argv[5])=="rt_full",featureTile,readoutUnroll,readoutWords,prefillBatch,matrixRows);uint32_t hello[5]{0x5254523a,model.depth,model.blocks,model.middle,model.quantization|(model.wideCooperative?256u:0u)|(model.attentionBackend>=2u?512u:0u)|((model.attentionBackend==2u||model.attentionBackend==4u||model.attentionBackend==5u)?2048u:0u)|(model.attentionBackend==1u?1024u:0u)|(!model.rtTransmissionMLP?4096u:0u)};hello[4]|=(model.detectorOrdered?131072u:0u);std::cout.write(reinterpret_cast<char*>(hello),20);std::cout.flush();uint32_t op;
        while(std::cin.read(reinterpret_cast<char*>(&op),4)){if(!op)break;uint32_t count,debug;if(!std::cin.read(reinterpret_cast<char*>(&count),4)||!std::cin.read(reinterpret_cast<char*>(&debug),4))throw std::runtime_error("Incomplete resident request");if(op==4||op==5||op==6||op==14){model.prefixReady=false;model.stableReady=false;model.gpuFieldReady=false;if(model.opticalPath)model.opticalPath->invalidateHistory();model.resetAttentionClassPending();model.attentionClassRestoreCopies.clear();}if(op==1||op==11||op==12)model.prefill(count,debug,op==12?2u:op==11?1u:0u);else if(op==2)model.generate(count,debug);else if(op==4)model.wordCheck();else if(op==5)model.codecCheck(count);else if(op==6)model.mlpCheck(count,debug);else if(op==7)model.apertureUpdate();else if(op==8)model.resources();else if(op==9)model.streamGenerate(count,debug);else if(op==10)model.growTree(count);else if(op==13)model.finishHistory();else if(op==14)model.attentionCheck(count,debug);else if(op==15)model.attentionStatistics(count);else throw std::runtime_error("Unknown resident operation");}return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
