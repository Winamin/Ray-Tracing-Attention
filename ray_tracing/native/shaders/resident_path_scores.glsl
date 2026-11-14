// A complete target Q owns a persistent score plane for this attention layer.
// RT validates one occurrence stamp for all heads; score consumers read the
// plane directly. Binding 42 remains the detector's normalization scratch.
layout(std430,binding=40) buffer PathScoreCache {uint pathScoreCache[];};
layout(std430,binding=41) coherent buffer PathScoreWork {uint pathScoreWork[];};
const uint PATH_SCORE_MAGIC=0x52545053u,PATH_SCORE_WORK_FLAGS=16u;
shared uint pathTargetMismatch,pathTargetBank,pathTargetGeneration;
bool pathTraversalSame=false;
uint pathTraversalStampBase=0u,pathTraversalGeneration=0u,pathTraversalHistoryBase=0u;
uint pathTraversalProducer=0u;

bool pathScoresEnabled(){
    return cfg.rows==1u&&cfg.mode==1u&&(cfg.debug&131072u)==0u&&
        uint(pathScoreCache.length())>=8u&&pathScoreCache[4u]==PATH_SCORE_MAGIC&&
        pathScoreCache[0u]==cfg.capacity&&pathScoreCache[2u]==headCount()&&
        pathScoreCache[3u]==headWidth();
}
uint pathQWords(){return pathScoreCache[2u]*pathScoreCache[3u];}
uint pathBankWords(){return 4u+pathQWords()+cfg.capacity*(2u+pathScoreCache[2u]);}
uint pathBankBase(uint bank){return 8u+bank*pathBankWords();}
uint pathLocalSlot(uint global){return global-parameters[blockBase()+41u]*cfg.capacity;}
uint pathStamp(uint global){return pathScoreWork[0u]+4u+pathQWords()+2u*pathLocalSlot(global);}
uint pathScorePlane(uint head){return pathScoreWork[0u]+4u+pathQWords()+2u*cfg.capacity+head*cfg.capacity;}
uint pathMissIndex(uint index){return pathScoreWork[PATH_SCORE_WORK_FLAGS+cfg.capacity+index];}
bool pathTargetReady(uint head){
    return pathScoresEnabled()&&head<headCount()&&pathScoreWork[2u]==1u;
}
float readPathScore(uint head,uint global){
    return uintBitsToFloat(pathScoreCache[pathScorePlane(head)+pathLocalSlot(global)]);
}

// Cache the target namespace once per ray/group; candidate visits use only
// this invocation's validated directory and occurrence stamp.
void beginPathScoreTraversal(uint row){
    pathTraversalSame=row==0u&&pathTargetReady(0u)&&pathScoreWork[3u]==1u;
    if(pathTraversalSame){
        pathTraversalStampBase=pathScoreWork[0u]+4u+pathQWords();
        pathTraversalGeneration=pathScoreWork[1u];
        pathTraversalHistoryBase=parameters[blockBase()+41u]*cfg.capacity;
        pathTraversalProducer=pathScoreWork[6u];
    }
}

// A single workgroup compares the complete FP32 target once. Invalid banks
// have no Q payload to inspect, and a new target never inspects slot history.
void preparePathTarget(uint head,uint lane){
    if(head!=0u)return;
    if(lane==0u)pathScoreWork[2u]=0u;
    if(!pathScoresEnabled())return;
    uint width=pathQWords(),pages=pathScoreCache[1u];
    if(lane==0u){
        pathTargetBank=0xffffffffu;pathScoreWork[3u]=0u;
        pathScoreWork[4u]=0u;pathScoreWork[5u]=0u;
        // Native selects the actual producer before target preparation:
        // legacy=0, ordered=1, local=2.
        pathScoreWork[6u]=cfg.index;
        pathScoreWork[8u]=0u;pathScoreWork[9u]=1u;pathScoreWork[10u]=1u;
    }
    barrier();
    for(uint bank=0u;bank<pages;bank++){
        uint base=pathBankBase(bank);bool valid=pathScoreCache[base+1u]==1u;
        if(lane==0u)pathTargetMismatch=valid?0u:1u;
        barrier();
        uint mismatch=0u;
        if(valid)for(uint axis=lane;axis<width;axis+=gl_WorkGroupSize.x){
            float q=queries[axis];
            if(isnan(q)||isinf(q)||floatBitsToUint(q)!=pathScoreCache[base+4u+axis])mismatch=1u;
        }
        if(mismatch!=0u)atomicOr(pathTargetMismatch,mismatch);
        barrier();
        bool matched=pathTargetMismatch==0u;
        // Capture the verdict in every warp before the following bank resets it.
        barrier();
        if(matched){if(lane==0u)pathTargetBank=bank;barrier();break;}
    }
    bool fresh=pathTargetBank==0xffffffffu;
    barrier();
    if(lane==0u){
        if(fresh){
            pathTargetBank=pathScoreCache[6u]%pages;
            pathScoreCache[6u]=(pathTargetBank+1u)%pages;
        }
        uint base=pathBankBase(pathTargetBank);
        pathTargetGeneration=pathScoreCache[base]+(fresh?1u:0u);
        pathTargetMismatch=0u;
        if(fresh)pathScoreCache[base+1u]=0u;
    }
    barrier();
    uint base=pathBankBase(pathTargetBank);
    if(fresh&&pathTargetGeneration==0u){
        for(uint slot=lane;slot<cfg.capacity;slot+=gl_WorkGroupSize.x)
            pathScoreCache[base+4u+width+2u*slot]=0u;
        barrier();if(lane==0u)pathTargetGeneration=1u;barrier();
    }
    uint mismatch=0u;
    if(fresh)for(uint axis=lane;axis<width;axis+=gl_WorkGroupSize.x){
        float q=queries[axis];
        if(isnan(q)||isinf(q))mismatch=1u;
        pathScoreCache[base+4u+axis]=floatBitsToUint(q);
    }
    if(mismatch!=0u)atomicOr(pathTargetMismatch,mismatch);
    barrier();
    if(lane==0u){
        bool ready=pathTargetMismatch==0u;
        pathScoreCache[base]=pathTargetGeneration;pathScoreCache[base+1u]=ready?1u:0u;
        pathScoreWork[0u]=base;pathScoreWork[1u]=pathTargetGeneration;
        pathScoreWork[2u]=ready?1u:0u;pathScoreWork[3u]=ready&&!fresh?1u:0u;
    }
}

// Traversal already verified this occurrence and its causal coverage. A new Q
// enters the original score path immediately, with no flag or history access.
bool publishPathScores(uint row,uint global){
    if(row!=0u||!pathTraversalSame)return false;
    uint slot=global-pathTraversalHistoryBase;
    uint source=slots[global].y>>2u,at=pathTraversalStampBase+2u*slot;
    bool found=source!=0u&&pathScoreCache[at]==pathTraversalGeneration&&
        pathScoreCache[at+1u]==source;
    // Only fused legacy QK needs a per-slot verdict. Local K already owns a
    // shared miss list; ordered and legacy stamp commits use the global queue.
    if(pathTraversalProducer==0u)pathScoreWork[PATH_SCORE_WORK_FLAGS+slot]=found?1u:0u;
    if(!found&&pathTraversalProducer!=2u){
        uint index=atomicAdd(pathScoreWork[4u],1u);
        if(index<cfg.capacity)pathScoreWork[PATH_SCORE_WORK_FLAGS+cfg.capacity+index]=absolute(0u)-slots[global].x;
        else atomicOr(control[3],65536u);
    }
    return found;
}
bool reusePathScore(uint row,uint head,uint global,out float score){
    score=0.0;
    if(row!=0u||!pathTargetReady(head)||pathScoreWork[3u]!=1u||
       pathScoreWork[6u]!=0u||
       pathScoreWork[PATH_SCORE_WORK_FLAGS+pathLocalSlot(global)]!=1u)return false;
    score=readPathScore(head,global);return true;
}
void savePathScore(uint row,uint head,uint global,float score){
    if(row!=0u||!pathTargetReady(head))return;
    if(isnan(score)||isinf(score)){atomicOr(control[3],65536u);return;}
    pathScoreCache[pathScorePlane(head)+pathLocalSlot(global)]=floatBitsToUint(score);
}

// Call after the existing score-dispatch barrier has completed every head.
// This publishes one stamp, with no per-score fence or metadata rewrite.
void publishPathScoreStamp(uint global){
    if(!pathTargetReady(0u)||control[3]!=0u||global>=uint(slots.length()))return;
    uint source=slots[global].y>>2u,at=pathStamp(global);
    // Sources are monotonic and survive immutable history restoration, so the
    // source identifies the occurrence without repeating its validated age.
    pathScoreCache[at+1u]=source;
    pathScoreCache[at]=source!=0u?pathScoreWork[1u]:0u;
}

// The compact miss queue is produced by RT, then an existing dispatch barrier
// publishes it before this single-lane indirect-command preparation.
void preparePathScoreDispatch(){
    if(gl_LocalInvocationID.x!=0u||uint(pathScoreWork.length())<11u)return;
    uint visible=min(absolute(0u)+1u,cfg.capacity);
    uint perKV=uint(RT_QUERY_WIDTH)/uint(RT_GQA_WIDTH);
    uint kvHeads=(headCount()+perKV-1u)/perKV;
    uint groups=pathTargetReady(0u)&&pathScoreWork[3u]==1u?
        (pathScoreWork[4u]+3u)/4u:(visible+255u)/256u;
    // Failed coverage or an invalid score must never dispatch queue readers.
    uint total=control[3]==0u?groups*kvHeads:0u;
    pathScoreWork[5u]=total;pathScoreWork[8u]=min(total,65535u);
    pathScoreWork[9u]=total==0u?1u:(total+65534u)/65535u;pathScoreWork[10u]=1u;
}
