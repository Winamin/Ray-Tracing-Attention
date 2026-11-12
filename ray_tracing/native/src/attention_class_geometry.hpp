#pragma once
#include "vocabulary_geometry.hpp"

// Shared by every layer, row and KV head. The primitive index is a hash bucket;
// exact equality and membership come from the current token's cached FP32 bits.
// This AS is built once and never updated, refitted or destroyed between batches.
class AttentionClassGeometry:public ImmutableBlockGeometry {
public:
    static constexpr uint32_t bucketCount=[]() constexpr {
        uint32_t buckets=1u;
        while(buckets<256u&&buckets<=uint32_t(RT_ATTENTION_HEAD_DIM)/2u)buckets*=2u;
        return buckets;
    }();
    uint64_t peakBuildBytes{};
    explicit AttentionClassGeometry(Engine& engine)
        :ImmutableBlockGeometry(engine,1u,bucketCount,true) {
        std::vector<VkAabbPositionsKHR> bounds(bucketCount);
        for(uint32_t bucket=0u;bucket<bucketCount;bucket++){
            float x=float(bucket);
            bounds[bucket]={x-.25f,-.25f,-.25f,x+.25f,.25f,.25f};
        }
        e.upload(vertices,bounds.data(),vertices->size);
        peakBuildBytes=sceneBytes();
        auto command=e.command();e.begin(command);
        ImmutableBlockGeometry::build(command,bucketCount,true);
        check(vkEndCommandBuffer(command),"end permanent class bucket scene");
        e.submit(command);vkFreeCommandBuffers(e.device,e.pool,1u,&command);
        // The completed BLAS/TLAS never need scratch again.
        releaseScratch();
    }
};
