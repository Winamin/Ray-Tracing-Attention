#pragma once
#include "vocabulary_geometry.hpp"

// Fixed transmission surfaces address (receiver, 256-coefficient block).
// The original packed materials remain in the per-layer matrix descriptors.
// Gate/up share a receiver; down uses a separate receiver range. All layers
// share this immutable support, since input brightness never changes a surface.
class MLPTransmissionGeometry:public ImmutableBlockGeometry {
public:
    MLPTransmissionGeometry(Engine& engine,uint32_t hidden,uint32_t middle)
        :ImmutableBlockGeometry(engine,hidden+middle,std::max(hidden,middle)/256u) {
        uint32_t blocks=std::max(hidden,middle)/256u;
        std::vector<VkAabbPositionsKHR> bounds(count);
        for(uint32_t receiver=0;receiver<hidden+middle;receiver++) {
            float x=2.f*(receiver%256u),y=2.f*(receiver/256u);
            for(uint32_t block=0;block<blocks;block++) {
                float z=1.f+block;
                bounds[size_t(receiver)*blocks+block]={x-.5f,y-.5f,z-.0001f,x+.5f,y+.5f,z+.0001f};
            }
        }
        e.upload(vertices,bounds.data(),vertices->size);
    }
};
