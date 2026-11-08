#include "resident_dimensions.glsl"
#extension GL_EXT_ray_query : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x=128,local_size_x_id=4) in;
layout(constant_id=0) const uint operation=0u;
layout(set=0,binding=0) uniform accelerationStructureEXT historyAS[4];
layout(set=0,binding=1) uniform accelerationStructureEXT lensAS[4];
layout(std430,binding=2) buffer OpticalMemory {uint opticalMemory[];};
layout(set=0,binding=3) uniform accelerationStructureEXT wordAS[4];
layout(set=0,binding=4) uniform accelerationStructureEXT scalarAS;
layout(std430,binding=5) readonly buffer Parameters {uint parameters[];};
#define RT_COEFFICIENT_ACCESS readonly
layout(std430,binding=6) RT_COEFFICIENT_ACCESS buffer Matrices {uint values[];} matrices[RT_LAYER_COUNT];
layout(std430,binding=7) RT_COEFFICIENT_ACCESS buffer Basis {uint basis[];};
layout(std430,binding=8) buffer State {float state[];};
layout(std430,binding=9) buffer Normalized {float normalized[];};
layout(std430,binding=10) buffer Query {float queries[];};
layout(std430,binding=11) buffer Key {float keys[];};
layout(std430,binding=12) buffer KeyGeometry {uvec2 keyGeometry[];};
layout(std430,binding=13) buffer HistoryVertices {float historyVertices[];};
layout(std430,binding=14) buffer Decoded {float decoded[];};
layout(std430,binding=15) buffer Geometry {float geometry[];};
layout(std430,binding=16) buffer Slots {uvec2 slots[];};
layout(std430,binding=17) buffer Rays {vec4 rays[];};
layout(std430,binding=18) buffer Hits {uint hits[];};
layout(std430,binding=19) buffer Paths {float paths[];};
layout(std430,binding=20) buffer Detector {float detector[];};
layout(std430,binding=21) buffer Physical {float physical[];};
layout(std430,binding=22) buffer WordDecoded {float wordDecoded[];};
layout(std430,binding=23) buffer ScalarVertices {float scalarVertices[];};
layout(std430,binding=24) buffer WordResult {uint wordResult[];};
layout(std430,binding=25) buffer LensVertices {float lensVertices[];};
layout(std430,binding=26) buffer BaseChart {vec2 baseChart[];};
layout(std430,binding=27) buffer FeedForward {float feedForward[];};
layout(std430,binding=28) buffer Auxiliary {float auxiliary[];};
layout(std430,binding=29) readonly buffer Prompt {uint prompt[];};
layout(std430,binding=30) buffer Control {uint control[];};
layout(std430,binding=31) buffer Audit {float audit[];};
layout(set=0,binding=32) uniform accelerationStructureEXT featureAS[RT_LAYER_COUNT];
layout(std430,binding=33) buffer FeatureVertices {float featureVertices[];};
layout(std430,binding=34) buffer FeatureWork {uint featureWork[];};
layout(std430,binding=35) buffer FeatureRays {vec4 featureRays[];};
layout(std430,binding=36) buffer FeatureDown {uint featureDown[];};
layout(std430,binding=37) buffer DownPartial {float partialDown[];};
layout(push_constant) uniform Config {uint stage,rows,layer,position,step,mode,index,capacity,depth,debug,padding0,padding1;} cfg;
const uint D=RT_HIDDEN_SIZE,A=RT_WORKING_WIDTH,S=RT_STATE_WIDTH,Q=RT_QUERY_WIDTH,W=RT_FF_STRIDE,P=RT_QUERY_WIDTH/2u,B=32u,H=RT_STATE_HEADS,HD=128u,WP=256u;
layout(constant_id=3) const uint WORD_TILE=256u;
layout(constant_id=5) const uint FEATURE_TILE=8u;
layout(constant_id=8) const uint coefficientFormat=0xffffffffu;
const uint V=RT_VOCABULARY_SIZE,WORD_TILES=(V+WORD_TILE-1u)/WORD_TILE,WORD_MASK=8192u+V,WORD_QUEUE=8192u+2u*V;
const uint ROUTE_BANKS=4u,FEATURE_WIDTH=RT_MIDDLE_SIZE,FEATURE_ROW=16u+FEATURE_WIDTH*2u,FEATURE_TILES=(FEATURE_WIDTH+FEATURE_TILE-1u)/FEATURE_TILE;
shared float reduction[4];
float angularSign(uint bank,uint axis,uint coordinate){return (bitCount(coordinate&(1u<<(bank*2u+axis)))&1)==0?1.0:-1.0;}
uint featureBanks(){return cfg.padding0==1u?ROUTE_BANKS:1u;}
uint featureRow(){return 16u+FEATURE_WIDTH*(cfg.padding0==1u?2u:1u);}
uint featureQueueOffset(){return 16u+(cfg.padding0==1u?FEATURE_WIDTH:0u);}
uint featureBase(uint row){return row*featureRow();}
uint featureGlobal(){return min(cfg.capacity,32u)*featureRow();}
float pf(uint at){return uintBitsToFloat(parameters[at]);}
uint blockBase(){return 32u+(cfg.layer%parameters[1])*64u;}
uint absolute(uint row){return cfg.mode==0u?row:cfg.mode==1u?cfg.position:cfg.position+row;}
uint keySlot(uint row){return parameters[blockBase()+41u]*cfg.capacity+absolute(row)%cfg.capacity;}
uint scalarCacheOffset(){return 11u+cfg.capacity;}
uint chunkIndex(uint layer,uint bank,uint segment){uint n=(cfg.capacity+15u)/16u;return parameters[32u+layer*64u+41u]*n+segment;}
uint lensIndex(uint layer,uint bank){return cfg.padding0==1u?(bank/8u)*(cfg.depth+1u)*8u+layer*8u+(bank%8u):layer==cfg.depth?RT_FULL_COUNT:parameters[32u+layer*64u+41u];}
uint rayStride(){return cfg.padding0==1u?4u:1u;}
float tokenEmbedding(uint word,uint axis);
float vocabularyValue(uint word,uint axis);
float wordValue(uint word,uint axis){
return tokenEmbedding(word,axis);
}
uint wordBits(uint word,uint axis){uint at=word*D+axis;return (basis[at/2u]>>((at%2u)*16u))&65535u;}
float sum128(float value){value=subgroupAdd(value);if(gl_SubgroupInvocationID==0u)reduction[gl_SubgroupID]=value;barrier();if(gl_LocalInvocationID.x==0u)reduction[0]+=reduction[1]+reduction[2]+reduction[3];barrier();float result=reduction[0];barrier();return result;}
vec2 chart(vec2 pair,float inv){vec2 value=pair*inv;float radius=length(value);float a=log(1.0+radius);return radius>1e-20?.7*a/(1.0+a)*value/radius:vec2(0);}
vec2 rotation(vec2 pair,float phase){return vec2(pair.x*cos(phase)-pair.y*sin(phase),pair.x*sin(phase)+pair.y*cos(phase));}
float softplus(float value){return max(value,0.0)+log(1.0+exp(-abs(value)));}
bool stopped(){return cfg.mode==1u&&control[2]==0u;}
uvec4 encodeCoordinate(float value){
    uint raw=floatBitsToUint(value),sign=raw>>31u,fraction=raw&0x7fffffu,exponent=(raw>>23u)&255u;int shiftExponent;
    if((raw&0x7fffffffu)==0u)return uvec4(0,0,0,256u|(sign<<9u));
    if(exponent==255u){atomicOr(control[3],256u);return uvec4(0);}
    if(exponent==0u){int top=findMSB(fraction);fraction=(fraction<<uint(23-top))&0x7fffffu;shiftExponent=top-148;}
    else shiftExponent=int(exponent)-126;
    float mantissa=uintBitsToFloat((126u<<23u)|fraction),high=uintBitsToFloat(floatBitsToUint(mantissa)&0xffff0000u),remainder=mantissa-high;
    float middle=uintBitsToFloat(floatBitsToUint(remainder)&0xffff0000u),low=remainder-middle;
    return uvec4(floatBitsToUint(high)>>16u,floatBitsToUint(middle)>>16u,floatBitsToUint(low)>>16u,uint(shiftExponent+256)|(sign<<9u));
}
float restoreCoordinate(uvec3 reference,uint meta){
    float mantissa=uintBitsToFloat(control[scalarCacheOffset()+reference.x])+uintBitsToFloat(control[scalarCacheOffset()+reference.y])+uintBitsToFloat(control[scalarCacheOffset()+reference.z]);
    uint sign=(meta>>9u)<<31u;if(mantissa==0)return uintBitsToFloat(sign);int exponent=int(meta&511u)-256+126;uint fraction=floatBitsToUint(mantissa)&0x7fffffu;
    if(exponent>0)return uintBitsToFloat(sign|(uint(exponent)<<23u)|fraction);
    return uintBitsToFloat(sign|((fraction|0x800000u)>>uint(1-exponent)));
}
uvec4 encodePair(vec2 value){uvec4 x=encodeCoordinate(value.x),y=encodeCoordinate(value.y);return uvec4(x.x|(y.x<<16u),x.y|(y.y<<16u),x.z|(y.z<<16u),x.w|(y.w<<10u));}
vec2 restorePair(uvec4 geometry){return vec2(restoreCoordinate(geometry.xyz&uvec3(65535u),geometry.w&1023u),restoreCoordinate(geometry.xyz>>16u,geometry.w>>10u));}

// A canonical sign/exponent/mantissa packet expands into the same three
// BF16 path references; the validated inverse is a bit-exact direct decode.
uvec2 encodePathPair(vec2 value){return floatBitsToUint(value);}
vec2 restorePathPair(uvec2 packet){return uintBitsToFloat(packet);}
const uint SHARED_WIDTH=RT_GQA_WIDTH,SHARED_PAIRS=RT_GQA_WIDTH/2u;
uint sharedAxis(uint axis){return (axis/(RT_ATTENTION_HEAD_DIM*(RT_QUERY_WIDTH/RT_GQA_WIDTH)))*RT_ATTENTION_HEAD_DIM+axis%RT_ATTENTION_HEAD_DIM;}
float quantizedPath(uint slot,uint axis,bool content);
float pathCoordinate(uint slot,uint axis,bool content){
return quantizedPath(slot,axis,content);
}

uint headCount(){return parameters[blockBase()+23u];}
uint headWidth(){return parameters[blockBase()+26u]==1u?RT_ATTENTION_HEAD_DIM:HD;}
float coordinatePhase(uint pair,uint time){uint activePairs=parameters[blockBase()+29u]/2u,local=pair%(headWidth()/2u);return local<activePairs?float(time)*pow(float(RT_ROPE_THETA),-float(local)/float(activePairs)):0.0;}
vec2 opticalTransfer(uint packed){return vec2(uintBitsToFloat(packed<<16u),uintBitsToFloat(packed&0xffff0000u));}
uint measuredCoefficient(uint reference){
    uint fraction=reference&127u,exponent=(reference>>7u)&255u,sign=reference&32768u;
    if(exponent==0u&&fraction==0u)return reference;
    uint shift=exponent==0u?uint(7-findMSB(fraction)):0u;
    uint normalized=(126u<<7u)|((fraction<<shift)&127u);
    uint measured=(control[scalarCacheOffset()+normalized]>>16u)&127u;
    return exponent==0u?sign|(((128u|measured)>>shift)&127u):sign|(exponent<<7u)|measured;
}
#include "resident_quant.glsl"
vec2 geometricWeight(uint offset,uint metadata,uint channel,uint cols,uint pack){
return quantizedTransfer(offset,metadata,channel,cols,pack);
}
float wordGeometryValue(uint word,uint axis){
return vocabularyValue(word,axis);
}
vec2 wordTransfer(uint word,uint pack){
return outputTransfer(word,pack);
}
vec2 signedEnergy(vec2 beamInput,vec2 transfer,bool positive){vec2 amplitude=32.0*abs(beamInput),transmission=abs(transfer)/32.0,energy=amplitude*transmission;vec2 signPair=sign(beamInput)*sign(transfer);return energy*vec2(positive?greaterThan(signPair,vec2(0)):lessThan(signPair,vec2(0)));}
float detectorSigmoid(float value){if(value>=0)return 1.0/(1.0+exp(-value));float light=exp(value);return light/(1.0+light);}
uint recurrentOffset(uint head,uint value){return ((parameters[blockBase()+42u]*H+head)*HD+value)*HD;}
uint convolutionOffset(){return RT_RECURRENT_COUNT*H*HD*HD+parameters[blockBase()+42u]*S*9u;}
uint frameOffset(uint row){return RT_RECURRENT_COUNT*(H*HD*HD+S*9u)+row*S*3u;}
vec2 fieldPair(uint at){return restorePathPair(uvec2(opticalMemory[at],opticalMemory[at+1u]));}
void writeField(uint at,vec2 value){uvec2 packet=encodePathPair(value);opticalMemory[at]=packet.x;opticalMemory[at+1u]=packet.y;}
float frame(uint row,uint coord){return uintBitsToFloat(opticalMemory[frameOffset(row)+coord]);}
float rawProjection(uint row,uint coord){return coord<S?queries[row*A+coord]:coord<2u*S?keys[row*A+coord-S]:feedForward[row*W+S+coord-2u*S];}
