#pragma once

inline uint32_t attentionBackendCode(const char* name){
    std::string value=name?name:"standard";
    if(value=="standard")return 0u;
    if(value=="standard_fp32")return 1u;
    if(value=="grouped_rt")return 2u;
    if(value=="grouped_rt_dense")return 3u;
    if(value=="grouped_rt_local")return 4u;
    if(value=="grouped_rt_tc")return 5u;
    throw std::runtime_error("Attention backend must be standard, standard_fp32, grouped_rt, grouped_rt_dense, grouped_rt_local or grouped_rt_tc");
}

// A receipt covers a complete age interval for one query head. The history
// below a diagnostic's explicit origin is masked; production uses origin zero.
struct AttentionReceiptLayout {
    uint32_t block;
    bool compact=true;
    bool scoreCache=false;
    uint32_t stride() const { return scoreCache?8u+block/32u+2u*block:compact?RT_ATTENTION_HEAD_DIM+16u+block/32u+(RT_ATTENTION_HEAD_DIM+3u)/4u+(RT_ATTENTION_HEAD_DIM+31u)/32u:RT_ATTENTION_HEAD_DIM+8u+block/32u; }
    uint32_t groups(uint32_t position,uint32_t capacity,uint32_t origin=0u) const {
        uint32_t active=std::min(position-origin+1u,capacity);
        return (active+block-1u)/block;
    }
    uint64_t records(uint32_t rows,uint32_t position,uint32_t capacity,uint32_t origin=0u,bool fixed=false) const {
        uint64_t result=0;
        for(uint32_t row=0;row<rows;row++)result+=groups(position+(fixed?0u:row),capacity,origin);
        return result*RT_ATTENTION_HEADS;
    }
};
