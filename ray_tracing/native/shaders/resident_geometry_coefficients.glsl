// Original source formats remain packed. Decode scale products and affine
// minima through the same arithmetic BVH as the model's matrix products.
uint geometryHistoryWordBase=0u;
uint geometryCoefficientWord(uint kind,uint at) {
    if(kind!=2u)return coefficientWord(kind,at);
    uint index=geometryHistoryWordBase+at/4u;
    uint value=keyGeometry[index/2u][index%2u];
    if(at%4u==0u)return value;
    uint next=keyGeometry[(index+1u)/2u][(index+1u)%2u];
    return (value>>16u)|(next<<16u);
}
uint geometryCoefficientByte(uint kind,uint at) {
    return (geometryCoefficientWord(kind,at&~3u)>>((at%4u)*8u))&255u;
}
float geometryCoefficientHalf(uint kind,uint at) {
    return unpackHalf2x16(geometryCoefficientWord(kind,at&~3u)>>((at%4u)*8u)).x;
}
float geometryGGUF(uint kind, uint at, uint type, uint index) {
    if (type == 0u) return uintBitsToFloat(geometryCoefficientWord(kind, at + index * 4u));
    if (type == 1u) return geometryCoefficientHalf(kind, at + index * 2u);
    if (type == 30u) return uintBitsToFloat((geometryCoefficientWord(kind,(at+index*2u)&~3u)>>(((at+index*2u)%4u)*8u)&65535u)<<16u);
    if (type == 12u) {
        uint group = index / 32u;
        uvec2 scaleMinimum;
        if(group<4u)scaleMinimum=uvec2(geometryCoefficientByte(kind,at+4u+group),geometryCoefficientByte(kind,at+8u+group))&63u;
        else {
            uint part=group-4u,mixed=geometryCoefficientByte(kind,at+12u+part);
            scaleMinimum=uvec2((mixed&15u)|((geometryCoefficientByte(kind,at+4u+part)>>6u)<<4u),
                               (mixed>>4u)|((geometryCoefficientByte(kind,at+8u+part)>>6u)<<4u));
        }
        uint packed = geometryCoefficientByte(kind, at + 16u + (index / 64u) * 32u + index % 32u);
        uint value = (packed >> ((index % 64u) / 32u * 4u)) & 15u;
        float scale = rtMultiply(geometryCoefficientHalf(kind, at), float(scaleMinimum.x));
        float minimum = rtMultiply(geometryCoefficientHalf(kind, at + 2u), float(scaleMinimum.y));
        return rtSubtract(rtMultiply(scale, float(value)), minimum);
    }
    uint halfIndex = index / 128u, local = index % 128u;
    uint low = (geometryCoefficientByte(kind, at + halfIndex * 64u + local % 64u) >> ((local / 64u) * 4u)) & 15u;
    uint high = (geometryCoefficientByte(kind, at + 128u + halfIndex * 32u + local % 32u) >> ((local / 32u) * 2u)) & 3u;
    int scale = bitfieldExtract(int(geometryCoefficientByte(kind, at + 192u + index / 16u)), 0, 8);
    float blockScale = rtMultiply(geometryCoefficientHalf(kind, at + 208u), float(scale));
    return rtMultiply(blockScale, float(int(low | (high << 4u)) - 32));
}

float geometryWeight(uint kind, uint offset, uint metadata, uint row, uint columns, uint axis) {
    uint type = metadata & 255u, mode = metadata >> 8u;
    if (mode == 1u || mode == 2u)
        row = (row / RT_ATTENTION_HEAD_DIM) * (2u * RT_ATTENTION_HEAD_DIM)
            + (mode == 2u ? RT_ATTENTION_HEAD_DIM : 0u) + originalRotaryAxis(row % RT_ATTENTION_HEAD_DIM);
    else if (mode == 3u)
        row = (row / RT_ATTENTION_HEAD_DIM) * RT_ATTENTION_HEAD_DIM + originalRotaryAxis(row % RT_ATTENTION_HEAD_DIM);
    else if (mode == 4u)
        axis = (axis / RT_ATTENTION_HEAD_DIM) * RT_ATTENTION_HEAD_DIM + originalRotaryAxis(axis % RT_ATTENTION_HEAD_DIM);
    else if (mode == 5u) row %= RT_STATE_KEY_HEADS * 128u;
    uint linear = row * columns + axis;
    if (type != 12u && type != 14u) return geometryGGUF(kind, offset * 4u, type, linear);
    uint bytes = type == 12u ? 144u : 210u;
    return geometryGGUF(kind, offset * 4u + (linear / 256u) * bytes, type, linear % 256u);
}

// Resolve two neighbouring coefficients for one dot pair. F16 storage keeps
// consecutive elements adjacent, and an even axis makes the pair word aligned, so
// both come back from a single load instead of two loads with two address
// computations. Everything else falls back to the generic path.
vec2 geometryWeightPair(uint kind,uint offset,uint metadata,uint row,uint columns,uint axis) {
    if((metadata&0xffffff00u)==1u) {
        uint at=offset*4u+(row*columns+axis)*2u;
        return unpackHalf2x16(geometryCoefficientWord(kind,at));
    }
    return vec2(geometryWeight(kind,offset,metadata,row,columns,axis),
                geometryWeight(kind,offset,metadata,row,columns,axis+1u));
}

const uint GEOMETRY_HISTORY_BYTES = ((RT_GQA_WIDTH + 255u) / 256u * 354u + 7u) / 8u * 8u;
float geometryHistory(uint slot, uint channel, bool content) {
    uint axis = sharedAxis(channel), blocks = (RT_GQA_WIDTH + 255u) / 256u;
    geometryHistoryWordBase=slot*(GEOMETRY_HISTORY_BYTES/4u);
    uint at = (content ? blocks * 144u : 0u)
        + (axis / 256u) * (content ? 210u : 144u);
    return geometryGGUF(2u, at, content ? 14u : 12u, axis % 256u);
}
