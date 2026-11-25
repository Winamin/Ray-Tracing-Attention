const uint RT_SIGN = 0x80000000u;
const uint RT_MAGNITUDE = 0x7fffffffu;
const uint RT_INFINITY = 0x7f800000u;

struct RTBinaryOperand {
    float significand;
    int exponent;
};

bool rtFinite(float v) { return (floatBitsToUint(v) & RT_MAGNITUDE) < RT_INFINITY; }

float rtInvalid(uint flag) {
    atomicOr(arithmeticStatus[0], flag);
    return uintBitsToFloat(0x7fc00000u);
}

float rtBad(uint flag, float a, float b) {
    return rtInvalid(flag);
}

float rtNegate(float value) {
    return uintBitsToFloat(floatBitsToUint(value) ^ RT_SIGN);
}

float rtMagnitude(float value) {
    return uintBitsToFloat(floatBitsToUint(value) & RT_MAGNITUDE);
}

RTBinaryOperand rtEncode(float value) {
    uint bits = floatBitsToUint(value) & RT_MAGNITUDE;
    uint encodedExponent = bits >> 23u;
    uint fraction = bits & 0x7fffffu;
    if (bits == 0u) return RTBinaryOperand(0.0, 0);
    if (encodedExponent == 0u) {
        int leading = findMSB(fraction);
        fraction = (fraction << uint(23 - leading)) & 0x7fffffu;
        return RTBinaryOperand(uintBitsToFloat(0x3f800000u | fraction), leading - 149);
    }
    return RTBinaryOperand(uintBitsToFloat(0x3f800000u | fraction), int(encodedExponent) - 127);
}

float rtScale(float value, int shift) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & RT_SIGN;
    uint magnitude = bits & RT_MAGNITUDE;
    if (magnitude == 0u || magnitude >= RT_INFINITY) return value;
    RTBinaryOperand operand = rtEncode(value);
    uint fraction = floatBitsToUint(operand.significand) & 0x7fffffu;
    int exponent = operand.exponent + shift;
    if (exponent > 127) return uintBitsToFloat(sign | RT_INFINITY);
    if (exponent >= -126) {
        return uintBitsToFloat(sign | (uint(exponent + 127) << 23u) | fraction);
    }
    uint right = uint(-126 - exponent);
    if (right > 24u) return uintBitsToFloat(sign);
    uint significand = fraction | 0x800000u;
    uint rounded = significand >> right;
    uint remainder = significand & ((1u << right) - 1u);
    uint halfway = 1u << (right - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u)) rounded++;
    return uintBitsToFloat(sign | rounded);
}

float rtIntersectionHit(uint operatorId, vec3 origin, vec3 direction, out uint primitive) {
    primitive = 0u;
#ifdef RT_GEOMETRY_LOCAL_COUNTER
    rtLocalQueries++;
#else
    uint old = atomicAdd(arithmeticStatus[1], 1u);
    if (old == 0xffffffffu) atomicAdd(arithmeticStatus[2], 1u);
#endif
    rayQueryEXT query;
    uint flags = gl_RayFlagsOpaqueEXT;
    if (operatorId != 3u) flags |= gl_RayFlagsTerminateOnFirstHitEXT;
    rayQueryInitializeEXT(query, arithmeticAS,
        flags,
        1u << operatorId, origin, 0.0, direction, 8.0);
    while (rayQueryProceedEXT(query)) {}
    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        atomicOr(arithmeticStatus[0], 128u << operatorId);
        return rtInvalid(1u);
    }
    uint instance=rayQueryGetIntersectionInstanceCustomIndexEXT(query,true);
    primitive=rayQueryGetIntersectionPrimitiveIndexEXT(query,true);
    if(operatorId==4u) {
        if((instance>>16u)!=4u||primitive>=512u)return rtInvalid(2u);
        primitive=(instance&65535u)+primitive/2u;
    } else if(instance!=operatorId)return rtInvalid(2u);
    float distance = rayQueryGetIntersectionTEXT(query, true);
    uint bits = floatBitsToUint(distance);
    if ((bits & RT_SIGN) != 0u || (bits & RT_MAGNITUDE) >= RT_INFINITY)
        return rtInvalid(4u);
    return distance;
}

float rtIntersection(uint operatorId, vec3 origin, vec3 direction) {
    uint primitive;
    return rtIntersectionHit(operatorId, origin, direction, primitive);
}

float rtAdd(float left, float right) {
    uint leftBits = floatBitsToUint(left), rightBits = floatBitsToUint(right);
    uint leftMagnitude = leftBits & RT_MAGNITUDE, rightMagnitude = rightBits & RT_MAGNITUDE;
    if (leftMagnitude >= RT_INFINITY || rightMagnitude >= RT_INFINITY) return rtBad(8u|(1u<<20),left,right);
    if (leftMagnitude == 0u) return right;
    if (rightMagnitude == 0u) return left;
    RTBinaryOperand a = rtEncode(left), b = rtEncode(right);
    int commonExponent = max(a.exponent, b.exponent);
    float alignedA = rtScale(a.significand, a.exponent - commonExponent);
    float alignedB = rtScale(b.significand, b.exponent - commonExponent);
    uint sign;
    float distance;
    if (((leftBits ^ rightBits) & RT_SIGN) == 0u) {
        distance = rtIntersection(0u, vec3(0.0, alignedB, rtNegate(alignedA)), vec3(0.0, 0.0, 1.0));
        if(!rtFinite(distance))atomicOr(arithmeticStatus[0],1u<<27);
        sign = leftBits & RT_SIGN;
    } else {
        if (leftMagnitude == rightMagnitude) return 0.0;
        if (alignedA == alignedB) return 0.0;
        bool leftLarger = leftMagnitude > rightMagnitude;
        float larger = leftLarger ? alignedA : alignedB;
        float smaller = leftLarger ? alignedB : alignedA;
        if (!rtFinite(larger) || !rtFinite(smaller)) return 0.0;
        if (!(larger > smaller)) return 0.0;
        uint largerBits = floatBitsToUint(larger), smallerBits = floatBitsToUint(smaller);
        if (((largerBits ^ smallerBits) & 0x7f800000u) == 0u) {
            uint delta = largerBits - smallerBits;
            int shift = int((largerBits >> 23u) & 0xffu) - 127 - 23 + commonExponent;
            return uintBitsToFloat(floatBitsToUint(rtScale(float(delta), shift)) | ((leftLarger ? leftBits : rightBits) & RT_SIGN));
        }
        distance = rtIntersection(0u, vec3(0.0, larger, smaller), vec3(0.0, 0.0, 1.0));
        if(!rtFinite(distance)) {
            atomicOr(arithmeticStatus[0],1u<<28);
        }
        sign = (leftLarger ? leftBits : rightBits) & RT_SIGN;
    }
    return uintBitsToFloat(floatBitsToUint(rtScale(distance, commonExponent)) | sign);
}

float rtSubtract(float left, float right) {
    return rtAdd(left, rtNegate(right));
}

uint rtMaximumChoice(float left, float right) {
    uint leftBits = floatBitsToUint(left), rightBits = floatBitsToUint(right);
    if (leftBits == 0xff800000u) return 1u;
    if (rightBits == 0xff800000u) return 0u;
    uint leftMagnitude = leftBits & RT_MAGNITUDE, rightMagnitude = rightBits & RT_MAGNITUDE;
    if (leftMagnitude >= RT_INFINITY || rightMagnitude >= RT_INFINITY) { rtInvalid(8u|(2u<<20)); return 0u; }
    if (leftBits == rightBits) return 0u;
    if (((leftBits ^ rightBits) & RT_SIGN) != 0u)
        return (leftBits & RT_SIGN) == 0u ? 0u : 1u;
    bool negative = (leftBits & RT_SIGN) != 0u;
    if (leftMagnitude == 0u) return negative ? 0u : 1u;
    if (rightMagnitude == 0u) return negative ? 1u : 0u;
    RTBinaryOperand a = rtEncode(left), b = rtEncode(right);
    int commonExponent = max(a.exponent, b.exponent);
    float alignedA = rtScale(a.significand, a.exponent - commonExponent);
    float alignedB = rtScale(b.significand, b.exponent - commonExponent);
    uint primitive;
    rtIntersectionHit(3u, vec3(alignedA, alignedB, negative ? 0.0 : 4.0),
                      vec3(0.0, 0.0, negative ? 1.0 : -1.0), primitive);
    return primitive < 2u ? 0u : 1u;
}

float rtMaximum(float left, float right) {
    return rtMaximumChoice(left, right) == 0u ? left : right;
}

struct RTPreparedMultiplier { float inverse; int exponent; uint bits; };

RTPreparedMultiplier rtPrepareMultiplier(float value) {
    uint bits = floatBitsToUint(value), magnitude = bits & RT_MAGNITUDE;
    if (magnitude == 0u) return RTPreparedMultiplier(0.0, 0, bits);
    if (magnitude >= RT_INFINITY) return RTPreparedMultiplier(rtInvalid(8u|(3u<<20)), 0, bits);
    RTBinaryOperand operand = rtEncode(value);
    float inverse = rtIntersection(1u, vec3(0.0), vec3(0.0, 0.0, operand.significand));
    if(!rtFinite(inverse))atomicOr(arithmeticStatus[0],1u<<31);
    return RTPreparedMultiplier(inverse, operand.exponent, bits);
}

float rtMultiplyPrepared(float value, RTPreparedMultiplier weight) {
    uint bits = floatBitsToUint(value), magnitude = bits & RT_MAGNITUDE;
    uint sign = (bits ^ weight.bits) & RT_SIGN;
    if (magnitude >= RT_INFINITY || (weight.bits & RT_MAGNITUDE) >= RT_INFINITY || (floatBitsToUint(weight.inverse) & RT_MAGNITUDE) >= RT_INFINITY) return rtInvalid(8u|(4u<<20));
    if (magnitude == 0u || (weight.bits & RT_MAGNITUDE) == 0u) return uintBitsToFloat(sign);
    RTBinaryOperand operand = rtEncode(value);
    float product = rtIntersection(0u, vec3(0.0, operand.significand, 0.0), vec3(0.0, 0.0, weight.inverse));
    if(!rtFinite(product))atomicOr(arithmeticStatus[0],1u<<29);
    return uintBitsToFloat(floatBitsToUint(rtScale(product, operand.exponent + weight.exponent)) | sign);
}

struct RTPreparedPair {
    uvec2 significand;
    ivec2 exponent;
    uvec2 bits;
};

float rtDivide(float numerator,float denominator);

RTPreparedPair rtPreparePair(float first,float second) {
    uvec2 bits=uvec2(floatBitsToUint(first),floatBitsToUint(second));
    uvec2 magnitude=bits&uvec2(RT_MAGNITUDE);
    if(any(greaterThanEqual(magnitude,uvec2(RT_INFINITY))))rtInvalid(8u|(5u<<20));
    RTBinaryOperand a=rtEncode(first),b=rtEncode(second);
    uvec2 significand=uvec2(floatBitsToUint(a.significand),floatBitsToUint(b.significand))&uvec2(0x7fffffu);
    if(magnitude.x!=0u)significand.x|=0x800000u;else significand.x=0u;
    if(magnitude.y!=0u)significand.y|=0x800000u;else significand.y=0u;
    return RTPreparedPair(significand,ivec2(a.exponent,b.exponent),bits);
}

const float RT_PAIR_OFFSET = 0.75;
float rtPairDigit(uvec2 digits,RTBinaryOperand a,RTBinaryOperand b,ivec2 exponent,
                  uvec2 signs,out bool used,out int pairExponent) {
    bool first=digits.x!=0u,second=digits.y!=0u;
    used=first||second;
    if(!used){pairExponent=0;return 0.0;}
    pairExponent=first&&second?max(exponent.x,exponent.y):first?exponent.x:exponent.y;
    float x=first?rtScale(a.significand,exponent.x-pairExponent-3):0.0;
    float y=second?rtScale(b.significand,exponent.y-pairExponent-3):0.0;
    if((signs.x&RT_SIGN)!=0u)x=rtNegate(x);
    if((signs.y&RT_SIGN)!=0u)y=rtNegate(y);
    uint code=((first?digits.x:0u)<<8u)|(second?digits.y:0u);
    uint group=code/256u,local=code%256u,primitive;
    float distance=rtIntersectionHit(4u,vec3(float(group*4u)+x,y,float(local*8u)),vec3(0.0,0.0,1.0),primitive);
    if(primitive!=code)return rtInvalid(64u);
    if(!rtFinite(distance))atomicOr(arithmeticStatus[0],1u<<24);
    return distance;
}

float rtDotPair(float first,float second,RTPreparedPair weights) {
    uvec2 bits=uvec2(floatBitsToUint(first),floatBitsToUint(second));
    uvec2 magnitude=bits&uvec2(RT_MAGNITUDE);
    if(any(greaterThanEqual(magnitude,uvec2(RT_INFINITY)))||
       any(greaterThanEqual(weights.bits&uvec2(RT_MAGNITUDE),uvec2(RT_INFINITY))))return rtInvalid(8u|(6u<<20));
    RTBinaryOperand a=rtEncode(first),b=rtEncode(second);
    uvec2 significand=weights.significand;
    if(magnitude.x==0u)significand.x=0u;
    if(magnitude.y==0u)significand.y=0u;
    ivec2 exponent=ivec2(a.exponent,b.exponent)+weights.exponent;
    uvec2 signs=bits^weights.bits;
    bool used;
    int pairExponent;
    uvec2 digits=min((significand+uvec2(0x8000u))>>16u,uvec2(255u));
    float distance=rtPairDigit(digits,a,b,exponent,signs,used,pairExponent);
    if(!used)return 0.0;
    int shift=pairExponent+4;
    return rtScale(distance,shift)-rtScale(RT_PAIR_OFFSET,shift);
}

float rtMultiply(float left, float right) {
    uint leftBits = floatBitsToUint(left), rightBits = floatBitsToUint(right);
    uint aMagnitude = leftBits & RT_MAGNITUDE, bMagnitude = rightBits & RT_MAGNITUDE;
    if (aMagnitude >= RT_INFINITY || bMagnitude >= RT_INFINITY) return rtInvalid(8u|(7u<<20));
    if (aMagnitude == 0u || bMagnitude == 0u) return uintBitsToFloat((leftBits ^ rightBits) & RT_SIGN);
    RTBinaryOperand a = rtEncode(left), b = rtEncode(right);
    float inverse = rtIntersection(1u, vec3(0.0), vec3(0.0, 0.0, b.significand));
    if(!rtFinite(inverse))atomicOr(arithmeticStatus[0],1u<<31);
    if ((floatBitsToUint(inverse) & RT_MAGNITUDE) >= RT_INFINITY) return inverse;
    float product = rtIntersection(0u, vec3(0.0, a.significand, 0.0), vec3(0.0, 0.0, inverse));
    uint sign = (leftBits ^ rightBits) & RT_SIGN;
    return uintBitsToFloat(floatBitsToUint(rtScale(product, a.exponent + b.exponent)) | sign);
}

float rtDivide(float numerator, float denominator) {
    uint aBits = floatBitsToUint(numerator), bBits = floatBitsToUint(denominator);
    uint aMagnitude = aBits & RT_MAGNITUDE, bMagnitude = bBits & RT_MAGNITUDE;
    if (aMagnitude >= RT_INFINITY || bMagnitude >= RT_INFINITY || bMagnitude == 0u) return rtBad(8u|(8u<<20),numerator,denominator);
    uint sign = (aBits ^ bBits) & RT_SIGN;
    if (aMagnitude == 0u) return uintBitsToFloat(sign);
    RTBinaryOperand a = rtEncode(numerator), b = rtEncode(denominator);
    float quotient = rtIntersection(0u, vec3(0.0, a.significand, 0.0), vec3(0.0, 0.0, b.significand));
    if(!rtFinite(quotient))atomicOr(arithmeticStatus[0],1u<<30);
    return uintBitsToFloat(floatBitsToUint(rtScale(quotient, a.exponent - b.exponent)) | sign);
}

float rtExp(float value) {
    uint bits = floatBitsToUint(value), magnitude = bits & RT_MAGNITUDE;
    bool negative = (bits & RT_SIGN) != 0u;
    if (magnitude > RT_INFINITY) return rtInvalid(8u|(9u<<20));
    if (magnitude == RT_INFINITY) return negative ? 0.0 : value;
    if (magnitude == 0u) return 1.0;
    if (negative && magnitude >= floatBitsToUint(103.97208404541016)) return 0.0;
    if (!negative && magnitude >= floatBitsToUint(88.72283935546875)) return uintBitsToFloat(RT_INFINITY);
    float binaryArgument = rtDivide(rtMagnitude(value), 0.6931471805599453);
    if ((floatBitsToUint(binaryArgument) & RT_MAGNITUDE) >= RT_INFINITY) return binaryArgument;
    uint whole = uint(binaryArgument);
    float fraction = rtSubtract(binaryArgument, float(whole));
    if ((floatBitsToUint(fraction) & RT_MAGNITUDE) > 0x3f800000u) return rtInvalid(16u);
    float chartInput = negative ? rtNegate(rtMagnitude(fraction)) : rtMagnitude(fraction);
    float interpolated = rtIntersection(2u, vec3(chartInput, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    return rtScale(interpolated, negative ? -int(whole) : int(whole));
}

float rtSigmoid(float value) {
    if ((floatBitsToUint(value) & RT_SIGN) == 0u)
        return rtDivide(1.0, rtAdd(1.0, rtExp(rtNegate(value))));
    float energy = rtExp(value);
    return rtDivide(energy, rtAdd(1.0, energy));
}

float rtSwiGLU(float gate, float up) {
    return rtMultiply(rtMultiply(gate, rtSigmoid(gate)), up);
}

