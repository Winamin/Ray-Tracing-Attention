#pragma once
// Vulkan extension ABI for headers preceding KHR cooperative matrices.
#ifndef VK_KHR_cooperative_matrix
struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    VkStructureType sType;void* pNext;VkBool32 cooperativeMatrix,cooperativeMatrixRobustBufferAccess;
};
#endif
#ifndef VK_NV_cooperative_matrix_decode_vector
struct VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV {
    VkStructureType sType;void* pNext;VkBool32 cooperativeMatrixDecodeVector;
};
#endif
#ifndef VK_NV_cooperative_matrix2
struct VkPhysicalDeviceCooperativeMatrix2FeaturesNV {
    VkStructureType sType;void* pNext;
    VkBool32 cooperativeMatrixWorkgroupScope,cooperativeMatrixFlexibleDimensions,
             cooperativeMatrixReductions,cooperativeMatrixConversions,
             cooperativeMatrixPerElementOperations,cooperativeMatrixTensorAddressing,cooperativeMatrixBlockLoads;
};
#endif
