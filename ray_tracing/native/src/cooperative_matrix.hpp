#pragma once
// Vulkan extension ABI for headers preceding KHR cooperative matrices.
#ifndef VK_KHR_cooperative_matrix
struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    VkStructureType sType;void* pNext;VkBool32 cooperativeMatrix,cooperativeMatrixRobustBufferAccess;
};
#endif
#ifndef VK_KHR_shader_bfloat16
// VK_KHR_shader_bfloat16 ABI. The feature layout and structure discriminator
// are also present in the installed VulkanInfo used by the capability probe.
struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType sType;void* pNext;
    VkBool32 shaderBFloat16Type,shaderBFloat16DotProduct,shaderBFloat16CooperativeMatrix;
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
