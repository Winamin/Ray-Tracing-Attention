#pragma once
#include "optical_geometry.hpp"

// One reusable TLAS contains scalar fields and an instanced 2D pair field. The shader sends
// operands as ray coordinates and directions; committed t carries the result.
class RTArithmeticSpace
{
public:
    static constexpr uint32_t exponentialSegments = 4096u;
    static constexpr uint32_t pairTemplates = 64u, pairGroups = 1024u, pairPlanes = 65536u;
    Engine &e;
    Buffer *vertices{}, *instances{}, *topStorage{}, *topScratch{};
    std::array<Leaf, 5> fields{};
    VkAccelerationStructureKHR top{};
    uint64_t residentBytes{};

    RTArithmeticSpace(Engine &engine) : e(engine)
    {
        std::vector<float> points;
        auto triangle = [&](const std::array<float, 3> &a, const std::array<float, 3> &b, const std::array<float, 3> &c)
        {
            for (const auto &point : {a, b, c}) points.insert(points.end(), point.begin(), point.end());
        };
        auto quad = [&](const std::array<float, 3> &a, const std::array<float, 3> &b,
                        const std::array<float, 3> &c, const std::array<float, 3> &d)
        {
            triangle(a, b, c);
            triangle(a, c, d);
        };
        // Addition, subtraction and division share the bounded plane z=y.
        // The x extent is deliberately asymmetric: every ray here has x=0, and a
        // symmetric quad puts the shared triangle diagonal at (0,1,1), which is
        // exactly where a significand of 1.0 lands. A hit on that edge is not
        // watertight and silently drops. This skew moves the diagonal to y=2.0,
        // outside the [1,2) significand range.
        quad({-3.f, -1.f, -1.f}, {1.f, -1.f, -1.f}, {1.f, 3.f, 3.f}, {-3.f, 3.f, 3.f});
        // The reciprocal field is the plane z=1.
        quad({-2.f, -2.f, 1.f}, {2.f, -2.f, 1.f}, {2.f, 2.f, 1.f}, {-2.f, 2.f, 1.f});
        // exp2 is compiled once into strips, never evaluated by a runtime
        // shader. Triangle interpolation is an approximation, not exact exp.
        for (uint32_t segment = 0; segment < exponentialSegments; segment++)
        {
            float x0 = -1.f + 2.f * float(segment) / float(exponentialSegments);
            float x1 = -1.f + 2.f * float(segment + 1u) / float(exponentialSegments);
            float z0 = float(std::exp2(double(x0))), z1 = float(std::exp2(double(x1)));
            quad({x0, -1.f, z0}, {x1, -1.f, z1}, {x1, 1.f, z1}, {x0, 1.f, z0});
        }
        // Crossing planes implement min through the native closest hit.
        // Downward rays select the larger operand through the hit identity.
        quad({-1.f, -1.f, -1.f}, {3.f, -1.f, 3.f}, {3.f, 3.f, 3.f}, {-1.f, 3.f, -1.f});
        quad({-1.f, -1.f, -1.f}, {3.f, -1.f, -1.f}, {3.f, 3.f, 3.f}, {-1.f, 3.f, 3.f});
        // Template p has z=8p+2+(p/256)y. Instance shears supply the other
        // coefficient and the upper two bits of the y coefficient. Inputs
        // travel in the direction, so instance translations never round them.
        for(uint32_t p=0;p<pairTemplates;p++)
        {
            float center=float(p*8u)+2.f,b=float(p)/256.f;
            quad({-1.f,-1.f,center-b},{1.f,-1.f,center-b},
                 {1.f,1.f,center+b},{-1.f,1.f,center+b});
        }
        const auto inputUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        vertices = e.buffer(points.size() * sizeof(float), inputUsage);
        e.upload(vertices, points.data(), vertices->size);
        std::vector<VkAccelerationStructureInstanceKHR> roots(4u+pairGroups);
        uint32_t firstVertex = 0;
        for (uint32_t id = 0; id < fields.size(); id++)
        {
            auto &field = fields[id];
            field.count = id == 2u ? exponentialSegments * 2u : id == 3u ? 4u : id == 4u ? pairTemplates*2u : 2u;
            field.geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            field.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            field.geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto &triangles = field.geometry.geometry.triangles;
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = vertices->address + uint64_t(firstVertex) * 12u;
            triangles.vertexStride = 12u;
            triangles.maxVertex = field.count * 3u - 1u;
            triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
            firstVertex += field.count * 3u;
            VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            info.geometryCount = 1u;
            info.pGeometries = &field.geometry;
            VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            e.getSizes(e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &field.count, &sizes);
            field.storage = e.buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
            field.scratch = e.buffer(sizes.buildScratchSize + e.alignment,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            create.type = info.type;
            create.buffer = field.storage->handle;
            create.size = field.storage->size;
            check(e.createAS(e.device, &create, nullptr, &field.handle), "create arithmetic triangle BLAS");
            VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            address.accelerationStructure = field.handle;
            VkDeviceAddress reference=e.asAddress(e.device,&address);
            if(id<4u)
            {
                auto &root=roots[id];
                root.transform.matrix[0][0]=root.transform.matrix[1][1]=root.transform.matrix[2][2]=1.f;
                root.instanceCustomIndex=id;root.mask=1u<<id;
                root.accelerationStructureReference=reference;
            }
            else for(uint32_t group=0;group<pairGroups;group++)
            {
                auto &root=roots[4u+group];
                root.transform.matrix[0][0]=root.transform.matrix[1][1]=root.transform.matrix[2][2]=1.f;
                root.transform.matrix[0][3]=float(group*4u);
                root.transform.matrix[2][0]=float(group/4u)/256.f;
                root.transform.matrix[2][1]=float(group%4u)/4.f;
                root.instanceCustomIndex=(4u<<16u)|(group*pairTemplates);
                root.mask=16u;root.accelerationStructureReference=reference;
            }
        }
        instances = e.buffer(roots.size()*sizeof(roots[0]), inputUsage);
        e.upload(instances, roots.data(), instances->size);
        VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.data.deviceAddress = instances->address;
        VkAccelerationStructureBuildGeometryInfoKHR topInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        topInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        topInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        topInfo.geometryCount = 1u;
        topInfo.pGeometries = &geometry;
        uint32_t rootCount = uint32_t(roots.size());
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        e.getSizes(e.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &topInfo, &rootCount, &sizes);
        topStorage = e.buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
        topScratch = e.buffer(sizes.buildScratchSize + e.alignment,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create.type = topInfo.type;
        create.buffer = topStorage->handle;
        create.size = topStorage->size;
        check(e.createAS(e.device, &create, nullptr, &top), "create arithmetic TLAS");
        auto command = e.command();
        e.begin(command);
        Engine::barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        for (auto &field : fields)
        {
            VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            info.dstAccelerationStructure = field.handle;
            info.geometryCount = 1u;
            info.pGeometries = &field.geometry;
            info.scratchData.deviceAddress = (field.scratch->address + e.alignment - 1u) & ~VkDeviceAddress(e.alignment - 1u);
            VkAccelerationStructureBuildRangeInfoKHR range{field.count, 0u, 0u, 0u};
            const auto *ranges = &range;
            e.buildAS(command, 1u, &info, &ranges);
        }
        Engine::barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        topInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        topInfo.dstAccelerationStructure = top;
        topInfo.scratchData.deviceAddress = (topScratch->address + e.alignment - 1u) & ~VkDeviceAddress(e.alignment - 1u);
        VkAccelerationStructureBuildRangeInfoKHR topRange{rootCount, 0u, 0u, 0u};
        const auto *topRanges = &topRange;
        e.buildAS(command, 1u, &topInfo, &topRanges);
        Engine::barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        check(vkEndCommandBuffer(command), "end arithmetic BVH build");
        e.submit(command);
        vkFreeCommandBuffers(e.device, e.pool, 1u, &command);
        e.release(topScratch);
        residentBytes = vertices->size + instances->size + topStorage->size;
        for (auto &field : fields)
        {
            e.release(field.scratch);
            residentBytes += field.storage->size;
        }
    }

    ~RTArithmeticSpace()
    {
        vkDeviceWaitIdle(e.device);
        e.destroyAS(e.device, top, nullptr);
        for (auto &field : fields)
        {
            e.destroyAS(e.device, field.handle, nullptr);
            e.release(field.scratch);
            e.release(field.storage);
        }
        e.release(topScratch);
        e.release(topStorage);
        e.release(instances);
        e.release(vertices);
    }
};

class RTGeometryProgram
{
public:
    struct Config
    {
        uint32_t operation, rows, columns, channels, left, right, outputOffset, scratch, scale = 0xffffffffu;
    };
    struct Result
    {
        std::vector<float> values;
        uint64_t rays;
        double gpuMs, wallMs;
    };
    Engine &e;
    RTArithmeticSpace space;
    Buffer *values{}, *status{};
    VkDescriptorSetLayout setLayout{};
    VkPipelineLayout layout{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    VkPipeline pipeline{};
    VkQueryPool timestamps{};

    RTGeometryProgram(Engine &engine) : e(engine), space(engine)
    {
        VkDescriptorSetLayoutBinding bindings[3]{
            {0u, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        create.bindingCount = 3u;
        create.pBindings = bindings;
        check(vkCreateDescriptorSetLayout(e.device, &create, nullptr, &setLayout), "arithmetic descriptor layout");
        VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(Config)};
        VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayout.setLayoutCount = 1u;
        pipelineLayout.pSetLayouts = &setLayout;
        pipelineLayout.pushConstantRangeCount = 1u;
        pipelineLayout.pPushConstantRanges = &constants;
        check(vkCreatePipelineLayout(e.device, &pipelineLayout, nullptr, &layout), "arithmetic pipeline layout");
        VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1u}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2u}};
        VkDescriptorPoolCreateInfo descriptorPool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        descriptorPool.maxSets = 1u;
        descriptorPool.poolSizeCount = 2u;
        descriptorPool.pPoolSizes = sizes;
        check(vkCreateDescriptorPool(e.device, &descriptorPool, nullptr, &pool), "arithmetic descriptor pool");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &setLayout;
        check(vkAllocateDescriptorSets(e.device, &allocate, &set), "arithmetic descriptor set");
        status = e.buffer(4u * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        pipeline = e.pipeline("resident_geometry_arithmetic", layout);
        VkQueryPoolCreateInfo queries{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queries.queryCount = 2u;
        check(vkCreateQueryPool(e.device, &queries, nullptr, &timestamps), "arithmetic timestamps");
    }

    Result run(const std::vector<float> &inputValues, uint32_t workspaceFloats,
               const std::vector<Config> &commands, uint32_t resultFirst, uint32_t resultCount)
    {
        auto started = Clock::now();
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(e.physical, &properties);
        uint64_t bytes = uint64_t(workspaceFloats) * sizeof(float);
        if (bytes > properties.limits.maxStorageBufferRange)
            throw std::runtime_error("Arithmetic workspace exceeds the device storage range");
        for (const auto &config : commands)
        {
            uint64_t groupsX = config.operation == 6u ? config.channels : (config.operation == 3u || config.operation == 5u) ? config.rows : (uint64_t(config.rows) * config.columns + 127u) / 128u;
            uint32_t groupsY = config.operation == 6u ? config.rows : 1u;
            if (groupsX > properties.limits.maxComputeWorkGroupCount[0] || groupsY > properties.limits.maxComputeWorkGroupCount[1])
                throw std::runtime_error("Arithmetic dispatch exceeds the device grid");
        }
        if (!values || values->size < bytes)
        {
            e.release(values);
            values = e.buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            VkWriteDescriptorSetAccelerationStructureKHR acceleration{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            acceleration.accelerationStructureCount = 1u;
            acceleration.pAccelerationStructures = &space.top;
            VkDescriptorBufferInfo buffers[2]{{values->handle, 0u, values->size}, {status->handle, 0u, status->size}};
            VkWriteDescriptorSet writes[3]{};
            for (uint32_t binding = 0; binding < 3u; binding++)
            {
                auto &write = writes[binding];
                write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set;
                write.dstBinding = binding;
                write.descriptorCount = 1u;
                write.descriptorType = binding == 0u ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                if (binding == 0u) write.pNext = &acceleration;
                else write.pBufferInfo = &buffers[binding - 1u];
            }
            vkUpdateDescriptorSets(e.device, 3u, writes, 0u, nullptr);
        }
        e.upload(values, inputValues.data(), inputValues.size() * sizeof(float));
        auto command = e.command();
        e.begin(command);
        vkCmdFillBuffer(command, status->handle, 0u, status->size, 0u);
        Engine::barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdResetQueryPool(command, timestamps, 0u, 2u);
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps, 0u);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0u, 1u, &set, 0u, nullptr);
        for (const auto &config : commands)
        {
            vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(config), &config);
            uint32_t groupsX = config.operation == 6u ? config.channels : (config.operation == 3u || config.operation == 5u) ? config.rows : uint32_t((uint64_t(config.rows) * config.columns + 127u) / 128u);
            vkCmdDispatch(command, groupsX, config.operation == 6u ? config.rows : 1u, 1u);
            Engine::barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, 1u);
        check(vkEndCommandBuffer(command), "end geometry arithmetic");
        e.submit(command);
        uint64_t times[2]{};
        check(vkGetQueryPoolResults(e.device, timestamps, 0u, 2u, sizeof(times), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT), "read arithmetic timestamps");
        vkFreeCommandBuffers(e.device, e.pool, 1u, &command);
        auto counters = e.download<uint32_t>(status, 4u);
        if (counters[0]) throw std::runtime_error("Geometry arithmetic intersection error " + std::to_string(counters[0]));
        auto output = e.download<float>(values, resultCount, uint64_t(resultFirst) * sizeof(float));
        return {std::move(output), uint64_t(counters[1]) | (uint64_t(counters[2]) << 32u), double(times[1] - times[0]) * e.period / 1e6, ms(started)};
    }

    ~RTGeometryProgram()
    {
        vkDeviceWaitIdle(e.device);
        vkDestroyQueryPool(e.device, timestamps, nullptr);
        vkDestroyPipeline(e.device, pipeline, nullptr);
        vkDestroyDescriptorPool(e.device, pool, nullptr);
        vkDestroyPipelineLayout(e.device, layout, nullptr);
        vkDestroyDescriptorSetLayout(e.device, setLayout, nullptr);
        e.release(status);
        e.release(values);
    }
};

// Binary debug protocol for scalar operators and complete MLP/attention
// compositions. Every intermediate stays in the program's GPU value buffer.
class RTGeometryWorker
{
public:
    Engine &e;
    std::unique_ptr<RTGeometryProgram> program;
    RTGeometryWorker(Engine &engine) : e(engine) {}

    static uint32_t size(uint64_t value)
    {
        if (!value || value > 0xffffffffu) throw std::runtime_error("Invalid arithmetic tensor size");
        return uint32_t(value);
    }

    std::vector<char> request(uint32_t operation, uint32_t rows, uint32_t mode)
    {
        if (!rows || operation < 16u || operation > 18u || (operation == 16u && mode > 7u))
            throw std::runtime_error("Invalid geometry arithmetic operation");
        std::vector<uint32_t> shape;
        input(shape, operation == 16u ? 2u : 3u);
        if (std::any_of(shape.begin(), shape.end(), [](uint32_t dimension) { return dimension == 0u; }))
            throw std::runtime_error("Arithmetic dimensions must be positive");
        uint32_t inputCount, workspaceCount, resultFirst, resultCount;
        std::vector<RTGeometryProgram::Config> commands;
        std::vector<float> inputValues;
        if (operation == 16u)
        {
            uint32_t columns = shape[0], channels = shape[1];
            uint32_t leftCount = size(uint64_t(rows) * columns);
            uint32_t rightCount = mode == 6u ? size(uint64_t(channels) * columns) : (mode == 0u || mode == 1u || mode == 2u || mode == 7u) ? leftCount : 0u;
            inputCount = size(uint64_t(leftCount) + rightCount);
            resultCount = mode == 3u ? rows : mode == 6u ? size(uint64_t(rows) * channels) : leftCount;
            resultFirst = inputCount;
            uint32_t scratch = size(uint64_t(resultFirst) + resultCount);
            workspaceCount = size(uint64_t(scratch) + (mode == 5u ? leftCount : 0u));
            commands.push_back({mode, rows, columns, channels, 0u, leftCount, resultFirst, scratch});
            input(inputValues, inputCount);
        }
        else if (operation == 17u)
        {
            uint32_t hidden = shape[0], middle = shape[1], outputWidth = shape[2];
            uint32_t inputRows = size(uint64_t(rows) * hidden);
            uint32_t matrix = size(uint64_t(middle) * hidden);
            uint32_t gate = inputRows, up = size(uint64_t(gate) + matrix), down = size(uint64_t(up) + matrix);
            inputCount = size(uint64_t(down) + uint64_t(outputWidth) * middle);
            uint32_t intermediate = size(uint64_t(rows) * middle);
            uint32_t gateResult = inputCount, upResult = size(uint64_t(gateResult) + intermediate);
            uint32_t activation = size(uint64_t(upResult) + intermediate), output = size(uint64_t(activation) + intermediate);
            uint32_t outputCount = size(uint64_t(rows) * outputWidth);
            workspaceCount = size(uint64_t(output) + outputCount);
            resultFirst = activation;
            resultCount = size(uint64_t(intermediate) + outputCount);
            commands = {{6u, rows, hidden, middle, 0u, gate, gateResult, 0u},
                        {6u, rows, hidden, middle, 0u, up, upResult, 0u},
                        {7u, rows, middle, 1u, gateResult, upResult, activation, 0u},
                        {6u, rows, middle, outputWidth, activation, down, output, 0u}};
            input(inputValues, inputCount);
        }
        else
        {
            uint32_t dimension = shape[0], history = shape[1], valueWidth = shape[2];
            uint32_t queries = size(uint64_t(rows) * dimension), keys = size(uint64_t(history) * dimension);
            uint32_t values = size(uint64_t(history) * valueWidth);
            uint32_t valueFirst = size(uint64_t(queries) + keys), scale = size(uint64_t(valueFirst) + values);
            input(inputValues, scale);
            inputValues.push_back(float(1.0 / std::sqrt(double(dimension))));
            inputCount = size(uint64_t(scale) + 1u);
            uint32_t scores = inputCount, scoreCount = size(uint64_t(rows) * history);
            uint32_t scratch = size(uint64_t(scores) + scoreCount), weights = size(uint64_t(scratch) + scoreCount);
            uint32_t output = size(uint64_t(weights) + scoreCount), outputCount = size(uint64_t(rows) * valueWidth);
            workspaceCount = size(uint64_t(output) + outputCount);
            resultFirst = weights;
            resultCount = size(uint64_t(scoreCount) + outputCount);
            commands = {{6u, rows, dimension, history, 0u, queries, scores, 0u, scale},
                        {5u, rows, history, 1u, scores, 0u, weights, scratch},
                        {6u, rows, history, valueWidth, weights, valueFirst, output, 0u}};
        }
        if (!std::all_of(inputValues.begin(), inputValues.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("Geometry arithmetic input must be finite");
        auto setup = Clock::now();
        if (!program) program = std::make_unique<RTGeometryProgram>(e);
        double setupMs = ms(setup);
        auto result = program->run(inputValues, workspaceCount, commands, resultFirst, resultCount);
        std::vector<char> response;
        append(response, std::vector<uint32_t>{0x52474131u, RTArithmeticSpace::exponentialSegments, 1u, resultCount});
        append(response, std::vector<uint64_t>{result.rays, program->space.residentBytes});
        append(response, std::vector<double>{setupMs, result.gpuMs, result.wallMs});
        append(response, result.values);
        return response;
    }

    void serve()
    {
        const uint32_t hello[4]{0x52544741u, 1u, RTArithmeticSpace::exponentialSegments, 8u};
        std::cout.write(reinterpret_cast<const char *>(hello), sizeof(hello));
        std::cout.flush();
        uint32_t operation;
        while (std::cin.read(reinterpret_cast<char *>(&operation), sizeof(operation)))
        {
            if (!operation) break;
            uint32_t rows, mode;
            if (!std::cin.read(reinterpret_cast<char *>(&rows), sizeof(rows)) ||
                !std::cin.read(reinterpret_cast<char *>(&mode), sizeof(mode)))
                throw std::runtime_error("Incomplete geometry arithmetic request");
            auto response = request(operation, rows, mode);
            if (response.size() > 0xffffffffu) throw std::runtime_error("Arithmetic response exceeds protocol size");
            uint32_t header[2]{0u, uint32_t(response.size())};
            std::cout.write(reinterpret_cast<const char *>(header), sizeof(header));
            std::cout.write(response.data(), response.size());
            std::cout.flush();
        }
    }
};
