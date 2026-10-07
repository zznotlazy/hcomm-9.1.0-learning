# coll_all_gather_mesh_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherMeshExecutor::CalcLoopMaxCount

[coll_all_gather_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L65)

```cpp
u64 CollAllGatherMeshExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}
```

### CollAllGatherMeshExecutor::KernelRun

[coll_all_gather_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L72)

```cpp
HcclResult CollAllGatherMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherMeshExecutor][KernelRun]AllGather mesh start");
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;

    HCCL_DEBUG("[CollAllGatherMeshExecutor][KernelRun]serverIndex is %u, commIndex is %u", serverIndex, commIndex);
    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * level0RankSize;
    u64 level0Offset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + level0Offset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshExecutor][KernelRun]AllGather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        baseOffset + level0Offset, inputMemSize), ret);

    // 第二步，各个AI Server 内 multi stream mesh AllGather
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = level0RankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    // mesh算法stream数量为server内rank数减1
    CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环AllGather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize * level0RankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);

    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC,
                                                                       dispatcher_);
    } else {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH,
                                                                       dispatcher_);
    }
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, nullptr, commIndex, level0RankSize));
    CHK_RET(level0TempAlg->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem,
        execMem.count * level0RankSize, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));
    u32 rankSize = level0RankSize;
    CHK_RET(level0TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("AllGather mesh HD level0 run success");

    //  第三步， AI server 间 recursive halving doubling AllGather
    u64 hdSize = inputMemSize * level0RankSize;
    u64 hdCount = hdSize / perDataSize;

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllGather mesh: using halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
    CHK_RET(level1TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, hdCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), 0));

    rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    HCCL_INFO("AllGather mesh HD level1 run success");
    return HCCL_SUCCESS;
}
```

### CollAllGatherMeshExecutor::SelectTempAlg

[coll_all_gather_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L193)

```cpp
HcclResult CollAllGatherMeshExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
            // 1-单server-SDMA
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather mesh: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllGather mesh: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
