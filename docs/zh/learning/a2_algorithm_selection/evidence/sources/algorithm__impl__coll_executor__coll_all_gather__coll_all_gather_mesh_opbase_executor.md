# coll_all_gather_mesh_opbase_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherMeshOpbaseExecutor::CalcLoopMaxCount

[coll_all_gather_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_executor.cc#L63)

```cpp
u64 CollAllGatherMeshOpbaseExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}
```

### CollAllGatherMeshOpbaseExecutor::KernelRun

[coll_all_gather_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_executor.cc#L82)

```cpp
HcclResult CollAllGatherMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherMeshOpbaseExecutor] AllGather KernelRun");
    u8 *curInputPtr = static_cast<u8 *>(execMem.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(execMem.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = 0;
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环AllGather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize); // 减少dma out大小
    CHK_SMART_PTR_NULL(currentOutputMem);

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };

    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, &opInfo, level0CommInfo.localRank, level0CommInfo.localRankSize));
    CHK_RET(level0TempAlg->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
        dataSegsSlice, baseOffset));

    u32 rankSize = level0CommInfo.localRankSize;
    CHK_RET(level0TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));

    HCCL_INFO("AllGather mesh level0 run success");
    return HCCL_SUCCESS;
}
```
