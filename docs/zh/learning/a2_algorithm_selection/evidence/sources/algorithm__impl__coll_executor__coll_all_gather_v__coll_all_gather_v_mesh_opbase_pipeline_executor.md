# coll_all_gather_v_mesh_opbase_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherVMeshOpbasePipelineExecutor::CalcLoopMaxCount

[coll_all_gather_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_opbase_pipeline_executor.cc#L97)

```cpp
u64 CollAllGatherVMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}
```

### CollAllGatherVMeshOpbasePipelineExecutor::KernelRun

[coll_all_gather_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_opbase_pipeline_executor.cc#L110)

```cpp
HcclResult CollAllGatherVMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllGatherVMeshOpbasePipelineExecutor][KernelRun]AllGatherVMeshOpbasePipelineExecutor begins.");

    // step 1 先获取 comm level0 \ comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    u32 perDataSize = SIZE_TABLE[param.VDataDes.dataType];
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < topoAttr_.userRankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * perDataSize;
        userslice.size = counts[rank] * perDataSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, 0, param.VDataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_V_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    Stream stream = param.stream;
    CHK_RET(tempAlg->Prepare(&opInfo, topoAttr_.userRank, execMem.count, execMem.inputMem, execMem.outputMem,
        level0CommInfo, level1CommInfo, stream,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, outputSlices));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshOpbasePipelineExecutor::SelectTempAlg

[coll_all_gather_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_opbase_pipeline_executor.cc#L168)

```cpp
HcclResult CollAllGatherVMeshOpbasePipelineExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_V_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
