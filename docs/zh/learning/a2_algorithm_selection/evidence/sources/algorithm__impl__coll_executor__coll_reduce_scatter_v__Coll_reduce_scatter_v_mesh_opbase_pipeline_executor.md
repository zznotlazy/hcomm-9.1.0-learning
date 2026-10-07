# Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterVMeshOpbasePipelineExecutor::SelectTempAlg

[Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc#L97)

```cpp
HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
    }
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVMeshOpbasePipelineExecutor::CalcLoopMaxCount

[Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc#L107)

```cpp
u64 CollReduceScatterVMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = ((inCCLbufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) \
            * HCCL_MIN_SLICE_ALIGN_910B - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    HCCL_INFO("[CollReduceScatterVMeshOpbasePipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollReduceScatterVMeshOpbasePipelineExecutor::KernelRun

[Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc#L141)

```cpp
HcclResult CollReduceScatterVMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbasePipelineExecutor] reducescatterv pipeline run");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    // 先获取 comm level0 \ comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < topoAttr_.userRankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.inputMem, dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_V_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType,
        param.root, param.reduceType};

    Stream stream = param.stream;
    CHK_RET(tempAlg->Prepare(&opInfo, execMem.inputMem, execMem.inputMem.size(), inputSlices, level0CommInfo,
        level1CommInfo, stream, algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux, reduceAttr));
    CHK_RET(tempAlg->RunAsync());

    return HCCL_SUCCESS;
}
```
