# coll_all_reduce_mesh_opbase_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshOpbasePipelineExecutor::CalcLoopMaxCount

[coll_all_reduce_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_pipeline_executor.cc#L71)

```cpp
u64 CollAllReduceMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    if (cclBuffSize <= HCCL_MIN_SLICE_ALIGN_910B) {
        return 0;
    }
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) /
        unitSize * topoAttr_.userRankSize;
    return maxCountPerLoop;
}
```

### CollAllReduceMeshOpbasePipelineExecutor::KernelRun

[coll_all_reduce_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_pipeline_executor.cc#L93)

```cpp
HcclResult CollAllReduceMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllReduceMeshOpbasePipelineExecutor][Run]CollAllReduceMeshOpbasePipelineExecutor begins.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_OPBASE_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));
    CHK_RET(tempAlg->Prepare(&opInfo, execMem.inputMem, execMem.outputMem, execMem.count,
        level1CommInfo, level0CommInfo, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshOpbasePipelineExecutor::SelectTempAlg

[coll_all_reduce_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_pipeline_executor.cc#L134)

```cpp
HcclResult CollAllReduceMeshOpbasePipelineExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_OPBASE_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
