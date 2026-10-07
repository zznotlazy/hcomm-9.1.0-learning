# coll_all_gather_mesh_opbase_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherMeshOpbasePipelineExecutor::CalcLoopMaxCount

[coll_all_gather_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_pipeline_executor.cc#L70)

```cpp
u64 CollAllGatherMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    return maxCountPerLoop;
}
```

### CollAllGatherMeshOpbasePipelineExecutor::KernelRun

[coll_all_gather_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_pipeline_executor.cc#L82)

```cpp
HcclResult CollAllGatherMeshOpbasePipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllGatherMeshOpbasePipelineExecutor][KernelRun]AllGatherMeshOpbasePipelineExecutor begins.");

    // step 1 先获取 comm level0 \ comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED
    };

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(&opInfo, topoAttr_.userRank, execMem.count, execMem.inputMem, execMem.outputMem,
        level0CommInfo, level1CommInfo, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```

### CollAllGatherMeshOpbasePipelineExecutor::SelectTempAlg

[coll_all_gather_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_pipeline_executor.cc#L124)

```cpp
HcclResult CollAllGatherMeshOpbasePipelineExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
