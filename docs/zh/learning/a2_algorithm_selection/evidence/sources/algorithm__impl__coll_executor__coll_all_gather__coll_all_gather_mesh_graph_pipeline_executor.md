# coll_all_gather_mesh_graph_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherMeshGraphPipelineExecutor::KernelRun

[coll_all_gather_mesh_graph_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_graph_pipeline_executor.cc#L68)

```cpp
HcclResult CollAllGatherMeshGraphPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllGatherMeshGraphPipelineExecutor][KernelRun]AllGatherMeshGraphPipelineExecutor begins.");

    // 先获取 comm level0 和 comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 激活从流
    CHK_RET(ActiveSlaveStreams(param.stream));

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED};

    std::unique_ptr<AlgTemplateBase> tempAlg =
        AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_GRAPH_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(&opInfo,
        topoAttr_.userRank,
        execMem.count,
        execMem.inputMem,
        execMem.outputMem,
        level0CommInfo,
        level1CommInfo,
        const_cast<Stream &>(param.stream),
        algResResp_->slaveStreams,
        algResResp_->notifiesMain,
        algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```
