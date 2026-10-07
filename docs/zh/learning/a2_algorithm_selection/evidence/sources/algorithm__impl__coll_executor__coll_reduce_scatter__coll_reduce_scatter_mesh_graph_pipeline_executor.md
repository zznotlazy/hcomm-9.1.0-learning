# coll_reduce_scatter_mesh_graph_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterMeshGraphPipelineExecutor::KernelRun

[coll_reduce_scatter_mesh_graph_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_graph_pipeline_executor.cc#L73)

```cpp
HcclResult CollReduceScatterMeshGraphPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollReduceScatterMeshGraphPipelineExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    DeviceMem userInMem = DeviceMem::create(param.inputPtr, param.DataDes.count * unitSize);
    u64 reduceAttr = GetReduceAttr(userInMem, userInMem, param.DataDes.dataType, param.reduceType);

    CHK_RET(ActiveSlaveStreams(param.stream));

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_GRAPH_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"",
        execMem.inputPtr,
        execMem.outputPtr,
        param.DataDes.count,
        param.DataDes.dataType,
        param.root,
        param.reduceType};

    CHK_RET(tempAlg->Prepare(&opInfo,
        execMem.inputMem,
        param.DataDes.count,
        0,
        0,
        level0CommInfo,
        level1CommInfo,
        const_cast<Stream &>(param.stream),
        algResResp_->slaveStreams,
        algResResp_->notifiesMain,
        algResResp_->notifiesAux,
        reduceAttr));
    CHK_RET(tempAlg->RunAsync());

    return HCCL_SUCCESS;
}
```
