# coll_all_reduce_mesh_graph_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshGraphPipelineExecutor::KernelRun

[coll_all_reduce_mesh_graph_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_graph_pipeline_executor.cc#L70)

```cpp
HcclResult CollAllReduceMeshGraphPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllReduceMeshGraphPipelineExecutor][Run]CollAllReduceMeshGraphPipelineExecutor begins.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.inputMem, param.DataDes.dataType, param.reduceType);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALLREDUCE_GRAPH_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));
    CHK_RET(tempAlg->Prepare(&opInfo, execMem.inputMem, execMem.outputMem, execMem.count,
        level1CommInfo, level0CommInfo, const_cast<Stream&>(param.stream),
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```
