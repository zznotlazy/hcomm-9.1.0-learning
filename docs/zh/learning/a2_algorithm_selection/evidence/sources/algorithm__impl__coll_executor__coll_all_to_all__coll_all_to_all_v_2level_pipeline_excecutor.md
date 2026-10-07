# coll_all_to_all_v_2level_pipeline_excecutor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllVTwoLevelPipeline::KernelRun

[coll_all_to_all_v_2level_pipeline_excecutor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_2level_pipeline_excecutor.cc#L137)

```cpp
HcclResult CollRunAlltoAllVTwoLevelPipeline::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVTwoLevelPipeline][KernelRun] AllToAllV two level pipeline start");

    bool cclEnough = true;
    if ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || topoAttr_.deviceType == DevType::DEV_TYPE_910_93)
        && GetAlltoall2LevelPipelineMaxScratchSize910B(allMeshAggregationSendRecvInfo_) > execMem.inputMem.size()) {
        cclEnough = false;
    }
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVTwoLevelPipeline][KernelRun] AllToAllV pipeline run %s algo",
        cclEnough ? "cclEnough" : "ping pong");
    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algResResp_->paramInputMem;
    a2aPipelineMemory.userOutput = algResResp_->paramOutputMem;
    // 具体传入 A2aPipelineMemory 对象的 alltoall pipeline executor 会根据图模式还是单算子模式
    // 选择使用 ccl 还是 scratch，不会访问空指针
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        a2aPipelineMemory.cclInBuffer = execMem.inputMem;
        a2aPipelineMemory.cclOutBuffer = execMem.outputMem;
    } else {
        a2aPipelineMemory.scratchMem = execMem.scratchMem;
    }
    std::unique_ptr<AlgTemplateBase> alltoallPipe = nullptr;
    if (cclEnough) {
        alltoallPipe = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH, dispatcher_);
    } else {
        alltoallPipe = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_PING_PONG, dispatcher_);
    }

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0);

    CHK_SMART_PTR_NULL(alltoallPipe);
    CHK_RET(alltoallPipe->Prepare(topoAttr_.userRank, a2aPipelineMemory, level0CommInfo, level1CommInfo,
        const_cast<Stream&>(param.stream), algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, allMeshAggregationSendRecvInfo_, workflowMode_));
    CHK_RET(alltoallPipe->RunAsync());
    HCCL_INFO("[CollRunAlltoAllVTwoLevelPipeline][kernelRun] AllToAllV two level pipeline exec end");
    return HCCL_SUCCESS;
}
```
