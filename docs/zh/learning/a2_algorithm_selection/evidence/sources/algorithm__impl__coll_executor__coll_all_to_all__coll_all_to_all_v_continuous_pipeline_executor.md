# coll_all_to_all_v_continuous_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAlltoAllVContinuousPipeline::Orchestrate

[coll_all_to_all_v_continuous_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_continuous_pipeline_executor.cc#L130)

```cpp
HcclResult CollAlltoAllVContinuousPipeline::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllVContinuousPipeline][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], CollAlltoAllVContinuousPipeline tempAlg orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAlltoAllVContinuousPipeline::KernelRun

[coll_all_to_all_v_continuous_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_continuous_pipeline_executor.cc#L155)

```cpp
HcclResult CollAlltoAllVContinuousPipeline::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllVContinuousPipeline][KernelRun] AllToAllV npu direct start.");

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    const u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_CONTINUOUS_PIPELINE, dispatcher_);

    CHK_SMART_PTR_NULL(tempAlg);

    SendRecvInfo sendRecvInfo;
    CHK_RET(FillLocalSendRecvInfo(param, sendRecvInfo));

    A2aPipelineMemory a2aPipelineMemory;
    a2aPipelineMemory.userInput = algResResp_->paramInputMem;
    a2aPipelineMemory.userOutput = algResResp_->paramOutputMem;
    a2aPipelineMemory.cclInBuffer = execMem.inputMem;
    a2aPipelineMemory.cclOutBuffer = execMem.outputMem;

    HCCL_INFO("[CollAlltoAllVContinuousPipeline] Memory info[addr, size]: userInput[%p, %llu], userOutput[%p, %llu], "
        "cclInBuffer[%p, %llu] ,cclOutBuffer[%p, %llu].",
        a2aPipelineMemory.userInput.ptr(), a2aPipelineMemory.userInput.size(),
        a2aPipelineMemory.userOutput.ptr(), a2aPipelineMemory.userOutput.size(),
        a2aPipelineMemory.cclInBuffer.ptr(), a2aPipelineMemory.cclInBuffer.size(),
        a2aPipelineMemory.cclOutBuffer.ptr(), a2aPipelineMemory.cclOutBuffer.size()
    );

#ifndef OPEN_HCCL_TEST
    std::vector<SendRecvInfo> sendRecvInfoList{sendRecvInfo};
#else
    // 适配算法检查器，传入全局的info
    std::vector<SendRecvInfo> sendRecvInfoList = allMeshAggregationSendRecvInfo_;
#endif

    CHK_RET(tempAlg->Prepare(topoAttr_.userRank,
        a2aPipelineMemory,
        level0CommInfo,
        level1CommInfo,
        param.stream,
        algResResp_->slaveStreams,
        algResResp_->notifiesMain,
        algResResp_->notifiesAux,
        sendRecvInfoList,
        param.All2AllDataDes.sendType,
        workflowMode_));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollAlltoAllVContinuousPipeline] executor run success.");
    return HCCL_SUCCESS;
}
```
