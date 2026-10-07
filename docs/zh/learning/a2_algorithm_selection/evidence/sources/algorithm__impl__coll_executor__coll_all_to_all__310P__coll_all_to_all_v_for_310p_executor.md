# coll_all_to_all_v_for_310p_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllVFor310PExecutor::Orchestrate

[coll_all_to_all_v_for_310p_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/310P/coll_all_to_all_v_for_310p_executor.cc#L21)

```cpp
HcclResult CollRunAlltoAllVFor310PExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
        HCCL_ERROR("[CollRunAlltoAllVFor310PExecutor][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], AlltoAllVFor310P orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllVFor310PExecutor::KernelRun

[coll_all_to_all_v_for_310p_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/310P/coll_all_to_all_v_for_310p_executor.cc#L94)

```cpp
HcclResult CollRunAlltoAllVFor310PExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVFor310PExecutor][KernelRun] AllToAll for 310p start.");

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 执行
    std::unique_ptr<AlgTemplateBase> executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_FOR310P, dispatcher_);
    CHK_SMART_PTR_NULL(executor);
    CHK_RET(executor->Prepare(algResResp_->paramInputMem, algResResp_->paramOutputMem, execMem.inputMem,
        execMem.outputMem, algResResp_->notifiesMain, algResResp_->notifiesAux,
        const_cast<Stream&>(param.stream), algResResp_->slaveStreams, outerCommInfo.links,
        topoAttr_.userRank, topoAttr_.userRankSize, allMeshAggregationSendRecvInfo_));

    u32 rankSize = outerCommInfo.localRankSize;
    CHK_RET(executor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(executor->RunAsync());

    HCCL_INFO("[CollRunAlltoAllVFor310PExecutor] executor run success.");
    return HCCL_SUCCESS;
}
```
