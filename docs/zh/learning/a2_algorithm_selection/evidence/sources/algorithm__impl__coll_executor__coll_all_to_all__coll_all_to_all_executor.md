# coll_all_to_all_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAlltoAllExecutor::Orchestrate

[coll_all_to_all_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor.cc#L22)

```cpp
HcclResult CollAlltoAllExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;
    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;

        auto opMeta = GetOpMeta(param.opType, algRes.paramInputMem.size());   // override
        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        bool massTasks = HasMassTasks(allMeshAggregationSendRecvInfo_);
        if (massTasks) {
            CHK_RET(SetNormalMode(dispatcher_));
        }
        ret = KernelRun(param, execMem);
    } else {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllExecutor][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

    HCCL_INFO("tag[%s], AlltoAll executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAlltoAllExecutor::CalcResRequest

[coll_all_to_all_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor.cc#L108)

```cpp
HcclResult CollAlltoAllExecutor::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };

    // AICPU aicpuUnfold展开模式下临时强制OP_BASE，使整个资源计算路径与AICPU侧一致
    // CalcScratchMemSize走OP_BASE分支正确计算scratch
    // CalcCommInfo走OP_BASE分支设outputMemType为CCL_OUTPUT而非SCRATCH
    // 避免transport因scratch未分配而拿到nullptr
    // AIV executor有独立的资源计算逻辑，不需要force OP_BASE
    const bool needForceOpBase = param.aicpuUnfoldMode && !param.isZeroCopy && !desc_.isAivMode;
    HCCL_INFO("[CollAlltoAllExecutor][CalcResRequest] aicpuUnfoldMode[%d] isZeroCopy[%d] "
        "needForceOpBase[%d] workflowMode[%d] tag[%s]",
        param.aicpuUnfoldMode, param.isZeroCopy, needForceOpBase,
        workflowMode_, param.tag.c_str());
    const HcclWorkflowMode savedWorkflowMode = workflowMode_;
    if (needForceOpBase) {
        HCCL_INFO("[CollAlltoAllExecutor][CalcResRequest] aicpuUnfoldMode force OpBase, "
            "originalWorkflowMode[%d], tag[%s]",
            savedWorkflowMode, param.tag.c_str());
        workflowMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    }

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(CalcAivBufferRequest(aivBufferRequest));
    CHK_RET(CalcCommInfo(opTransport));

    if (needForceOpBase) {
        HCCL_DEBUG("[CollAlltoAllExecutor][CalcResRequest] restore workflowMode "
            "after resource calc, scratchMemSize[%llu]", scratchMemSize);
        workflowMode_ = savedWorkflowMode;
    }

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO("[CollAlltoAllExecutor][%s] streamNum[%u], notifyNum[%u], sctrachMemSize[%llu], aivBufferRequest[%llu]",
        __func__, resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.aivBufferRequest);
    // 打印建链诉求
    for (u32 levelIndex = 0; levelIndex < COMM_LEVEL_RESERVED; levelIndex++) {
        LevelNSubCommTransport &levelTransport = resourceRequest.opTransport[levelIndex];
        u32 ringSize = levelTransport.size();
        for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
            SingleSubCommTransport &subCommTransport = levelTransport[ringIndex];
            u32 rankSize = subCommTransport.transportRequests.size();
            for (u32 rankIndex = 0; rankIndex < rankSize; rankIndex++) {
                if (subCommTransport.transportRequests[rankIndex].isValid == true) {
                    HCCL_INFO("[CollAlltoAllExecutor][CalcResRequest]" \
                        "levelIndex[%u], ringIndex[%u], rankIndex[%u], userRank[%u], remoteRank[%u]" \
                        "isUsedRdma[%d]",
                        levelIndex, ringIndex, rankIndex, subCommTransport.transportRequests[rankIndex].localUserRank,
                        subCommTransport.transportRequests[rankIndex].remoteUserRank,
                        subCommTransport.transportRequests[rankIndex].isUsedRdma);
                }
            }
        }
    }
    CHK_RET(CheckNeedCreateVirtualLinks(resourceRequest));
    HCCL_DEBUG("[%s] process success", __func__);
    return HCCL_SUCCESS;
}
```
