# coll_native_executor_base.cc

共享执行器函数；按设备条件确认A2分支。

### CollNativeExecutorBase::CalcResRequest

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L33)

```cpp
HcclResult CollNativeExecutorBase::CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest)
{
    (void)ParseParam(param);

    u64 scratchMemSize = 0U;
    u32 streamNum = 0U;
    u32 notifyNum = 0U;
    u64 aivBufferRequest = 0U;
    std::vector<LevelNSubCommTransport> opTransport {
        std::vector<LevelNSubCommTransport>(static_cast<u32>(COMM_LEVEL_RESERVED))
    };

    CHK_RET(CalcScratchMemSize(scratchMemSize));
    CHK_RET(CalcOptimalIntraRing(param));
    CHK_RET(CalcStreamNum(streamNum));
    CHK_RET(CalcNotifyNum(streamNum, notifyNum));
    CHK_RET(CalcAivBufferRequest(aivBufferRequest));
    CHK_RET(CalcCommInfo(opTransport));

    CHK_RET(BuildResourceRequest(scratchMemSize, streamNum, notifyNum, aivBufferRequest, opTransport, resourceRequest));
    HCCL_INFO("streamNum[%u], notifyNum[%u], sctrachMemSize[%llu], aivBufferRequest[%llu]",
        resourceRequest.streamNum, resourceRequest.notifyNum, resourceRequest.scratchMemSize,
        resourceRequest.aivBufferRequest);
    // 打印建链诉求
    PrintTransportRequest(resourceRequest);
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::CalcAivBufferRequest

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L85)

```cpp
HcclResult CollNativeExecutorBase::CalcAivBufferRequest(u64 &aivBufferRequest)
{
    if (desc_.isAivMode) {
        SalSetBitOne(aivBufferRequest, ATTR_POS_AIV_COMM_BUFFER);
    }
    if (desc_.isAivCrossNode) {
        SalSetBitOne(aivBufferRequest, ATTR_POS_AIV_COMM_INFO_BUFFER);
    }
    HCCL_INFO("[CollNativeExecutorBase][CalcAivBufferRequest]tag[%s] aivBufferRequest is [%llu]", tag_.c_str(),
        aivBufferRequest);
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::CalcLevel1CommInfo

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L211)

```cpp
HcclResult CollNativeExecutorBase::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] start", tag_.c_str());
    u32 root = root_;
    if (opType_ == HcclCMDType::HCCL_CMD_BROADCAST && topoAttr_.superPodNum > 1) {
        root = topoMatcher_->GetSubRootWithSuperPod(topoAttr_.userRank, root_);
        HCCL_DEBUG("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] subroot is %u usrRank is %u root_ is %u",
            tag_.c_str(), root, topoAttr_.userRank, root_);
    }
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX, root);

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc RingCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NHRCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NHRV1CommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        commParaLevel1.commPlane = CommPlane::COMM_LEVEL1_AHC;
        commParaLevel1.commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc AHCCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        commParaLevel1.commPlane = CommPlane::COMM_LEVEL1_AHC;
        commParaLevel1.commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc AHCBrokeCommInfo", tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc NBCommInfo", tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc HDCommInfo", tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[commParaLevel1.commPlane], inputType, outputType));
    HCCL_INFO("[CollNativeExecutorBase][COMM_LEVEL1]tag[%s] Calc CommInfo Finish", tag_.c_str());

    HCCL_INFO("[CollNativeExecutorBase][CalcLevel1CommInfo]tag[%s] Calc CommInfo Finish", tag_.c_str());

    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::CalcLevel2CommInfo

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L264)

```cpp
HcclResult CollNativeExecutorBase::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[%s] select AHC bypass level2 comm calculate", __func__);
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX, root_);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s] Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s] Calc NBCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s] Calc HDCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s] Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::KernelRun

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L315)

```cpp
HcclResult CollNativeExecutorBase::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_WARNING("[CollNativeExecutorBase][KernelRun]Using the default kernel run, nothing is done.");
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::AddSubStreamToProfiling

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L336)

```cpp
HcclResult CollNativeExecutorBase::AddSubStreamToProfiling()
{
#ifndef OPEN_HCCL_TEST
    if (((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        !hccl::ProfilingManagerPub::GetAdditionInfoState() &&
        !hccl::ProfilingManagerPub::GetTaskApiState() &&
        !hccl::ProfilingManagerPub::GetThreadCaptureStatus())) {
        return HCCL_SUCCESS;
    }

    for (u32 streamIndex = 0; streamIndex < algResResp_->slaveStreams.size(); streamIndex++) {
        // profiling加入从环的stream
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(algResResp_->slaveStreams[streamIndex].id(), tag_, streamIndex + 1, algType_);
    }
#endif
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::CopyAivCommInfoToDevice

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L796)

```cpp
HcclResult CollNativeExecutorBase::CopyAivCommInfoToDevice(const CommPlane levelIndex, const u32 subLevelIndex,
    AlgResourceResponse& algResource)
{
    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(levelIndex, subLevelIndex + 1));
    SubCommInfo commInfo = GetSubCommInfo(levelIndex, subLevelIndex);
    u32 localRank = commInfo.localRank;
    u32 localRankSize = commInfo.localRankSize;

    void* buffersInOut[MAX_RANK_SIZE_A3 * 2] = {};
    bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    for (u32 i = 0; i < localRankSize; i++) {
        u32 idx = (i << 1);
        if (i != localRank) {
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersInOut[idx])));
            CHK_RET(commInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersInOut[idx + 1])));
        } else {
            buffersInOut[idx] = isOpbaseMode ? algResource.cclInputMem.ptr() : algResource.paramInputMem.ptr();
            buffersInOut[idx + 1] = algResource.aivOutputMem.ptr();
        }
    }
    const u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(algResource.aivCommInfoMem.ptr(), sizeof(u64) * localRankSize * bufferNum,
        buffersInOut, sizeof(u64) * localRankSize * bufferNum, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::SelectTempAlg

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L829)

```cpp
HcclResult CollNativeExecutorBase::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    (void) level1TempAlg;
    (void) level1RankSize;
    return HCCL_SUCCESS;
}
```

### CollNativeExecutorBase::SetOpCache

[coll_native_executor_base.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_native_executor_base.cc#L841)

```cpp
HcclResult CollNativeExecutorBase::SetOpCache(const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs, 
    const AivAlgArgs& algArgs, ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo, bool isA3CrossNode)
{
    cacheInfo_.opArgs = opArgs;
    cacheInfo_.topoArgs = topoArgs;
    cacheInfo_.resourceArgs = resourceArgs;
    cacheInfo_.algArgs = algArgs;
    cacheInfo_.profilingInfo = aivProfilingInfo;
    cacheInfo_.extraArgs = extraArgs;
    cacheInfo_.isUseCache = true;

    if (isA3CrossNode) {
        u8 buffersOutSize = 2 * sizeof(void *);
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersIn, sizeof(void *), resourceArgs.buffersIn, sizeof(void *)));
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersOut, buffersOutSize, resourceArgs.buffersOut, buffersOutSize));
    } else {
        u64 bufferInfoSize = sizeof(void *) * topoArgs.rankSize;
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersIn, bufferInfoSize, resourceArgs.buffersIn, bufferInfoSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(cacheInfo_.buffersOut, bufferInfoSize, resourceArgs.buffersOut, bufferInfoSize));
    }

    HCCL_INFO("[CollNativeExecutorBase][SetOpCache] cmdType:%d, count:%llu, dataType:%d, op:%d, " \
        "rank:%u, rankSize:%u, serverNum:%u, isA3CrossNode:%d, buffersIn:%p, buffersOut:%p", opArgs.cmdType, opArgs.count, opArgs.dataType, 
        opArgs.op, topoArgs.rank, topoArgs.rankSize, topoArgs.serverNum, isA3CrossNode,
        cacheInfo_.buffersIn, cacheInfo_.buffersOut);

    return HCCL_SUCCESS;
}
```
