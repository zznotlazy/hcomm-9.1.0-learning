# coll_all_gather_aiv_rdma_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherExecutor

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L18)

```cpp
: CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
    desc_.isAivMode = true;
}
```

### CollAllGatherAivRdmaExecutor::CalcStreamNum

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L24)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum=0;
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::CalcCommInfo

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L31)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_OUTPUT, TransportMemType::AIV_INPUT, opTransport));
    CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::CalcLevel0CommInfo

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L38)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::CalNumBlocks

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L48)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    u32 bestNumBlocks = numBlocks;
    
    CHK_PRT_RET(numBlocks_ < numBlocks,
        HCCL_WARNING("[CollAllGatherAivRdmaExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
        numBlocks_, numBlocks), HCCL_E_PARA);

    HCCL_INFO("[CollAllGatherAivRdmaExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommanded[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::Orchestrate

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L62)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    u64 totalSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.scratchMem = algRes.scratchMem;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclOutputMem;
        execMem.outputMem = algRes.aivInputMem;  // 使用aivIn的第33M作为Flag区
    } else {
        execMem.inputMem = algRes.aivInputMem;
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize * topoAttr_.userRankSize);
    }

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherAivRdmaExecutor]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllGather executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::KernelRun

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L93)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][KernelRun]AllGather aiv enter");
    HcclWorkflowMode workflow = workflowMode_;
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    //获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 outerRankSize = outerCommInfo.localRankSize;
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = innerCommInfo.localRank;
    u64 inputMemSize = perDataSize * param.DataDes.count;

    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(inputMemSize);
    auto opMeta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData, false, CopyPattern::BCOPY, 
        false, true);

    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    DeviceMem dstMem;
    u64 baseOffset = serverIndex * inputMemSize;
    if (isOpbase) {
        dstMem = execMem.inputMem.range(baseOffset, inputMemSize);
        CHK_SMART_PTR_NULL(dstMem);
    }
    else {
        baseOffset = serverIndex * inputMemSize * outerRankSize;
        u64 outerOffset = commIndex * inputMemSize;
         dstMem = execMem.inputMem.range(baseOffset + outerOffset, inputMemSize);
        CHK_SMART_PTR_NULL(dstMem);
    }

    // STP1，将数据从userinput内存拷贝到ccloutput内存的对应位置
    DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(param.inputPtr), inputMemSize);
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherAivRdmaExecutor][KernelRun]AllGather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        baseOffset, inputMemSize), ret);
    //  STP2， AI server 间 recursive halving doubling AllGather
    u64 hdCount = inputMemSize / perDataSize;
    std::unique_ptr<ExecutorBase> innerExecutor;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllGather mesh: using halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    DeviceMem rdmadstMem;
    if (isOpbase) {
        rdmadstMem = execMem.inputMem.range(0, inputMemSize * innerCommInfo.localRankSize);
        CHK_SMART_PTR_NULL(rdmadstMem);
    }
    else {
        hdCount = inputMemSize*outerRankSize / perDataSize;
        rdmadstMem = execMem.inputMem;
    }
 
    CHK_RET(innerExecutor->Prepare(rdmadstMem, rdmadstMem, execMem.inputMem, hdCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), 0));
 
    u32 rankSize = innerCommInfo.localRankSize;
    CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

    //STP3 机内重排
    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;

    void* tmpAivBufferData;
    void* tmpAivBufferFlag;

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(tmpAivBufferData)));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(tmpAivBufferFlag)));
            buffersIn[i] = static_cast<u8 *>(tmpAivBufferData);
            buffersOut[i] = static_cast<u8 *>(tmpAivBufferFlag) + HCCL_MID_COUNT_32_MB;
        } else {
            buffersIn[i] = static_cast<u8 *>(execMem.inputMem.ptr());
            buffersOut[i] = static_cast<u8 *>(execMem.outputMem.ptr()) + HCCL_MID_COUNT_32_MB;
        }
    }
    
    u32 serverNum = innerCommInfo.localRankSize;

    AivOpArgs opArgs {
        HcclCMDType::HCCL_CMD_ALLGATHER, execMem.inputPtr, execMem.outputPtr, param.DataDes.count,
        param.DataDes.dataType, HCCL_REDUCE_RESERVED, 0, isOpbase
    };
    AivTopoArgs topoArgs {
        localRank, localRankSize, topoAttr_.isDiffDeviceModule ? topoAttr_.devicePhyId : A_X_SIZE,
        0, serverNum, topoAttr_.deviceType, algoAttr_.identifier
    };
    u32 numBlocks;
    CHK_PRT_RET(CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs {
        param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(), numBlocks_, param.aivTag
    };
    AivAlgArgs algArgs {0};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;

    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][KernelRun]allGather aiv run success.");
    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::Getlevel1CommRank

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L235)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    if (CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_8P_RING) ? LEVEL0_PLANE_NUM_IN_8PRING :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    u32 commIndex = (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) ? topoAttr_.devicePhyId : level0CommInfo.localRank;

    if (CheckCommSize(COMM_LEVEL1, commIndex + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    return HCCL_SUCCESS;
}
```

### CollAllGatherAivRdmaExecutor::SelectTempAlg

[coll_all_gather_aiv_rdma_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L253)

```cpp
HcclResult CollAllGatherAivRdmaExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather mesh: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllGather mesh: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
