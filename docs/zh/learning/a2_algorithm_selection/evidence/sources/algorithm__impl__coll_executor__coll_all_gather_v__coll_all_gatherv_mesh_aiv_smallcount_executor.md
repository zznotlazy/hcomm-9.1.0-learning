# coll_all_gatherv_mesh_aiv_smallcount_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherVExecutor

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L17)

```cpp
: CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::CalcCommInfo

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L22)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::CalcTransportMemType

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L31)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    inputType = TransportMemType::AIV_INPUT;
    outputType = TransportMemType::AIV_OUTPUT;
    HCCL_INFO("[CollAllGatherVMeshAivSmallCountExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::CalcLevel0CommInfo

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L40)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::CalNumBlocks

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L50)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(numBlocks_ < numBlocks,
        HCCL_WARNING("[CollAllGatherVMeshAivSmallCountExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
        numBlocks_, numBlocks), HCCL_E_PARA);

    HCCL_INFO("[CollAllGatherVMeshAivSmallCountExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommanded[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::Orchestrate

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L64)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;

    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.count = (static_cast<const u64 *>(param.VDataDes.counts))[topoAttr_.userRank];

    execMem.inputMem = algRes.aivInputMem;
    execMem.outputMem = algRes.aivOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherVMeshAivSmallCountExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], CollAllGatherVMeshAivSmallCountExecutor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAllGatherVMeshAivSmallCountExecutor::KernelRun

[coll_all_gatherv_mesh_aiv_smallcount_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L90)

```cpp
HcclResult CollAllGatherVMeshAivSmallCountExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherVMeshAivSmallCountExecutor][KernelRun]AllGatherV aiv enter.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[CollAllGatherVMeshAivSmallCountExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    ExtraArgs extraArgs;
    for (u32 i = 0; i < localRankSize; i++) {
        extraArgs.recvCounts[i] = *(static_cast<const u64 *>(param.VDataDes.counts) + i);
        extraArgs.recvDispls[i] = *(static_cast<const u64 *>(param.VDataDes.displs) + i);
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
        if (extraArgs.recvCounts[i] > extraArgs.maxCount) {
            extraArgs.maxCount = extraArgs.recvCounts[i];
        }
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    AivOpArgs opArgs {
        HcclCMDType::HCCL_CMD_ALLGATHER_V, execMem.inputPtr, execMem.outputPtr, extraArgs.maxCount,
        param.VDataDes.dataType, param.reduceType, param.root, isOpbase
    };
    AivTopoArgs topoArgs { localRank, localRankSize };
    u32 numBlocks;
    CHK_PRT_RET(CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    topoArgs.identify = algoAttr_.identifier;
    AivResourceArgs resourceArgs {
        param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(), numBlocks_, param.aivTag
    };
    AivAlgArgs algArgs {};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherVMeshAivSmallCountExecutor][KernelRun]AllGatherV aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollAllGatherVMeshAivSmallCountExecutor][KernelRun]AllGatherV aiv run success.");
    return HCCL_SUCCESS;
}
```
