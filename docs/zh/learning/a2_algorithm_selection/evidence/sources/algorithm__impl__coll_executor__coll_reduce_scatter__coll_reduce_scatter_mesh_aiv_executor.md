# coll_reduce_scatter_mesh_aiv_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterMeshAivExecutor::CalcStreamNum

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L23)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流
    HCCL_INFO("[CollReduceScatterMeshAivExecutor][CalcStreamNum] tag[%s] streamNum[%u].",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::CalcCommInfo

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L31)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::CalcTransportMemType

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L40)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }
    HCCL_INFO("[CollReduceScatterMeshAivExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d].", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::CalcLevel0CommInfo

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L55)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::CalNumBlocks

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L65)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::CalNumBlocks(u32& numBlocks, u32 rankSize, u64 dataSize, HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV

    bool isOpBase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && !isOpBase) {
        numBlocks = rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3 > MAX_NUM_BLOCKS ?
            rankSize * NUM_BLOCKS_THREE_PER_RANK_A3 : rankSize * NUM_BLOCKS_FOUR_PER_RANK_A3;
    } else if (isOpBase) {
        numBlocks = NUM_BLOCKS_FACTOR_TWO * rankSize; // 单机场景，单算子ReduceScatter大数据使用2倍 rankSize个aiv
    }

    u32 bestNumBlocks = numBlocks;
    CHK_PRT_RET(numBlocks_ < rankSize && topoAttr_.deviceType == DevType::DEV_TYPE_910_93,
        HCCL_WARNING("[CollReduceScatterMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
        numBlocks_, rankSize), HCCL_E_PARA);
    
    CHK_PRT_RET(numBlocks_ < bestNumBlocks && (topoAttr_.deviceType != DevType::DEV_TYPE_910_93 || isOpBase),
        HCCL_WARNING("[CollReduceScatterMeshAivExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
        numBlocks_, bestNumBlocks), HCCL_E_PARA);

    if (numBlocks_ < bestNumBlocks) {
        numBlocks = numBlocks_ / rankSize * rankSize;
    }

    HCCL_INFO("[CollReduceScatterMeshAivExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommanded[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::GetAivExecParam

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L95)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::GetAivExecParam(const OpParam& param, AlgResourceResponse& algRes, AivSuperKernelArgs &args)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
 
    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
 
    // 单算子大数据量
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.aivOutputMem;

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
 
    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollReduceScatterMeshAivExecutor][GetAivExecParam] userRank [%d] localRank [%d]",
        topoAttr_.userRank, localRank);
 
    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(args.buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(args.buffersOut[i])));
        } else {
            args.buffersIn[i] = execMem.inputMem.ptr();
            args.buffersOut[i] = execMem.outputMem.ptr();
        }
    }
    HCCL_INFO("SPK, buffersIn [%p] [%p] [%p] [%p]"
        "buffersOut [%p] [%p] [%p] [%p]", args.buffersIn[0], args.buffersIn[1], args.buffersIn[2], args.buffersIn[3],
        args.buffersOut[0], args.buffersOut[1], args.buffersOut[2], args.buffersOut[3]);
    args.rank = localRank;
    args.rankSize = localRankSize;
    args.len = execMem.count;
    args.dataType = param.DataDes.dataType;
    args.unitSize = SIZE_TABLE[param.DataDes.dataType];
    args.reduceOp = param.reduceType;
    args.devType = static_cast<u32>(topoAttr_.deviceType);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
    HCCL_INFO("SPK [CollReduceScatterMeshAivExecutor][GetAivExecParam], rank[%llu], rankSize[%llu], len[%llu],datatype[%llu], op[%llu], devType[%u]",
        args.rank, args.rankSize, args.len, args.dataType, args.reduceOp, args.devType);

    HCCL_INFO("tag[%s], ReduceScatter executor getalgexecparam success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::Orchestrate

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L148)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
 
    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.aivOutputMem;
        ret = KernelRun(param, execMem);
    } else {
        execMem.inputMem = algRes.cclInputMem; // 仍然使用CCLIN
        execMem.outputMem = algRes.aivOutputMem;
        ret = KernelRun(param, execMem);
    }
 
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterMeshAivExecutor][Orchestrate]errNo[0x%016llx] tag[%s] executor kernel "
            "run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
 
    HCCL_INFO("tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshAivExecutor::KernelRun

[coll_reduce_scatter_mesh_aiv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L179)

```cpp
HcclResult CollReduceScatterMeshAivExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] ReduceScatter aiv enter.", __func__);
 
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];
 
    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollReduceScatterMeshAivExecutor][KernelRun] userRank [%d] localRank [%d]", topoAttr_.userRank, localRank);
 
    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(level0CommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
    }
    u32 numBlocks;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_DEBUG("[CollReduceScatterMeshAivExecutor][KernelRun]isOpbase is %d", isOpbase);

    AivOpArgs opArgs {
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, execMem.inputPtr, execMem.outputPtr, execMem.count,
            param.DataDes.dataType, param.reduceType, 0, isOpbase
    };
    AivTopoArgs topoArgs { localRank, localRankSize, MAX_RANK_SIZE, 0, 1, topoAttr_.deviceType};
    topoArgs.identify = algoAttr_.identifier;
    CHK_PRT_RET(CalNumBlocks(numBlocks, localRankSize) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__),
        HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs {
        param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size(), numBlocks_, param.aivTag
    };
    struct AivProfilingInfo aivProfilingInfo;
    AivAlgArgs algArgs {};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    aivProfilingInfo.counter = opCounter_;

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterMeshAivExecutor][KernelRun]ReduceScatter aiv failed, return[%d]", ret),
        ret);

    ExtraArgs extraArgs;
    CHK_RET(SetOpCache(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo, false));
 
    HCCL_INFO("[CollReduceScatterMeshAivExecutor][KernelRun]ReduceScatter aiv run success.");
    return HCCL_SUCCESS;
}
```
