# coll_scatter_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollScatterExecutor::RunLoop

[coll_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc#L53)

```cpp
HcclResult CollScatterExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    auto dataType = param.DataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    RankId root = param.root;

    auto totalRecvCount = param.DataDes.count;

    u8 *curUserInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curUserOutputPtr = static_cast<u8 *>(param.outputPtr);
    if (topoAttr_.userRank == root) {
        CHK_PTR_NULL(curUserInputPtr);
    }
    CHK_PTR_NULL(curUserOutputPtr);

    auto inCCLbuffer = algRes.cclInputMem;
    auto outCCLbuffer = algRes.cclOutputMem;

    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbuffer.size() / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_DEBUG("[CollScatterExecutor][RunLoop]tag[%s], userRankSize is [%u], root is [%u], "
               "maxCountPerLoop is [%llu], totalRecvCount is [%llu]",
        tag_.c_str(), topoAttr_.userRankSize, root, maxCountPerLoop, totalRecvCount);

    for (u64 countLeft = totalRecvCount, curRecvCount = 0, inputOffset = 0, outputOffset = 0;
        countLeft > 0; countLeft -= curRecvCount) {
        curUserInputPtr += inputOffset;
        curUserOutputPtr += outputOffset;

        // 判断剩余数据量对应的input size是否大于中转input size
        curRecvCount =
            ((countLeft * unitSize * topoAttr_.userRankSize) > inCCLbuffer.size()) ? maxCountPerLoop : countLeft;
        CHK_PRT_RET((curRecvCount == 0), HCCL_ERROR("[RunLoop][Scatter]In OP_BASE curRecvCount is zero"), HCCL_E_PARA);
        u64 curRecvSize = curRecvCount * unitSize;               // 单位：字节
        u64 curSendSize = topoAttr_.userRankSize * curRecvSize;  // 单位：字节

        DeviceMem curCCLInputMem(inCCLbuffer.ptr(), curSendSize);
        DeviceMem curCCLOutputMem(outCCLbuffer.ptr(), curRecvSize);

        ExecMem execMem;
        execMem.count = curRecvCount;
        execMem.inputMem = curCCLInputMem;
        execMem.outputMem = curCCLOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curUserInputPtr;
        execMem.outputPtr = curUserOutputPtr;

        HCCL_DEBUG("[RunLoop][Scatter] ScatterLoop: inputOffset[%llu], outputOffset[%llu], "
            "curUserInputPtr[%p], curUserOutputPtr[%p], curRecvCount[%llu], curRecvSize[%llu], "
            "curSendSize[%llu], inCCLbuffer.ptr[%p], outCCLbuffer.ptr[%p]",
            inputOffset, outputOffset, curUserInputPtr, curUserOutputPtr, curRecvCount, curRecvSize,
            curSendSize, inCCLbuffer.ptr(), outCCLbuffer.ptr());

        CHK_RET(RunLoopInner(param, execMem, algRes));

        inputOffset = curRecvSize;
        outputOffset = curRecvSize;
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}
```

### CollScatterExecutor::RunLoopInner

[coll_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc#L117)

```cpp
HcclResult CollScatterExecutor::RunLoopInner(OpParam &param, ExecMem &execMem, AlgResourceResponse &algRes)
{
    auto dataType = param.DataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    RankId root = param.root;

    auto totalRecvCount = param.DataDes.count;
    u64 totalRecvSize = totalRecvCount * unitSize;

    u64 recvSize = execMem.outputMem.size();

    auto meta = HcclOpMetaInfo::GetOneForScatter(root, IsHugeData(execMem.outputMem.size()));
    CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));

    DeviceMem dstMem;
    DeviceMem srcMem;
    if (topoAttr_.userRank == root) {
        // 本rank为root节点，非root节点不需要拷贝到中转内存
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为totalRecvSize
            srcMem = DeviceMem::create((u8*)execMem.inputPtr + totalRecvSize * i, recvSize);
            dstMem = algRes.cclInputMem.range(recvSize * i, recvSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        }
    }

    if (recvSize % HCCL_MIN_SLICE_ALIGN != 0) {
        // 不支持内存不对齐的轮次
        DMAReduceFlag_ = false;
    }

    /* 入参的正确性由HCCL确保 */
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterExecutor][RunLoop]errNo[0x%016llx] OP_BASE hcclComm scatter error, tag[%s], "
                   "input_ptr[%p], output_ptr[%p], recvSize[%llu], data_type[%d], root[%u]",
            HCCL_ERROR_CODE(ret), tag_.c_str(), algRes.cclInputMem.ptr(), algRes.cclOutputMem.ptr(),
            recvSize, dataType, root),
        ret);

    // 将 CCLOut 上的数据搬运到 userOut
    if (!DMAReduceFlag_) {
        srcMem = algRes.cclOutputMem.range(0, recvSize);
        dstMem = DeviceMem::create(execMem.outputPtr, recvSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
    }
    return HCCL_SUCCESS;
}
```

### CollScatterExecutor::KernelRunLevel1

[coll_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc#L197)

```cpp
HcclResult CollScatterExecutor::KernelRunLevel1(DeviceMem& inputMem, u64 count, HcclDataType dataType,
    u32 &commIndex, u32 root, u32 &subRoot, CommPlane commLevel, Stream& stream)
{
    CHK_RET(CheckCommSize(commLevel, commIndex + 1));
    SubCommInfo subCommInfo = GetSubCommInfo(commLevel, commIndex);

    u32 subCommSize = subCommInfo.localRankSize;

    if (subCommSize <= 1 || subRoot != topoAttr_.userRank) {
        HCCL_INFO("[Scatter][KernelRunLevel1]: no need to run intra-server, subCommSize[%u], subRoot[%u]," \
            "userRank[%u]", subCommSize, subRoot, topoAttr_.userRank);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[Scatter][KernelRunLevel1]: start to run intra-server, subCommSize[%u], subRoot[%u]," \
        "userRank[%u]", subCommSize, subRoot, topoAttr_.userRank);

    u32 rootRankLevel1 = 0;
    CHK_RET(GetRankByUserRank(commLevel, commIndex, root, rootRankLevel1));

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        // server间NB算法走NB
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_NB in COMM_LEVEL1", __func__);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_NHR, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_NHR in COMM_LEVEL1", __func__);
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_SCATTER_RING in COMM_LEVEL1", __func__);
    }

    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(inputMem, inputMem, inputMem, count * topoAttr_.userRankSize,
        dataType, stream, HCCL_REDUCE_RESERVED, rootRankLevel1, std::vector<Slice>(0))); // count是output的数据个数
    CHK_RET(level1TempAlg->RegisterProfiler(
        (subCommSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(level1TempAlg, subCommInfo));

    return HCCL_SUCCESS;
}
```

### CollScatterExecutor::Orchestrate

[coll_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc#L243)

```cpp
HcclResult CollScatterExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    HcclResult ret = HCCL_SUCCESS;
    bool needLaunchAtTheEnd = true; // 是否需要在Orchestrate()结束时launch任务
    // 图模式和单卡场景下不需要Loop
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
    } else if (topoAttr_.userRankSize == 1) {
        ret = KernelRun(param, execMem);
        needLaunchAtTheEnd = false;
    } else {
        ret = RunLoop(param, algRes);
        needLaunchAtTheEnd = false;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollScatterExecutor][Orchestrate]errNo[0x%016llx]Scatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO("tag[%s] Scatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```
