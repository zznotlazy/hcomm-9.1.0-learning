# coll_reduce_scatter_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterExecutor::Orchestrate

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L22)

```cpp
HcclResult CollReduceScatterExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    tag_ = param.tag;
    algResResp_ = &algRes;
    const u64 count = param.GetDataCount(topoAttr_.userRank);
    const HcclDataType dataType = param.GetDataType();
    HcclResult ret = HCCL_SUCCESS;
    bool needLaunchAtTheEnd = !is310P3Common_; // 是否需要在Orchestrate()结束时launch任务
    // 图模式和单卡场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ExecMem execMem;
        execMem.count = count;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        ret = KernelRun(param, execMem);
        if (algOpContext_.opRetryHandler.isPostSync == true) {
            // post Sync
            CHK_RET(RetryPostSync(param, execMem));
        }
    } else if (topoAttr_.userRankSize == 1) {
        ExecMem execMem;
        execMem.count = count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        ret = KernelRun(param, execMem);
        needLaunchAtTheEnd = false;
    } else if (desc_.isZeroCopy) {
        // 在Level0执行KernelRun
        ExecMem execMem;
        execMem.count = count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.paramInputMem;
        ret = KernelRunIntraServerPre(param, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollReduceScatterExecutor][Orchestrate]errNo[0x%016llx]ReduceScatter executor KernelRunIntraServerPre failed",
                HCCL_ERROR_CODE(ret)), ret);
        if (algOpContext_.opRetryHandler.isPostSync == true) {
            // post Sync
            CHK_RET(RetryPostSync(param, execMem));
        }
        // 在Level1和Level2执行RunLoop
        if (topoAttr_.serverNum > 1) {
            ret = RunLoop(param, algRes); 
        } else {        // 单机场景，数据直接从UserInput搬到UserOutput
            u64 totalSize = count * SIZE_TABLE[dataType];
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(algRes.paramInputMem.ptr()) + totalSize * topoAttr_.userRank, totalSize);
            DeviceMem dstMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        }
    } else {
        if (algOpContext_.opRetryHandler.isInplacePreSync == true) {
            /*当重执行场景，UserInMem > CCLBuffer时，需要在reduce scatter算子前增加一个PreSync函数，提升重执行成功概率*/
            ExecMem execMem;
            execMem.count = count;
            execMem.inputPtr = param.inputPtr;
            execMem.outputPtr = param.outputPtr;
            execMem.inputMem = algRes.cclInputMem;
            execMem.outputMem = algRes.cclOutputMem;
            execMem.scratchMem = algRes.scratchMem;
            ret = InplaceOpSync(param, execMem);
        } else if (isReduceScatterV_) {
            ret = RunLoopV(param, algRes);
            needLaunchAtTheEnd = false;
        } else {
            ret = RunLoop(param, algRes);
            needLaunchAtTheEnd = false;
        }
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterExecutor][Orchestrate]errNo[0x%016llx]executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO("tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterExecutor::CalcLoopMaxCount

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L117)

```cpp
u64 CollReduceScatterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_INFO("[CollReduceScatterExecutor][CalcLoopMaxCount]using default maxCountPerLoop[%llu] as "
        "CCLBuffSize / (userRankSize * unitSize). rsv[%u]", maxCountPerLoop, isReduceScatterV_);
    return maxCountPerLoop;
}
```

### CollReduceScatterExecutor::RunLoop

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L140)

```cpp
HcclResult CollReduceScatterExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoop]maxCountPerLoop is zero."),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
        param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);
    HcclResult ret;
    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        } else {
            execMem.scratchMem = algRes.cclOutputMem; // 不需要申请则传入outputmem为scratchmem
        }
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoop]scratchMem address [%p]", execMem.scratchMem.ptr());

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        ret = RunLoopInner(param, reduceType, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollReduceScatterExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s]",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

        inputOffset = curSize;
        outputOffset = curSize;
    }
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        CHK_RET(RetryPostSync(param, execMem));
    }
    return HCCL_SUCCESS;
}
```

### CollReduceScatterExecutor::RunLoopInner

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L207)

```cpp
HcclResult CollReduceScatterExecutor::RunLoopInner(OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInner]In OP_BASE curCount is zero."), HCCL_E_PARA);
        
    // 不开启dma消减，且通信buffer足够大时，将user in到ccl的拷贝任务合并成一个
    const bool preloadCopyOpt = IsPreloadCopyOptimizeCondition(param, execMem);

    if (!is310P3Common_) {
        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
        bool hugeData = IsHugeData(curSize, &param);
        bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);
        bool dataSplit = false;
        u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(autoSelectedAlgTypeLevel1, param.DataDes.dataType,
            reduceType, hugeData, smallData, CopyPattern::BCOPY, dataSplit, deterministic, false, preloadCopyOpt);

        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
    }

    if (CCLMemSlice_) {
        u32 sliceNum = desc_.isZeroCopy ? topoAttr_.serverNum : topoAttr_.userRankSize;
        execMem.inputMem = execMem.inputMem.range(0, curSize * sliceNum);
        execMem.outputMem = execMem.outputMem.range(0, curSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, curSize * topoAttr_.userRankSize);
        }
    }

    // 执行
    if (!DMAReduceFlag_) {   // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
        DeviceMem dstMem;
        DeviceMem srcMem;
        if (preloadCopyOpt) {
            // 中转内存大小足够时，一次性搬完
            const u64 copySize = param.DataDes.count * unitSize * topoAttr_.userRankSize;
            dstMem = execMem.inputMem.range(0, copySize);
            srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), copySize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        } else {
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
                dstMem = execMem.inputMem.range(curSize * i, curSize);
                srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr) + param.DataDes.count * unitSize * i,
                    curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            }
        }
    }

    HcclResult ret = HCCL_SUCCESS;
    if (!desc_.isZeroCopy) {  
        ret = KernelRun(param, execMem);
    } else {
        ret = KernelRunInterServer(param, execMem);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInner]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d], preloadCopyOpt[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType, param.reduceType, preloadCopyOpt),
        ret);

    if (!DMAReduceFlag_) {
        // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
        DeviceMem srcMem = execMem.outputMem.range(0, curSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
    }
    HCCL_DEBUG("[CollReduceScatterExecutor][RunLoopInner]inputMem ptr is [%p], outputMem ptr is [%p]",
        execMem.inputMem.ptr(), execMem.outputMem.ptr());

    if (!is310P3Common_) {
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return ret;
}
```

### CollReduceScatterExecutor::RunLoopV

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L288)

```cpp
HcclResult CollReduceScatterExecutor::RunLoopV(OpParam &param, AlgResourceResponse &algRes)
{
    // 每轮loop需要重新计算counts和displs
    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsLeft = std::vector<u64>(countsPtr, countsPtr + topoAttr_.userRankSize);
    const auto *displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    auto displs = std::vector<u64>(displsPtr, displsPtr + topoAttr_.userRankSize);

    const HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);

    if (UNLIKELY(countsLeft[topoAttr_.userRank] == 0 && curOutputPtr == nullptr)) {
        // 若本rank的output count为0，此时允许curOutputPtr传入空指针，为保证后续流程正常执行，赋值为cclout的地址
        curOutputPtr = static_cast<u8 *>(algRes.cclOutputMem.ptr());
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoopV]Since the output count is 0, set curOutputPtr to "
            "ccl output[%p]", curOutputPtr);
    }
    CHK_PTR_NULL(curOutputPtr);

    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) && (dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    // 计算MaxCountPerLoop
    const u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);

    HcclResult ret;
    bool finished = false;
    while (!finished) {
        // 每个块尽可能平分，以均衡利用带宽
        auto curCounts = std::vector<u64>();
        auto curDispls = std::vector<u64>();
        finished = CalcCurCountsAndCurDispls(maxCountPerLoop, countsLeft, displs, curCounts, curDispls, unitSize);
        // 打印调测信息
        PrintCurCountAndCurDispls(curCounts, curDispls);

        OpParam curParam = param;
        curParam.VDataDes.counts = curCounts.data();
        curParam.VDataDes.displs = curDispls.data();
        curParam.VDataDes.dataType = dataType;

        ExecMem execMem;
        execMem.count = curCounts[topoAttr_.userRank];
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        if (scratchMemFlag_) {
            execMem.scratchMem = algRes.scratchMem;
        } else {
            execMem.scratchMem = algRes.cclOutputMem; // 不需要申请则传入outputmem为scratchmem
        }
        ret = RunLoopInnerV(curParam, reduceType, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollReduceScatterExecutor][RunLoopV]errNo[0x%016llx]kernel run error, tag[%s]",
            HCCL_ERROR_CODE(ret), curParam.tag.c_str()), ret);

        const auto outputSize = curCounts[topoAttr_.userRank] * unitSize;
        curOutputPtr += outputSize;
        // ReduceScatterV curInputPtr不需要偏移，input的偏移由displs计算
        HCCL_DEBUG("[CollReduceScatterExecutor][RunLoopV]kernel run, finished[%u]", finished);
    }
    return HCCL_SUCCESS;
}
```

### CollReduceScatterExecutor::RunLoopInnerV

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L356)

```cpp
HcclResult CollReduceScatterExecutor::RunLoopInnerV(OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    const auto *counts = static_cast<u64*>(param.VDataDes.counts);
    u64 count = counts[topoAttr_.userRank];
    HcclDataType dataType = param.VDataDes.dataType;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = count * unitSize; // 单位：字节;

    /* 设置子图复用标志 */
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(curSize, &param);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForReduceScatterV(autoSelectedAlgTypeLevel1,
        dataType, reduceType, hugeData, false, CopyPattern::BCOPY, false, deterministic);

    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (CCLMemSlice_) {
        const u64 inputCounts = std::accumulate(counts, counts + topoAttr_.userRankSize, 0ULL);
        execMem.inputMem = execMem.inputMem.range(0, inputCounts * unitSize);
        execMem.outputMem = execMem.outputMem.range(0, curSize);
        if (scratchMemFlag_) {
            execMem.scratchMem = execMem.scratchMem.range(0, inputCounts * unitSize);
        }
    }

    // 执行
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterExecutor][RunLoopInnerV]errNo[0x%016llx]kernel run error, tag[%s], "
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]", HCCL_ERROR_CODE(ret),
        param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(), execMem.count, dataType, param.reduceType),
        ret);

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return ret;
}
```

### CollReduceScatterExecutor::PrepareAivBuffers

[coll_reduce_scatter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_executor.cc#L507)

```cpp
HcclResult CollReduceScatterExecutor::PrepareAivBuffers(u32 rankSize, u32 rankId, u32 rankOffset,
    DeviceMem &inputMem, DeviceMem &outputMem, std::vector<LINK> &links, void **dataBuffers, void **flagBuffers,
    UserMemType dataMemType, UserMemType flagMemType, u32 dataMemOffset, u32 flagMemOffset)
{
    void *tmpCCLBufferData = nullptr;
    void *tmpCCLBufferFlag = nullptr;
    for (u32 i = 0; i < rankSize; i++) {
        if (i != rankId) {
            if (links[i + rankOffset] != nullptr) {
                CHK_RET(links[i + rankOffset]->GetRemoteMem(dataMemType, &(tmpCCLBufferData)));
                CHK_RET(links[i + rankOffset]->GetRemoteMem(flagMemType, &(tmpCCLBufferFlag)));
                dataBuffers[i] = static_cast<u8 *>(tmpCCLBufferData) + dataMemOffset;
                flagBuffers[i] = static_cast<u8 *>(tmpCCLBufferFlag) + flagMemOffset;
            }
        } else {
            dataBuffers[i] = static_cast<u8 *>(inputMem.ptr()) + dataMemOffset;
            flagBuffers[i] = static_cast<u8 *>(outputMem.ptr()) + flagMemOffset;
        }
    }
    return HCCL_SUCCESS;
}
```
