# coll_reduce_scatter_deter_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterDeterPipelineExecutor::CalcLoopMaxCount

[coll_reduce_scatter_deter_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_pipeline_executor.cc#L92)

```cpp
u64 CollReduceScatterDeterPipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，这里每一块的地址不要求128byte对齐
    u64 maxCountPerLoop = ((inCCLbufferSize_ / topoAttr_.userRankSize) - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollReduceScatterDeterPipelineExecutor::RunLoop

[coll_reduce_scatter_deter_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_pipeline_executor.cc#L100)

```cpp
HcclResult CollReduceScatterDeterPipelineExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterDeterPipelineExecutor][RunLoop] tag[%s], userRank[%u] begins.",
        tag_.c_str(), topoAttr_.userRank);

    CHK_PRT_RET((param.reduceType == HCCL_REDUCE_PROD) || (param.DataDes.dataType == HCCL_DATA_TYPE_INT64),
        HCCL_ERROR("[CollReduceScatterDeterPipelineExecutor] unsupported reduceType[%u] or unsupported dataType[%u]",
        param.reduceType, param.DataDes.dataType), HCCL_E_INTERNAL);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop "
        "is [%llu].", tag_.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();

    for (u64 countLeft = param.DataDes.count, curCount = 0, curSize = 0; countLeft > 0;
        countLeft -= curCount) {
        curInputPtr += curSize;
        curOutputPtr += curSize;

        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        curSize = curCount * unitSize;

        HCCL_CONFIG_DEBUG(HCCL_ALG, "[CollReduceScatterDeterPipelineExecutor][RunLoop]tag[%s], curOffset[%llu]," \
            "curInputPtr[%p], curOutputPtr[%p], curCount[%llu], dataType[%d].",
            tag_.c_str(), curOffset_, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        constexpr s64 HCCL_MEDIUM_COUNT_2_MB = 2 * 1024 * 1024;
        bool smallData = curSize < HCCL_MEDIUM_COUNT_2_MB ? 1 : 0;
        bool hugeData = IsHugeData(curSize);
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(autoSelectedAlgTypeLevel1, param.DataDes.dataType,
            ReduceType::INLINE_REDUCE, hugeData, smallData, CopyPattern::ZCOPY, false, deterministic, false);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey(), true));
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.cclOutputMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        CHK_RET(KernelRun(param, execMem));

        CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream),
            const_cast<std::vector<Stream> &>(algResResp_->slaveStreams)));

        curOffset_ += curSize;
    }
    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][RunLoop] tag[%s], userRank[%u] run loop success.",
        tag_.c_str(), topoAttr_.userRank);

    return HCCL_SUCCESS;
}
```

### CollReduceScatterDeterPipelineExecutor::KernelRun

[coll_reduce_scatter_deter_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_pipeline_executor.cc#L196)

```cpp
HcclResult CollReduceScatterDeterPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] starts.",
        tag_.c_str(), topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::vector<Slice> bufferSlices; // 数据分成ranksize份，每份的起始偏移和大小
    CHK_RET(PrepareDataSlice(param, execMem, level0CommInfo, level1CommInfo, bufferSlices));

    CHK_RET(ActiveSlaveStreams(param.stream));

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MULTI_DETERMINISTIC_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};

    CHK_RET(tempAlg->Prepare(&opInfo, execMem.scratchMem, execMem.count, curOffset_, bufferSlices, level0CommInfo,
        level1CommInfo, const_cast<Stream&>(param.stream), algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollReduceScatterDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] run success.",
        tag_.c_str(), topoAttr_.userRank);
    return HCCL_SUCCESS;
}
```
