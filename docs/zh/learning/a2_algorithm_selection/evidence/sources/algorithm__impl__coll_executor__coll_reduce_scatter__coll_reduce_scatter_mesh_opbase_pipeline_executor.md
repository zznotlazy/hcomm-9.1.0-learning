# coll_reduce_scatter_mesh_opbase_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterMeshOpbasePipelineExecutor::RunLoop

[coll_reduce_scatter_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc#L23)

```cpp
HcclResult CollReduceScatterMeshOpbasePipelineExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop] begins.");

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(unitSize);
    HCCL_DEBUG("[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop "
        "is [%llu].", param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop);

    // 先获取 comm level0 \ comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem userInMem = DeviceMem::create(param.inputPtr, param.DataDes.count * unitSize);
    u64 reduceAttr = GetReduceAttr(userInMem, const_cast<DeviceMem&>(algRes.cclInputMem), param.DataDes.dataType,
        param.reduceType); // scratchMem用的cclin
    u64 bufferSize = algRes.cclInputMem.size();

    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);

    for (u64 countLeft = param.DataDes.count, curCount = 0, curOffset = 0, curSize = 0; countLeft > 0;
        countLeft -= curCount) {
        curInputPtr += curSize;
        curOutputPtr += curSize;

        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        curSize = curCount * unitSize;

        HCCL_CONFIG_DEBUG(HCCL_ALG, "[CollReduceScatterMeshOpbasePipelineExecutor][RunLoop]tag[%s], curOffset[%llu]," \
            "curInputPtr[%p], curOutputPtr[%p], curCount[%llu], dataType[%d].",
            param.tag.c_str(), curOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        bool hugeData = IsHugeData(curSize);
        u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
        auto meta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            hugeData, false, CopyPattern::BCOPY, false, deterministic, false);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));

        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;

        std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(tempAlg);

        HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
            param.root, param.reduceType};

        CHK_RET(tempAlg->Prepare(&opInfo, execMem.inputMem, curCount, bufferSize, curOffset, level0CommInfo,
            level1CommInfo, const_cast<Stream&>(param.stream), algResResp_->slaveStreams, algResResp_->notifiesMain,
            algResResp_->notifiesAux, reduceAttr));
        CHK_RET(tempAlg->RunAsync());

        CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

        curOffset += curSize;
    }

    return HCCL_SUCCESS;
}
```

### CollReduceScatterMeshOpbasePipelineExecutor::CalcLoopMaxCount

[coll_reduce_scatter_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc#L158)

```cpp
u64 CollReduceScatterMeshOpbasePipelineExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = ((inCCLbufferSize_ / (HCCL_MIN_SLICE_ALIGN_910B * PIPELINE_DEPTH)) \
            * HCCL_MIN_SLICE_ALIGN_910B - HCCL_MIN_SLICE_ALIGN_910B) / unitSize;
    HCCL_INFO("[CollReduceScatterMeshOpbasePipelineExecutor][CalcLoopMaxCount] maxCountPerLoop[%llu]", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollReduceScatterMeshOpbasePipelineExecutor::SelectTempAlg

[coll_reduce_scatter_mesh_opbase_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc#L186)

```cpp
HcclResult CollReduceScatterMeshOpbasePipelineExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
    }
    return HCCL_SUCCESS;
}
```
