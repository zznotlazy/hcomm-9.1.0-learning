# coll_all_reduce_mesh_small_count_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshSmallCountExecutor::Orchestrate

[coll_all_reduce_mesh_small_count_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_small_count_executor.cc#L113)

```cpp
HcclResult CollAllReduceMeshSmallCountExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    ParseParam(param);
    algResResp_ = &algRes;
    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.paramInputMem;
        execMem.outputMem = algRes.paramOutputMem;
        execMem.scratchMem = algRes.scratchMem;
    } else {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
    }
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceMeshSmallCountExecutor][Orchestrate]errNo[0x%016llx]executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (!is310P3Common_) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_,
            const_cast<Stream &>(param.stream),
            const_cast<std::vector<Stream> &>(algResResp_->slaveStreams)));
    }

    HCCL_INFO("[CollAllReduceMeshSmallCountExecutor]tag[%s], AllReduce executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshSmallCountExecutor::KernelRun

[coll_all_reduce_mesh_small_count_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_small_count_executor.cc#L157)

```cpp
HcclResult CollAllReduceMeshSmallCountExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    if (!CalcScratchMemFlag(totalSize_)) {
        execMem.scratchMem = execMem.outputMem;
    }

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    ReduceType reduceType = ((param.reduceType != HCCL_REDUCE_PROD) &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
        true, 1, false, CopyPattern::BCOPY, 1, false, true, false, deterministic);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    bool isUsedRegister = false;
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        bool aicpu = true;
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_HD_OPTIM, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_HD_OPTIM in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
            level0CommInfo.localRank, &opInfo, aicpu));
    } else if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE) {
        isUsedRegister = true;
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_REDUCE_BCAST, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_REDUCE_BCAST in COMM_LEVEL0", __func__);
    } else if (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT) {
        if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || aicpuUnfoldMode_) {
            level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING, 
                dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_DOUBLING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(level0TempAlg);
            CHK_RET(level0TempAlg->Prepare(reduceAttr));
        } else {
            level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_DOUBLING_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(level0TempAlg);
            CHK_RET(level0TempAlg->Prepare(reduceAttr, &opInfo));
        }
    } else {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_LOCAL_REDUCE_BCAST, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_LOCAL_REDUCE_BCAST in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
            level0CommInfo.localRank, level0CommInfo.localRankSize, topoAttr_.userRank, &opInfo));
    }
    CHK_SMART_PTR_NULL(level0TempAlg);

    if (isUsedRegister) {
        PrepareData prepareData;
        prepareData.reduceAttr = reduceAttr;
        prepareData.subStreamsPtr = &algResResp_->slaveStreams;
        prepareData.signalPtr = &algResResp_->notifiesMain;
        prepareData.signalAuxPtr = &algResResp_->notifiesAux;
        prepareData.interRank = level0CommInfo.localRank;
        prepareData.interRankSize = level0CommInfo.localRankSize;
        prepareData.userRank = topoAttr_.userRank;
        prepareData.opInfo = &opInfo;

        prepareData.inputMem = execMem.inputMem;
        prepareData.outputMem = execMem.outputMem;
        prepareData.scratchMem = execMem.scratchMem;
        prepareData.count = execMem.count;
        prepareData.dataType = param.DataDes.dataType;
        prepareData.stream = param.stream;
        prepareData.reductionOp = param.reduceType;
        prepareData.slicesPtr = &dataSegsSlice;

        CHK_RET(level0TempAlg->Prepare(prepareData));
    } else {
        CHK_RET(level0TempAlg->Prepare(execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));
    }

    CHK_RET(
        level0TempAlg->RegisterProfiler(
            (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("AllReduce small count executor run success.");
    return HCCL_SUCCESS;
}
```
