# coll_reduce_scatter_deter_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterDeterExecutor::CalcLoopMaxCount

[coll_reduce_scatter_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_executor.cc#L112)

```cpp
u64 CollReduceScatterDeterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop;
    bool isLocalReduce91073 = ((((topoAttr_.userRankSize & (topoAttr_.userRankSize - 1)) != 0) ||
        aicpuUnfoldMode_ || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) 
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93)) && (topoAttr_.serverNum == 1);

    bool isLocalReduce910B = ((totalSize_ > HCCL_SMALL_COUNT_32_KB) ||
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ||
        ((topoAttr_.deviceNumPerAggregation != DEVICE_EIGHT) && (topoAttr_.deviceNumPerAggregation != DEVICE_FOUR)))&&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B);
    if (isLocalReduce91073 || isLocalReduce910B) { 
        maxCountPerLoop = (inCCLbufferSize_ - HCCL_MIN_SLICE_ALIGN_910B * topoAttr_.deviceNumPerAggregation) /
            unitSize / (topoAttr_.deviceNumPerAggregation - 1);
        maxCountPerLoop = maxCountPerLoop / HCCL_MIN_SLICE_ALIGN_910B;
        maxCountPerLoop = maxCountPerLoop * HCCL_MIN_SLICE_ALIGN_910B;
    } else {
        const u32 base = 2;
        maxCountPerLoop = inCCLbufferSize_ * base / topoAttr_.userRankSize;
    }
    return maxCountPerLoop;
}
```

### CollReduceScatterDeterExecutor::KernelRun

[coll_reduce_scatter_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_executor.cc#L153)

```cpp
HcclResult CollReduceScatterDeterExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    CommPlane commPlane = COMM_LEVEL0;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93 && topoAttr_.serverNum != 1) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};

    bool isLocalReduce91073 = ((((topoAttr_.userRankSize & (topoAttr_.userRankSize - 1)) != 0) ||
        aicpuUnfoldMode_ || (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) 
        && (topoAttr_.deviceType == DevType::DEV_TYPE_910_93)) && (topoAttr_.serverNum == 1);

    bool isLocalReduce910B = ((param.DataDes.count * unitSize > HCCL_SMALL_COUNT_32_KB) ||
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ||
        ((topoAttr_.deviceNumPerAggregation != DEVICE_EIGHT) && (topoAttr_.deviceNumPerAggregation != DEVICE_FOUR)))&&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B);

    if (isLocalReduce91073 || isLocalReduce910B) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_LOCAL_REDUCE, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_LOCAL_REDUCE in COMM_COMBINE_ORDER", __func__);
    } else {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_HDSTAGE in COMM_COMBINE_ORDER", __func__);
    }

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0,
        reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, &opInfo));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("ReduceScatter mesh deter run success");
    return HCCL_SUCCESS;
}
```
