# coll_reduce_scatter_v_deter_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterVDeterExecutor::CalcLoopMaxCount

[coll_reduce_scatter_v_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L44)

```cpp
u64 CollReduceScatterVDeterExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop ;
    if(scratchMemFlag_) {
        maxCountPerLoop = maxCount_;
    } else {
        maxCountPerLoop = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    }
    HCCL_INFO("[CollReduceScatterVDeterExecutor][CalcLoopMaxCount] maxCountPerLoop = [%llu] .", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollReduceScatterVDeterExecutor::RunReduceScattervLevel0

[coll_reduce_scatter_v_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L171)

```cpp
HcclResult CollReduceScatterVDeterExecutor::RunReduceScattervLevel0(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    CHK_RET(ActiveSlaveStreams(param.stream));
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    u32 level0RankSize = level0CommInfo.localRankSize;

    const auto curCounts = static_cast<u64*>(param.VDataDes.counts);
    const auto curDispls = static_cast<u64*>(param.VDataDes.displs);
    GroupSlicesInfo groupSlicesInfoLevel0;
    for (u32 groupId = 0; groupId < topoAttr_.moduleNum; groupId++) {
        MemBlockInfo memInfo;
        u32 groupSlicesOffset = groupId * level0RankSize ;
        for (u32 localRankId = 0; localRankId < level0RankSize; localRankId++) {
            u64 size = curCounts[localRankId + groupSlicesOffset] * unitSize;
            u64 userMemInOffset = curDispls[localRankId + groupSlicesOffset] * unitSize;
            
            memInfo.size.push_back(size);
            memInfo.userInputOffsets.push_back(userMemInOffset);
            memInfo.inputOffsets.push_back(minBiasOffset_ * unitSize * (localRankId + groupSlicesOffset));
            memInfo.outputOffsets.push_back(minBiasOffset_ * unitSize * (localRankId + groupSlicesOffset));
        }
        groupSlicesInfoLevel0.push_back(memInfo);
    }

    all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;  // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数） 
    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);

    // execMem.scratchMem在单算子模式下为cclout，图模式为scrach，因此output传入scrach即可
    CHK_RET(level0TempAlg->Prepare(execMem.inputPtr, execMem.inputMem, execMem.scratchMem, param.stream, 
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        groupSlicesInfoLevel0, param.reduceType, all2allOffset_, dataType, isNeedSpaceBorrow_)); 

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVDeterExecutor::RunReduceScattervLevel1ForMeshTopo

[coll_reduce_scatter_v_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L214)

```cpp
HcclResult CollReduceScatterVDeterExecutor::RunReduceScattervLevel1ForMeshTopo(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    u32 level0RankId = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0RankId + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0RankId);
    u32 level0Ranksize = level0CommInfo.localRankSize;
    
    // 切分数据，记录每组的起始偏移和大小（仅1组）
    auto unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u32 inputBaseIndex = (all2allOffset_ + level0RankId) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数） 
    u32 level1Ranksize = level1CommInfo.localRankSize;
    const auto curCounts = static_cast<u64*>(param.VDataDes.counts);
    MemBlockInfo memInfo;
    for (u32 localRank = 0; localRank < level1Ranksize; localRank++) {
        u64 inputIndex = inputBaseIndex + localRank * level0Ranksize;
        u64 outputIndex = level0RankId + localRank * level0Ranksize;
        u64 size = curCounts[level0RankId +  localRank * level0Ranksize] * unitSize;
 
        memInfo.userInputOffsets.push_back(minBiasOffset_ * unitSize * outputIndex);
        memInfo.inputOffsets.push_back(minBiasOffset_ * unitSize* inputIndex);
        memInfo.outputOffsets.push_back(minBiasOffset_ * unitSize * outputIndex);
        memInfo.size.push_back(size);
    }

    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.scratchMem,
        param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        memInfo, param.reduceType, param.VDataDes.dataType, level0RankId == level0LastRank - 1,
        level0RankId == level0LastRank, isNeedSpaceBorrow_));

    CHK_RET(level1TempAlg->RegisterProfiler((level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        level0CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVDeterExecutor::RunReduceScattervLevel1

[coll_reduce_scatter_v_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L273)

```cpp
HcclResult CollReduceScatterVDeterExecutor::RunReduceScattervLevel1(const OpParam &param, ExecMem &execMem,
    const SubCommInfo &level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    HcclDataType dataType = param.VDataDes.dataType;

    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    HCCL_DEBUG("RunReduceScattervLevel1 begin");
    /* ******************第一步: 机间reducescatter *******************************/
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HCCL_INFO("reducescatterv mesh: using ring algo inter-server");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nonuniform-bruck algo inter-server");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr)); 
    } else { 
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nhr algo inter-server");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
        level1TempAlg->CloseBarrier();
    }

    std::vector<Slice> slices;
    CHK_RET(CalReduceScatterVSliceData(param, level0RankSize, level1RankSize, slices));
  
    CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.inputMem, 0,
        dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, slices));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVDeterExecutor::KernelRun

[coll_reduce_scatter_v_deter_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L323)

```cpp
HcclResult CollReduceScatterVDeterExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][CollReduceScatterVDeterExecutor] ReduceScatterV deter run start, tag[%s]", __func__, tag_.c_str());
    
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    auto unitSize = SIZE_TABLE[param.VDataDes.dataType];
    const auto curCounts = static_cast<u64*>(param.VDataDes.counts);
    const auto curDispls = static_cast<u64*>(param.VDataDes.displs);
    u64 dataSize = execMem.count * unitSize;
    DeviceMem srcMem;

    if (isMeshTopo_) {
        u64 maxCount = *std::max_element(curCounts, curCounts + topoAttr_.userRankSize);
        u64 maxCountPerloop = CalcLoopMaxCount(unitSize);
        minBiasOffset_ = maxCount < maxCountPerloop ? maxCount : maxCountPerloop;
        // L0 节点内 reduce scatter v
        CHK_RET(RunReduceScattervLevel0(param, execMem, level0CommInfo));
        // L1 节点间 reduce scatter v
        if (topoAttr_.moduleNum > 1) {
            CHK_RET(RunReduceScattervLevel1ForMeshTopo(param, execMem, level0CommInfo));
        }
        srcMem = execMem.scratchMem.range(minBiasOffset_ * topoAttr_.userRank * unitSize, dataSize);// Opbase: CO/Sr->UO
    } else { // 处理 Nx1 场景的图模式
        CHK_RET(RunReduceScattervLevel1(param, execMem, level0CommInfo));
        srcMem = execMem.inputMem.range(curDispls[topoAttr_.userRank] * unitSize, dataSize);// Offload:UI->UO
    }

    Stream stream = param.stream;
    DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
    HCCL_CONFIG_INFO(HCCL_ALG,"[%s]ReduceScatterV deter run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;    
}
```
