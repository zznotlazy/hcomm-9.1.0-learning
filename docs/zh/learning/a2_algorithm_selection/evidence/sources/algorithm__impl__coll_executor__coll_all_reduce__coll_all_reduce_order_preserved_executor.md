# coll_all_reduce_order_preserved_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel0

[coll_all_reduce_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L146)

```cpp
HcclResult CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel0(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    if (level0CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;
        HCCL_INFO("[%s] single rank per module, skip L0 AllToAll and LocalReduce, tag[%s]",
            __func__, tag_.c_str());
        CHK_RET(RunReduceScatterLevel0SingleRank(param, execMem, level0CommInfo));
        return HCCL_SUCCESS;
    }

    // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小)
    GroupSlicesInfo groupSlicesInfoLevel0;
    for (u32 groupId = 0; groupId < topoAttr_.moduleNum; groupId++) {
        MemBlockInfo memInfo;
        for (u32 dataId = 0; dataId < level0CommInfo.localRankSize; dataId ++) {
            u64 globalDataId = groupId * level0CommInfo.localRankSize + dataId;
            u64 size = groupSize_[globalDataId];
            u64 offset = globalDataId * sizePerBlock_;
            memInfo.size.push_back(size);
            memInfo.userInputOffsets.push_back(offset);
            memInfo.inputOffsets.push_back(offset);
            memInfo.outputOffsets.push_back(offset);
        }
        groupSlicesInfoLevel0.push_back(memInfo);
    }

    CHK_RET(ActiveSlaveStreams(param.stream));
    all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;  // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数） 
    
    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    
    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    CHK_RET(level0TempAlg->Prepare(execMem.inputPtr, execMem.inputMem, outputMem, param.stream,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfoLevel0,
        param.reduceType, all2allOffset_, param.DataDes.dataType, true));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel1

[coll_all_reduce_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L205)

```cpp
HcclResult CollAllReduceOrderPreservedExecutor::RunReduceScatterLevel1(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    u32 level0Ranksize = level0CommInfo.localRankSize;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    MemBlockInfo memInfo;
    u32 inputBaseIndex = (all2allOffset_ + commIndex) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数） 
    for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId ++) {
        u64 inputIndex = inputBaseIndex + dataId * level0Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * sizePerBlock_);
        memInfo.size.push_back(groupSize_[commIndex + dataId * level0Ranksize]);
        u64 outputIndex = commIndex + dataId * level0Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * sizePerBlock_);
        memInfo.userInputOffsets.push_back(outputIndex * sizePerBlock_);
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    bool isUseCclIn = (level0Ranksize == 1) || (commIndex == level0LastRank - 1);
    CHK_RET(level1TempAlg->Prepare(execMem.inputMem, outputMem, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, memInfo, param.reduceType,
        param.DataDes.dataType, isUseCclIn, commIndex == level0LastRank, true));
    
    CHK_RET(level1TempAlg->RegisterProfiler((level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        level0CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceOrderPreservedExecutor::RunAllGatherLevel0

[coll_all_reduce_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L242)

```cpp
HcclResult CollAllReduceOrderPreservedExecutor::RunAllGatherLevel0(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    u64 count = execMem.count / topoAttr_.userRankSize;
    u64 serverOffsetConut = topoAttr_.userRank / level0RankSize * level0RankSize;

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level0RankSize; rank++) {
        Slice userslice;
        userslice.size = groupSize_[rank + serverOffsetConut];
        userslice.offset = userslice.size == 0 ? 0 : (rank + serverOffsetConut) * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;
    
    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, nullptr, commIndex, level0RankSize));
    CHK_RET(level0TempAlg->Prepare(outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(level0TempAlg->RegisterProfiler((level0RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceOrderPreservedExecutor::RunAllGatherLevel1

[coll_all_reduce_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L276)

```cpp
HcclResult CollAllReduceOrderPreservedExecutor::RunAllGatherLevel1(const OpParam &param, ExecMem &execMem,
    const SubCommInfo &level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    std::unique_ptr<AlgTemplateBase> level1TempAlg;  // Level1Allgather(根据算法选择)
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
    }

    DeviceMem outputMem = scratchMemFlag_ ? execMem.scratchMem : execMem.outputMem;

    u32 level0RankSize = level0CommInfo.localRankSize;
    u64 count = execMem.count / level1CommInfo.localRankSize;

    std::vector<u64> level1GroupSize;
    for (u32 rank = 0; rank < level1CommInfo.localRankSize; rank++) {
        u64 size = 0;
        for (u32 level0RankId = 0; level0RankId < level0RankSize; level0RankId++) {
            size += groupSize_[rank * level0RankSize + level0RankId];
        }
        level1GroupSize.push_back(size);
    }

    // allgather 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> dataSegsSlice;
    for (u32 rank = 0; rank < level1CommInfo.localRankSize; rank++) {
        Slice userslice;
        userslice.size = level1GroupSize[rank];
        userslice.offset = rank * level0RankSize * sizePerBlock_;
        dataSegsSlice.emplace_back(std::move(userslice));
    }

    CHK_SMART_PTR_NULL(level1TempAlg);
    CHK_RET(level1TempAlg->Prepare(outputMem, outputMem, outputMem, count, param.DataDes.dataType, param.stream, 
        HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, dataSegsSlice));
    
    CHK_RET(level1TempAlg->RegisterProfiler((level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) 
        + level1CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceOrderPreservedExecutor::KernelRun

[coll_all_reduce_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L331)

```cpp
HcclResult CollAllReduceOrderPreservedExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[%s]The CollAllReduceOrderPreservedExecutor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    CalcSizePerBlock(param, execMem);
    CalGroupSlices(param, execMem);

    u64 inputSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    scratchMemFlag_ = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && 
        (inputSize < (topoAttr_.userRankSize - 1) * sizePerBlock_);

    // L0 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel0(param, execMem, level0CommInfo));
    // L1 节点间 reduce scatter
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScatterLevel1(param, execMem, level0CommInfo));
    }

    // Level0 节点内 AllGatherMeshAtomic
    CHK_RET(RunAllGatherLevel0(param, execMem, level0CommInfo));
    if (topoAttr_.moduleNum > 1) {
        // L1 节点间 allgather
        CHK_RET(RunAllGatherLevel1(param, execMem, level0CommInfo));
    }

    // 单算子需要 execMem.outputMem最后拷贝至UserOut
    if (scratchMemFlag_ || workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        void *srcPtr = scratchMemFlag_ ? execMem.scratchMem.ptr() : execMem.outputMem.ptr();
        DeviceMem srcMem = DeviceMem::create(srcPtr, dataSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    
    HCCL_INFO("[%s]order preserved AllReduce run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;
}
```
