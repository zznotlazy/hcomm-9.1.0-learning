# coll_reduce_scatter_order_preserved_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0HD

[coll_reduce_scatter_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc#L139)

```cpp
HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0HD(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType, 0};

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(execMem.inputMem, execMem.scratchMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0,
        reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, &opInfo));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0

[coll_reduce_scatter_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc#L163)

```cpp
HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel0(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    if (level0CommInfo.localRankSize == 1) {
        all2allOffset_ = topoAttr_.moduleNum > 1 ? 1 : 0;
        HCCL_INFO("[%s] single rank per module, skip L0 AllToAll and LocalReduce, tag[%s]",
            __func__, tag_.c_str());
        CHK_RET(RunReduceScatterLevel0SingleRank(param, execMem, level0CommInfo));
	return HCCL_SUCCESS;
    }

    CHK_RET(ActiveSlaveStreams(param.stream));
    if (isUseHDAlg_) {
        CHK_RET(RunReduceScatterLevel0HD(param, execMem, level0CommInfo));
    } else {
        // 切分数据(ReduceScatter分组，记录每组的起始偏移和大小) 
        GroupSlicesInfo groupSlicesInfoLevel0;
        u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        for (u32 groupId = 0; groupId < topoAttr_.moduleNum; groupId++) {
            MemBlockInfo memInfo;
            for (u32 dataId = 0; dataId < level0CommInfo.localRankSize; dataId ++) {
                u64 offset = (dataId + groupId * level0CommInfo.localRankSize) * size;
                u64 userMemInOffset = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] *
                    (dataId + groupId * level0CommInfo.localRankSize);
                memInfo.size.push_back(size);
                memInfo.userInputOffsets.push_back(userMemInOffset);
                memInfo.inputOffsets.push_back(offset);
                memInfo.outputOffsets.push_back(offset);
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
            groupSlicesInfoLevel0, param.reduceType, all2allOffset_, param.DataDes.dataType, false));
        CHK_RET(level0TempAlg->RegisterProfiler(
            (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    }
    return HCCL_SUCCESS;
}
```

### CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel1

[coll_reduce_scatter_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc#L212)

```cpp
HcclResult CollReduceScatterOrderPreservedExecutor::RunReduceScatterLevel1(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // 切分数据，记录每组的起始偏移和大小（仅1组）
    u64 size = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    MemBlockInfo memInfo;
    u32 level0Ranksize = level0CommInfo.localRankSize;
    u32 inputBaseIndex = (all2allOffset_ + commIndex) % level0Ranksize; // 多机场景需要偏移1（给L1预留计算位，减少拷贝次数） 
    for (u32 dataId = 0; dataId < level1CommInfo.localRankSize; dataId ++) {
        u64 inputIndex = inputBaseIndex + dataId * level0Ranksize;
        memInfo.inputOffsets.push_back(inputIndex * size);
        u64 outputIndex = commIndex + dataId * level0Ranksize;
        memInfo.outputOffsets.push_back(outputIndex * size);
        memInfo.userInputOffsets.push_back(outputIndex * size);
        memInfo.size.push_back(size);  
    }

    std::unique_ptr<AlgTemplateBase> level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 level0LastRank = level0Ranksize - 1;
    bool isUseCclIn = (level0Ranksize == 1) || (commIndex == level0LastRank - 1);
    bool borrowSpace = level0Ranksize == 1;
    CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.scratchMem,
        param.stream, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        memInfo, param.reduceType, param.DataDes.dataType, isUseCclIn,
        commIndex == level0LastRank, borrowSpace));
    CHK_RET(level1TempAlg->RegisterProfiler((level0Ranksize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        level0CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterOrderPreservedExecutor::KernelRun

[coll_reduce_scatter_order_preserved_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc#L250)

```cpp
HcclResult CollReduceScatterOrderPreservedExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s]CollReduceScatterOrderPreservedExecutor starts, tag[%s]", __func__, tag_.c_str());
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // L0 节点内 reduce scatter
    CHK_RET(RunReduceScatterLevel0(param, execMem, level0CommInfo));
    // L1 节点间 reduce scatter
    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScatterLevel1(param, execMem, level0CommInfo));
    }

    if (!isUseHDAlg_) {
        // 非HD算法 execMem.scratchMem最后拷贝至UserOut
        u64 dataSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        DeviceMem srcMem = execMem.scratchMem.range(dataSize * topoAttr_.userRank, dataSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, dataSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[%s]order preserved ReduceScatter run success, tag[%s]", __func__, tag_.c_str());
    return HCCL_SUCCESS;
}
```
