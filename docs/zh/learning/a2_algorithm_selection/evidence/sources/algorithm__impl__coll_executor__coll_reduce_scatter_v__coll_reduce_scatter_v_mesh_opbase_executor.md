# coll_reduce_scatter_v_mesh_opbase_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDisplsMultiModule

[coll_reduce_scatter_v_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L107)

```cpp
HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDisplsMultiModule(const u64 maxTotalCount,
    std::vector<u64> &countsLeft, std::vector<u64> &displs, std::vector<u64> &curCounts, std::vector<u64> &curDispls,
    bool &finished)
{
    curCounts = std::vector<u64>(countsLeft.size(), 0);
    curDispls = std::vector<u64>(displs.size(), 0);
    auto allocatableCount = maxTotalCount;

    // 先设置本轮的displacements，等于入参displs
    std::copy(displs.begin(), displs.end(), curDispls.begin());

    // 分配本轮的counts，如果CCLbuffer空间还没完全利用，则再进行分配
    while (allocatableCount > 0) {
        // 计算现在还有几个rank还有数据需要去通信(countsLeft不为0)
        const auto nonZeroCount =
            std::count_if(countsLeft.begin(), countsLeft.end(), [](const u64 count) { return count != 0; });
        if (nonZeroCount == 0) {
            finished = true;
            HCCL_INFO("[%s] Calc CurCountsAndCurDispls for multiModule finish.", __func__);
            return HCCL_SUCCESS;
        }
        // 计算每个rank可以分到多少count
        const auto perRankCount = allocatableCount / nonZeroCount;
        if (perRankCount == 0) {
            break;
        }
        HCCL_DEBUG("[CollReduceScatterVMeshOpbaseExecutor]Calc for perRankCount start");
        for (auto i = 0U; i < countsLeft.size(); ++i) {
            const auto curCount = countsLeft[i] < perRankCount ? countsLeft[i] : perRankCount;
            allocatableCount -= curCount;
            curCounts[i] += curCount;
            countsLeft[i] -= curCount;
            displs[i] += curCount;
        } 
    }
    //特殊情况下，allocatableCount 刚好使用完毕时，不仅如此while循环，导致RunLoop额外循环一次
    const auto nonZeroCount =
        std::count_if(countsLeft.begin(), countsLeft.end(), [](const u64 count) { return count != 0; });
    if (nonZeroCount == 0) {
        finished = true;
    }
    HCCL_INFO("[%s] Calc CurCountsAndCurDispls for multiModule finish.", __func__);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0SingleModule

[coll_reduce_scatter_v_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L164)

```cpp
HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0SingleModule(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbaseExecutor] Run ReduceScatterV Level0 SingleModule ");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    u32 level0RankSize = level0CommInfo.localRankSize;

    /* *******************节点内reducescatter ******************************************/
    // reduce_scatter_v 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rankId = 0; rankId < level0RankSize; ++rankId) {
        Slice userslice;
        userslice.offset = displs[rankId] * unitSize;
        userslice.size = counts[rankId] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    HcomCollOpInfo *opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType,
        param.root, param.reduceType};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType,
        param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, opInfoPtr));

    CHK_RET(TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, level0CommInfo));

    return HCCL_SUCCESS;
}
```

### CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0

[coll_reduce_scatter_v_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L210)

```cpp
HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel0(const OpParam &param, ExecMem &execMem,
    SubCommInfo &level0CommInfo)
{
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];
    u32 level0rankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    /* *******************节点内reducescatter ******************************************/
   
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    u64 offset = 0;
    
    for (u32 moduleId = 0; moduleId < topoAttr_.moduleNum; moduleId++) {
        for (u32 rankId = 0; rankId < level0rankSize; ++rankId) {
            if (topoAttr_.userRank / level0rankSize == moduleId) {
                Slice userslice;
                userslice.size = counts[rankId + moduleId * level0rankSize] * unitSize;
                userslice.offset = offset * unitSize;
                inputSlices.emplace_back(std::move(userslice));
            }
            offset += counts[rankId + moduleId * level0rankSize];
        }
    }
    
    HcomCollOpInfo *opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType,
        param.root, param.reduceType};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType,
        param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr,    
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, opInfoPtr));

    CHK_RET(TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, level0CommInfo));

    // 机间reduceScatter 结果 搬运到 cclout
    DeviceMem srcMem = execMem.inputMem.range(inputSlices[commIndex].offset,
       inputSlices[commIndex].size);
    CHK_SMART_PTR_NULL(srcMem);
    Stream stream = param.stream;
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, stream));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel1

[coll_reduce_scatter_v_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L289)

```cpp
HcclResult CollReduceScatterVMeshOpbaseExecutor::RunReduceScattervLevel1(const OpParam &param, ExecMem &execMem,
    const SubCommInfo &level0CommInfo)
{
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    HcclDataType dataType = param.VDataDes.dataType;

    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    /* ******************第一步: 机间reducescatter *******************************/
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HCCL_INFO("reducescatterv mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nonuniform-bruck algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr)); 
    } else { 
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_INFO("reducescatterv mesh: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
        level1TempAlg->CloseBarrier();
    }

    std::vector<Slice> slices;
    CalReduceScatterVSliceData(param, level0RankSize, level1RankSize, slices);
  
    CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, 0,
        dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, slices));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterVMeshOpbaseExecutor::KernelRun

[coll_reduce_scatter_v_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L338)

```cpp
HcclResult CollReduceScatterVMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshOpbaseExecutor] reducescatterv mesh run");
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo  = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    if (topoAttr_.moduleNum > 1) {
        CHK_RET(RunReduceScattervLevel1(param, execMem, level0CommInfo));
        CHK_RET(RunReduceScattervLevel0(param, execMem, level0CommInfo));    
    } else {
        CHK_RET(RunReduceScattervLevel0SingleModule(param, execMem, level0CommInfo));    
    }
    return HCCL_SUCCESS;
}
```
