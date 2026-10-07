# coll_all_reduce_mesh_opbase_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshOpbaseExecutor::CalcLoopMaxCount

[coll_all_reduce_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_executor.cc#L71)

```cpp
u64 CollAllReduceMeshOpbaseExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / unitSize * topoAttr_.userRankSize;
    return maxCountPerLoop;
}
```

### CollAllReduceMeshOpbaseExecutor::KernelRun

[coll_all_reduce_mesh_opbase_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_executor.cc#L91)

```cpp
HcclResult CollAllReduceMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllReduceMeshOpbaseExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::unique_ptr<AlgTemplateBase> level0TempAlg;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    CHK_RET(ActiveSlaveStreams(param.stream));
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_MESH_DIRECT, 
        dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, 
        algResResp_->notifiesAux, level0CommInfo.localRank, level0CommInfo.localRankSize, 
        topoAttr_.userRank, &opInfo));
    CHK_RET(level0TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));
    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("AllReduce mesh opbase run success.");
    return HCCL_SUCCESS;
}
```
