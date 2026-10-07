# coll_reduce_scatter_mesh_dma_elimination.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterMeshDmaEliminationExecutor::CalcLoopMaxCount

[coll_reduce_scatter_mesh_dma_elimination.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_dma_elimination.cc#L74)

```cpp
u64 CollReduceScatterMeshDmaEliminationExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / unitSize;
    return maxCountPerLoop;
}
```

### CollReduceScatterMeshDmaEliminationExecutor::KernelRun

[coll_reduce_scatter_mesh_dma_elimination.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_dma_elimination.cc#L94)

```cpp
HcclResult CollReduceScatterMeshDmaEliminationExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollReduceScatterMeshDmaEliminationExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面

    /* *******************节点内reducescatter ******************************************/
    CHK_RET(ActiveSlaveStreams(param.stream));

    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;

    HCCL_DEBUG("inputMem.size()=%llu, level0CommInfo.localRankSize=%u, commIndex=%u",
        execMem.inputMem.size(), level0CommInfo.localRankSize, commIndex);

    HcomCollOpInfo *opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64) &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(param.tag, execMem.inputMem, execMem.scratchMem,
            execMem.count, param.DataDes.dataType, param.reduceType, dataSegsSlice, const_cast<Stream&>(param.stream),
            COMM_LEVEL0, 0, opInfoPtr));
    } else {
        std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
        // mesh算法stream数量为rank数减1
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(param.tag, execMem.inputMem, execMem.scratchMem,
            execMem.count, param.DataDes.dataType, param.reduceType, multiStreamSlice, param.stream, COMM_LEVEL0, 0));
    }

    return HCCL_SUCCESS;
}
```
