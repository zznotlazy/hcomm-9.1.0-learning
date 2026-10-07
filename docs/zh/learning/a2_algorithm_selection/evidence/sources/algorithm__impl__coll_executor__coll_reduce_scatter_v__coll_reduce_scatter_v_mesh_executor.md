# coll_reduce_scatter_v_mesh_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterVMeshExecutor::KernelRun

[coll_reduce_scatter_v_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_executor.cc#L84)

```cpp
HcclResult CollReduceScatterVMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterVMeshExecutor] ReduceScatterV mesh run");
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo subCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 rankSize = subCommInfo.localRankSize;

    CHK_RET(ActiveSlaveStreams(param.stream));

    /* *******************节点内reducescatter ******************************************/
    // reduce_scatter_v 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, dataType,
        param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank));

    CHK_RET(tempAlg->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, subCommInfo));

    return HCCL_SUCCESS;
}
```
