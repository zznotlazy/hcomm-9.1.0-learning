# coll_all_gather_v_mesh_graph_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherVMeshGraphExecutor::KernelRun

[coll_all_gather_v_mesh_graph_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_graph_executor.cc#L64)

```cpp
HcclResult CollAllGatherVMeshGraphExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherVMeshGraphExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.VDataDes.dataType];

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < level0RankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * perDataSize;
        userslice.size = counts[rank] * perDataSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    u64 inputMemSize = execMem.inputMem.size();
    u64 level0Offset = outputSlices[commIndex].offset;
    DeviceMem dstMem = execMem.outputMem.range(level0Offset, inputMemSize);// 根据卡序排
    CHK_SMART_PTR_NULL(dstMem);
    // 将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherVMeshGraphExecutor][KernelRun]AllGatherV mesh memcpy Failed, Offset[%llu], Size[%llu].",
        level0Offset, inputMemSize), ret);

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环AllGather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem;
    CHK_SMART_PTR_NULL(currentOutputMem);

    std::unique_ptr<AlgTemplateBase> level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, nullptr, commIndex, level0RankSize));
    CHK_RET(level0TempAlg->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem,
        execMem.count , param.VDataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, outputSlices, 0));
    u32 rankSize = level0RankSize;
    CHK_RET(level0TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("AllGatherV mesh run success");
    return HCCL_SUCCESS;
}
```
