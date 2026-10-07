# coll_all_gather_v_mesh_graph_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherVMeshGraphPipelineExecutor::KernelRun

[coll_all_gather_v_mesh_graph_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_graph_pipeline_executor.cc#L65)

```cpp
HcclResult CollAllGatherVMeshGraphPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllGatherVMeshGraphPipelineExecutor][KernelRun]AllGatherMeshGraphPipelineExecutor begins.");

    // 先获取 comm level0 和 comm level1 的value
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    u32 perDataSize = SIZE_TABLE[param.VDataDes.dataType];
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < topoAttr_.userRankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * perDataSize;
        userslice.size = counts[rank] * perDataSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    // 激活从流
    CHK_RET(ActiveSlaveStreams(param.stream));

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, 0, param.VDataDes.dataType, 0, HCCL_REDUCE_RESERVED};

    std::unique_ptr<AlgTemplateBase> tempAlg =
        AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_V_GRAPH_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(&opInfo, topoAttr_.userRank, execMem.count, execMem.inputMem, execMem.outputMem,
                             level0CommInfo, level1CommInfo,  const_cast<Stream &>(param.stream),
                             algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, outputSlices));
    CHK_RET(tempAlg->RunAsync());
    return HCCL_SUCCESS;
}
```
