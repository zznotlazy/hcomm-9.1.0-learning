# coll_reduce_comm_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceCommExecutor::KernelRun

[coll_reduce_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_comm_executor.cc#L61)

```cpp
HcclResult CollReduceCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] ReduceCommExecutor starts.", __func__);
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCE_RING, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCE_RING in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));

    // 获取root
    u32 root = 0;
    CHK_RET(GetRankByUserRank(commPlane, COMM_INDEX_0, param.root, root));

    u32 rankSize = combinedCommInfo.localRankSize;
    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        root, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        combinedCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    HCCL_INFO("[CollReduceCommExecutor] ReduceCommExecutor run success");
    return HCCL_SUCCESS;
}
```
