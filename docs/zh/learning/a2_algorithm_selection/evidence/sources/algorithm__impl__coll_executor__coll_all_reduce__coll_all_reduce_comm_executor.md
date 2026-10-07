# coll_all_reduce_comm_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceCommExecutor::CalcCombinedCommInfo

[coll_all_reduce_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_comm_executor.cc#L46)

```cpp
HcclResult CollAllReduceCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));

    return HCCL_SUCCESS;
}
```

### CollAllReduceCommExecutor::KernelRun

[coll_all_reduce_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_comm_executor.cc#L83)

```cpp
HcclResult CollAllReduceCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] userRank[%u] starts.", __func__, topoAttr_.userRank);
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, 0);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> tempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
        if (curSize <= NHR_ALLREDUCE_SMALL_SIZE) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        }
        HCCL_INFO("AllReduce comm: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
        if (topoAttr_.deviceType != DevType::DEV_TYPE_910_93 || algoAttr_.isSupportAtomicWrite) {
            tempAlg->CloseBarrier();
        }
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_V1 in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NB in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
    } else {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_COMBINE/COMM_COMBINE_ORDER", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));
    }
    CHK_SMART_PTR_NULL(tempAlg);

    u32 rankSize = combinedCommInfo.localRankSize;
    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        combinedCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, combinedCommInfo));
    return HCCL_SUCCESS;
}
```
