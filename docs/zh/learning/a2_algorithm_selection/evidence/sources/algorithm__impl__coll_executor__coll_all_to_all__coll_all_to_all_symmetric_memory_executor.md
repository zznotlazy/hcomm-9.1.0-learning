# coll_all_to_all_symmetric_memory_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllFullMeshSymmetricMemory::Orchestrate

[coll_all_to_all_symmetric_memory_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_symmetric_memory_executor.cc#L23)

```cpp
HcclResult CollRunAlltoAllFullMeshSymmetricMemory::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.paramOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllFullMeshSymmetricMemory][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], AlltoAllFullMeshSymmetricMemory tempAlg orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllFullMeshSymmetricMemory::KernelRun

[coll_all_to_all_symmetric_memory_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_symmetric_memory_executor.cc#L171)

```cpp
HcclResult CollRunAlltoAllFullMeshSymmetricMemory::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] AllToAll fullmesh start.", __func__);

    // 准备数据
    CHK_RET(ActiveSlaveStreams(param.stream));
    CHK_RET(GetAlltoAllTmpRankSendRecvInfo(param));

    // 获取当前超节点内总卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(devNumInlocalPod, rankIdxInPod));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    // isSuPodAsym 表示A2A3卡数不一致场景或者A3多超节点server数不同场景
    bool isSuPodAsym = false;

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY in COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    PrepareData prepareData;
    prepareData.stream = param.stream;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.userRankSize = topoAttr_.userRankSize;
    prepareData.linksPtr = &level0CommInfo.links;
    prepareData.sendRecvInfoPtr = &sendRecvInfo_;
    prepareData.devNumInlocalPod = devNumInlocalPod;
    prepareData.rankIdxInPod = rankIdxInPod;

    prepareData.inputMem = algResResp_->paramInputMem;
    prepareData.outputMem = algResResp_->paramOutputMem;
    prepareData.cclInMem = execMem.inputMem;
    prepareData.cclOutMem = execMem.outputMem;
    prepareData.workMode = workflowMode_;
    prepareData.subStreamsPtr = &algResResp_->slaveStreams;
    prepareData.signalPtr = &algResResp_->notifiesMain;
    prepareData.signalAuxPtr = &algResResp_->notifiesAux;
    prepareData.isSuPodAsym = isSuPodAsym;
    prepareData.opType = param.opType;
    prepareData.algOpContext = algOpContext_;

    CHK_RET(tempAlg->Prepare(prepareData));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollRunAlltoAllFullMeshSymmetricMemory] executor run success.");
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        OpParam postSyncParam = param;
        if ((*prepareData.subStreamsPtr).size() == 0) {
            CHK_RET(PostSyncWithoutSubstream(postSyncParam, execMem));
        } else {
            PrepareData postSyncPrepareData = prepareData;
            CHK_RET(PostSyncWithSubstream(postSyncParam, execMem, postSyncPrepareData));
        }
    }
    return HCCL_SUCCESS;
}
```
