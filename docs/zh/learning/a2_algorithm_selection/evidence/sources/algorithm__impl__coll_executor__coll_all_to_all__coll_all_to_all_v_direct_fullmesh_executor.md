# coll_all_to_all_v_direct_fullmesh_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllDirectFullmesh::Orchestrate

[coll_all_to_all_v_direct_fullmesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc#L22)

```cpp
HcclResult CollRunAlltoAllDirectFullmesh::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);
    
    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

    HCCL_INFO("tag[%s], AlltoAllDirectFullmesh tempAlg orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllDirectFullmesh::GetAdjInfo

[coll_all_to_all_v_direct_fullmesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc#L52)

```cpp
HcclResult CollRunAlltoAllDirectFullmesh::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    HCCL_INFO("[GetAdjInfo-nslbdp] GetAdjInfo.");
    algResResp_ = &algRes;
    SubCommInfo levelCommInfo = {0};
    AdjInfo nslbAdjInfo = {0};
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;

    u32 localRank= topoAttr_.userRank;
    u32 localRankSize = topoAttr_.userRankSize;

    std::unique_ptr<AlgTemplateBase> levelTempAlg;

    HCCL_INFO("[GetAdjInfo-nslbdp] SelectTempAlg.");
    levelTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(levelTempAlg);
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    if (devNumInlocalPod == INVALID_VALUE_RANKSIZE) {
        HCCL_INFO("[GetAdjInfo-nslbdp] devNumInlocalPod == INVALID_VALUE_RANKSIZE.");
        return HCCL_SUCCESS;
    }

    nslbAdjInfo.dstRankNum = devNumInlocalPod;
    CHK_RET(levelTempAlg->GetNslbAdjInfo(localRank, localRankSize, levelCommInfo.links, nslbAdjInfo));

    adjInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[GetAdjInfo-nslbdp] adjInfo.dstRankNum[%u].", adjInfo.dstRankNum);
    
    for (size_t i = 0; i < nslbAdjInfo.nsAdjInfo.size(); i++) {
        NslbDpAdjInfo dpAdjInfo = {0};
        dpAdjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[i].dstLocalRankId;
        dpAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[i].phaseId;
        dpAdjInfo.rev = 0;
        adjInfo.nsAdjInfo.push_back(dpAdjInfo); 
        HCCL_INFO("[nslbdp]GetAdjInfo dstLocalRankId[%u], phaseId[%u].",
                   nslbAdjInfo.nsAdjInfo[i].dstLocalRankId, nslbAdjInfo.nsAdjInfo[i].phaseId);
    }
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllDirectFullmesh::KernelRun

[coll_all_to_all_v_direct_fullmesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc#L311)

```cpp
HcclResult CollRunAlltoAllDirectFullmesh::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] AllToAll fullmesh start.", __func__);

    // 准备数据
    CHK_RET(ActiveSlaveStreams(param.stream));
    CHK_RET(GetAlltoAllvTmpRankSendRecvInfo(param));

    // 获取当前超节点内总卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    bool isA2MultiModule = topoAttr_.deviceType == DevType::DEV_TYPE_910B &&
                            !topoAttr_.isSingleMeshAggregation;
    // isSuPodAsym 表示A2A3卡数不一致场景或者A3多超节点server数不同场景
    bool isSuPodAsym = false;
    if (topoAttr_.superPodNum > 1) {
        isSuPodAsym = (static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode) || static_cast<bool>(topoAttr_.multiSuperPodDiffServerNumMode));
    } else {
        isSuPodAsym = (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) || isA2MultiModule) && static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode);
    }

    // 执行
    // 注意: 如果使用了非AlltoAllVDirectFullMesh的算法模板, 需要同步修改framework中的NeedOpUnfoldCache()函数
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH in COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    PrepareData prepareData;
    prepareData.stream = param.stream;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.userRankSize = topoAttr_.userRankSize;
    prepareData.linksPtr = &level0CommInfo.links;
    prepareData.localSendRecvInfoPtr = &localSendRecvInfo_;
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

    // 如果使能alltoallv aicpu cache
    if (needAlltoallvCache_) {
        // 注意: 一定是alltoallv类算子才有可能设置needAlltoallvCache_, 让alltoallv temp alg感知cache并保存算法中间结果
        CHK_PRT_RET(!(param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC),
            HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][KernelRun] needAlltoallvCache_[%u] opType[%u]",
                needAlltoallvCache_, param.opType),
            HCCL_E_INTERNAL);

        // 使能alltoallv temp alg感知alltoallv aicpu cache
        prepareData.needAlltoallvCache = true;
    } else {
        prepareData.needAlltoallvCache = false;
    }

    CHK_RET(tempAlg->Prepare(prepareData));

    CHK_RET(tempAlg->RunAsync());

    if (needAlltoallvCache_) {
        // 在tempAlg被销毁前保存hcclOffset-dstRank之间的mapping信息
        // 注意: CollRunAlltoAllDirectFullmesh executor使用的一定是AlltoAllVDirectFullMesh template
        hcclOffsetDstRanksMap_.clear();
        HCCL_INFO("[CollRunAlltoAllDirectFullmesh][KernelRun] get hcclOffset-dstRanks mapping for AlltoAllVDirectFullMesh");
        CHK_RET(tempAlg->GetHcclOffsetDstRanksMap(hcclOffsetDstRanksMap_));
    }

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] executor run success.");
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

### CollRunAlltoAllDirectFullmesh::SelectTempAlg

[coll_all_to_all_v_direct_fullmesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc#L411)

```cpp
HcclResult CollRunAlltoAllDirectFullmesh::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    (void) level1RankSize;
    HCCL_INFO("[GetAdjInfo-nslbdp] SelectTempAlg.");
    level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    return HCCL_SUCCESS;
}
```
