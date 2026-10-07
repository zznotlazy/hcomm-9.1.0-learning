# coll_all_to_all_v_staged_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllVStaged::PrepareAlltoAllVStaged1

[coll_all_to_all_v_staged_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_staged_executor.cc#L229)

```cpp
HcclResult CollRunAlltoAllVStaged::PrepareAlltoAllVStaged1(DeviceMem &sendBuf, DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosIntra,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosIntra,
    Stream &stream, const std::string &tag, std::unique_ptr<AlgTemplateBase> &alltoallLevel0,
    ExecMem &execMem)
{
    (void) tag;
    // opbase BCopy 不支持fullmesh算法，因此不必做算法选择
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer拷贝模式
        HCCL_INFO("Running AllToAllV Staged Pairwise intra Server");
        alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
        CHK_SMART_PTR_NULL(alltoallLevel0);
        CHK_RET(alltoallLevel0->Prepare(sendBuf, scratchMem, execMem.inputMem, execMem.outputMem, sendAddrInfosIntra,
            recvAddrInfosIntra, isAlltoAllZCopyMode_, stream));
    } else {
        bool isOpBaseZCopy = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAlltoAllZCopyMode_;
        DeviceMem inBuf = (isOpBaseZCopy) ? execMem.inputMem : sendBuf;
        // 单MeshAggregation下, 分级算法不做第二级, 结果输出到outCCLbuffer_
        DeviceMem outBuf = (isOpBaseZCopy && topoAttr_.isSingleMeshAggregation) ? recvBuf : scratchMem;
        // opbase ZCopy 与 graph，除input buffer差异外，其余行为应保持一致
        if (isOpBaseZCopy) { // 单算子 && ZCopy模式
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.inputMem, sendBuf, stream));
        }
        std::vector<HcclAlgoType> algoTypeArr = topoMatcher_->GetAlgoConfig();
        // 互联场景, alltoall暂不支持走fullmesh+pairwise
        if ((algoTypeArr[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE &&
            algoTypeArr[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE) ||
            const_cast<HcclTopoInfo &>(topoAttr_).pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] != 0 ||
            topoAttr_.meshAggregationRankSize == 1) {
            HCCL_INFO("Running AllToAllV Staged Pairwise intra Server");
            alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
            CHK_SMART_PTR_NULL(alltoallLevel0);
            CHK_RET(alltoallLevel0->Prepare(inBuf, outBuf, sendAddrInfosIntra, recvAddrInfosIntra,
                isAlltoAllZCopyMode_, stream));
        } else {
            HCCL_INFO("Running AllToAllV Staged Mesh intra Server");
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                CHK_RET(ActiveSlaveStreams(AlltoAllVParam_.stream));
            }
            if (GetExternalInputHcclEnableFfts() || algResResp_->slaveStreams.size() == 0) {
                alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_MESH, dispatcher_);
            } else {
                alltoallLevel0 = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_MESH, vDispatcher_);
            }
            CHK_SMART_PTR_NULL(alltoallLevel0);
            CHK_RET(alltoallLevel0->Prepare(inBuf, outBuf, sendAddrInfosIntra, recvAddrInfosIntra,
                isAlltoAllZCopyMode_, topoAttr_.userRank, stream, algResResp_->slaveStreams,
                algResResp_->notifiesMain, algResResp_->notifiesAux));
        }
    }
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllVStaged::PrepareAlltoAllVStaged2

[coll_all_to_all_v_staged_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_staged_executor.cc#L365)

```cpp
HcclResult CollRunAlltoAllVStaged::PrepareAlltoAllVStaged2(DeviceMem &recvBuf, DeviceMem &scratchMem,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter,
    Stream &stream, const std::string &tag, std::unique_ptr<AlgTemplateBase> &alltoallLevel1,
    ExecMem &execMem)
{
    (void) tag;
    alltoallLevel1 = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
    CHK_SMART_PTR_NULL(alltoallLevel1);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && BCopy模式
        CHK_RET(alltoallLevel1->Prepare(scratchMem, recvBuf, execMem.inputMem, execMem.outputMem, sendAddrInfosInter,
            recvAddrInfosInter, isAlltoAllZCopyMode_, stream));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) { // 单算子 && ZCopy模式
        CHK_RET(alltoallLevel1->Prepare(scratchMem, execMem.outputMem, execMem.inputMem, execMem.outputMem,
            sendAddrInfosInter, recvAddrInfosInter, isAlltoAllZCopyMode_, stream));
    } else {
        CHK_RET(alltoallLevel1->Prepare(scratchMem, recvBuf, sendAddrInfosInter, recvAddrInfosInter,
            isAlltoAllZCopyMode_, stream));
    }
    return HCCL_SUCCESS;
}
```

### CollRunAlltoAllVStaged::KernelRun

[coll_all_to_all_v_staged_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_staged_executor.cc#L390)

```cpp
HcclResult CollRunAlltoAllVStaged::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVStaged][KernelRun] AllToAllV staged starts");
    CHK_PRT_RET(topoAttr_.userRankSize % topoAttr_.meshAggregationRankSize != 0,
        HCCL_ERROR("userRankSize[%u] is not an Integer multiple of MeshAggregation Dev Num[%u]",
        topoAttr_.userRankSize, topoAttr_.meshAggregationRankSize), HCCL_E_PARA);

    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = topoAttr_.userRank;
    userRankInfo.userRankSize = topoAttr_.userRankSize;
    bool alltoallMeshReadOnly = FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(topoAttr_.deviceType,
        topoAttr_.userRankSize, topoAttr_.useSuperPodMode, topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL));

    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosIntra;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosIntra;
    bool isSingleMesh = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_ && topoAttr_.isSingleMeshAggregation;
    CalcIntraMeshAggregationAlltoAllMemInfo(userRankInfo, allMeshAggregationSendRecvInfo_, sendAddrInfosIntra,
        recvAddrInfosIntra, topoAttr_.meshAggregationRankSize, isSingleMesh);

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    std::unique_ptr<AlgTemplateBase> alltoallLevel0 = nullptr;
    CHK_RET(PrepareAlltoAllVStaged1(algResResp_->paramInputMem, algResResp_->paramOutputMem, execMem.scratchMem,
        sendAddrInfosIntra, recvAddrInfosIntra, const_cast<Stream&>(param.stream), tag_, alltoallLevel0, execMem));
    if ((algResResp_->slaveStreams.size() != 0) &&
        (!GetExternalInputHcclEnableFfts()) && isAlltoAllZCopyMode_) {
        HCCL_INFO("[AlltoAllOperator][RunAlltoAllVStaged] staged 0 use parallel multi-thread delivery of tasks");
        CHK_RET(RunTemplateWithVirtualLink(alltoallLevel0, level0CommInfo));
        // 多流场景下，并行多线程下发task处理
        CHK_RET(ParallelTaskLoaderProcess(tag_, const_cast<Stream&>(param.stream), level0CommInfo,
            algResResp_->slaveStreams));
    } else {
        CHK_RET(RunAlltoAllVTemplateStaged(alltoallLevel0, level0CommInfo));
    }

    HCCL_INFO("[hcclImpl][RunAlltoAllVStaged] stage0 run success!");
        
    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosInter;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosInter;
    CalcInterMeshAggregationAlltoAllMemInfo(userRankInfo, allMeshAggregationSendRecvInfo_,sendAddrInfosInter,
        recvAddrInfosInter, topoAttr_.meshAggregationRankSize);

    if (((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
            isAlltoAllZCopyMode_) || alltoallMeshReadOnly)  && topoAttr_.isSingleMeshAggregation) {
        HCCL_DEBUG("we don't need to do stage 2 when there is only one mesh aggregation");
        // we don't need to do stage 2 when there is only one mesh aggregation
    } else {
        HCCL_INFO("[hcclImpl][RunAlltoAllVStaged] stage1 run starts!");
        CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0);
        std::unique_ptr<AlgTemplateBase> alltoallLevel1 = nullptr;
        CHK_RET(PrepareAlltoAllVStaged2(algResResp_->paramOutputMem, execMem.scratchMem, sendAddrInfosInter,
            recvAddrInfosInter, const_cast<Stream&>(param.stream), tag_, alltoallLevel1, execMem));
        CHK_RET(RunAlltoAllVTemplateStaged(alltoallLevel1, level1CommInfo));
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_ && !topoAttr_.isSingleMeshAggregation) {
        DeviceMem srcMem = (execMem.outputMem).range(0, algResResp_->paramOutputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, algResResp_->paramOutputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[CollRunAlltoAllVStaged][kernelRun] AllToAllV staged ends");
    return HCCL_SUCCESS;
}
```
