# coll_all_to_all_v_fullmesh_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollRunAlltoAllVFullMesh::KernelRun

[coll_all_to_all_v_fullmesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_fullmesh_executor.cc#L90)

```cpp
HcclResult CollRunAlltoAllVFullMesh::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllVFullMesh][KernelRun] AllToAllV fullmesh start");
    bool opbaseCopyMode = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_;

    // 构造入参
    AlltoAllVBufferInfo sendInfo;
    sendInfo.mem = opbaseCopyMode ? execMem.inputMem : algResResp_->paramInputMem;
    sendInfo.counts = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].sendCounts[0];
    sendInfo.displs = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].sendDispls[0];
    sendInfo.dataType = param.All2AllDataDes.sendType;

    AlltoAllVBufferInfo recvInfo;
    recvInfo.mem = opbaseCopyMode ? execMem.outputMem : algResResp_->paramOutputMem;
    recvInfo.counts = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].recvCounts[0];
    recvInfo.displs = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].recvDispls[0];
    recvInfo.dataType = param.All2AllDataDes.recvType;

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    // 执行算法
    std::map<u32, std::vector<u64>> rankSendDisplsMap;
    std::map<u32, std::vector<u64>> rankRecvDisplsMap;
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || isAlltoAllZCopyMode_) {
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            rankSendDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo_[i].sendOffset));
            rankRecvDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo_[i].recvOffset));
        }
    }

    std::unique_ptr<AlgTemplateBase> pairWisePtr = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_PAIRWISE, dispatcher_);
    CHK_SMART_PTR_NULL(pairWisePtr);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer Copy模式
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, execMem.inputMem, execMem.outputMem, isAlltoAllZCopyMode_,
            const_cast<Stream&>(param.stream), workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) {
        DeviceMem dstMem = execMem.inputMem.range(0, algResResp_->paramInputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, algResResp_->paramInputMem, const_cast<Stream&>(param.stream)));

        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, execMem.inputMem, execMem.outputMem,
            isAlltoAllZCopyMode_, const_cast<Stream&>(param.stream),
            workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo)); // inputMem_ -> outputMem_

        DeviceMem srcMem = execMem.outputMem.range(0, algResResp_->paramOutputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, algResResp_->paramOutputMem, srcMem, const_cast<Stream&>(param.stream)));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, isAlltoAllZCopyMode_, const_cast<Stream&>(param.stream),
        workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        // 保证最新的commMesh是为该次alltoallv创建（不支持多线程）
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo));
    } else {
        HCCL_ERROR("[hcclImpl][RunAlltoAllVFullMesh]work flow mode is invalid");
        return HCCL_E_PARA;
    }

    HCCL_INFO("[CollRunAlltoAllVFullMesh] executor run success");

    return HCCL_SUCCESS;
}
```
