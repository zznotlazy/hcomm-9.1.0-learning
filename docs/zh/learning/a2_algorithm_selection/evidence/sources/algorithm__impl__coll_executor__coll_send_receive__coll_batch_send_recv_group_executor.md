# coll_batch_send_recv_group_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollBatchSendRecvGroupExecutor::Orchestrate

[coll_batch_send_recv_group_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_group_executor.cc#L119)

```cpp
HcclResult CollBatchSendRecvGroupExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvGroupExecutor] groupsendrecv starts.");
    
    sendStreamNum_ = GROUP_MAX_CONCURRENT;
    recvStreamNum_ = GROUP_MAX_CONCURRENT;
    HCCL_INFO("[Orchestrate] sendStreamNum_[%u], recvStreamNum_[%u]", sendStreamNum_, recvStreamNum_);

    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(param.stream));
    if (topoAttr_.userRankSize == 1) {
        HCCL_INFO("tag[%s] BatchSendRecvGroup Executor orchestrate success, take time [%lld]us.",
            param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcPodRange());
    CHK_RET(CalcPingPongHalfSize()); // ping-pong double buffering + RDMA slot
    CHK_RET(OrganizeSendItemByStream());
    CHK_RET(OrganizeRecvItemByStream());

    CHK_RET(CalcSendSlices());
    CHK_RET(CalcRecvSlices());

    CHK_RET(RunLoop(param));

    HCCL_INFO("tag[%s] BatchSendRecvGroup Executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvGroupExecutor::RunLoop

[coll_batch_send_recv_group_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_group_executor.cc#L545)

```cpp
HcclResult CollBatchSendRecvGroupExecutor::RunLoop(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream,
            dispatcher_));
    }
    CHK_RET(SetNormalModeIfDeviceDirect());
    CHK_RET(RunTasks(param));
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        HCCL_INFO("LaunchTaskExtend!");
    }
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvGroupExecutor::CalcSendLoopMaxCount

[coll_batch_send_recv_group_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_group_executor.cc#L741)

```cpp
u64 CollBatchSendRecvGroupExecutor::CalcSendLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcSendLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollBatchSendRecvGroupExecutor::CalcRecvLoopMaxCount

[coll_batch_send_recv_group_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_group_executor.cc#L750)

```cpp
u64 CollBatchSendRecvGroupExecutor::CalcRecvLoopMaxCount(const u32 unitSize) const
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = bufferSliceSize_ / unitSize;
    HCCL_INFO("[CollBatchSendRecvGroupExecutor][CalcRecvLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}
```
