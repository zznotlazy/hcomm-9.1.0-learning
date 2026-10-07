# coll_batch_send_recv_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollBatchSendRecvExecutor::Orchestrate

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L150)

```cpp
HcclResult CollBatchSendRecvExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvExecutor] batchsendrecv starts.");

    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));
    CHK_RET(GetPairWiseList(param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
    CHK_RET(ProcessSelfSendRecvTasks(param.stream));
    if (topoAttr_.userRankSize == 1) {
        HCCL_INFO("tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.",
            param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
        return HCCL_SUCCESS;
    }
    CHK_RET(CalcSendSlices(algResource));
    CHK_RET(CalcRecvSlices(algResource));

    HCCL_DEBUG("[CollBatchSendRecvExecutor][Orchestrate] aicpuUnfoldMode %d", aicpuUnfoldMode_);
    if(aicpuUnfoldMode_) {
        CHK_RET(RunLoopInAicpuUnfoldMode(param));
    } else {
        CHK_RET(RunLoopInHostUnfoldMode(param));
    }
    HCCL_INFO("tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L218)

```cpp
HcclResult CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode(OpParam& param)
{
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        auto meta = HcclOpMetaInfo::GetOneForBatchSendRecv();
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey()));
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem, algResResp_->cclOutputMem, param.stream,
            dispatcher_));
    }
    bool isSetNormalMode = false; // 设置过一次就不需要再设置了
    for (u32 i = 0; i < sendDataSilces_.size(); ++i) {
        SendRecvSlice& slice = sendDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetSendTargetLink(slice.remoteRank, targetLink));
        if (TransportType::TRANS_TYPE_DEVICE_DIRECT == targetLink->GetTransportType()) {
            CHK_RET(SetNormalMode(dispatcher_));
            isSetNormalMode = true;
            HCCL_INFO("[CollBatchSendRecvExecutor][RunLoopInHostUnfoldMode]Send Set NormalMode true");
            break;
        }
    }

    for (u32 i = 0; i < recvDataSilces_.size() && !isSetNormalMode; ++i) {
        SendRecvSlice& slice = recvDataSilces_[i];
        LINK targetLink;
        CHK_RET(GetRecvTargetLink(slice.remoteRank, targetLink));
        if (targetLink->GetTransportType() == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            CHK_RET(SetNormalMode(dispatcher_));
            HCCL_INFO("[CollBatchSendRecvExecutor][RunLoopInHostUnfoldMode]Recv Set NormalMode dispatcher");
            break;
        }
    }

    CHK_RET(MainPostSubWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if(!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(param.stream, false, false));
            sendDataSilces_.pop_front();
        }
        if(!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false));
            recvDataSilces_.pop_front();
        }
    }

    CHK_RET(SubPostMainWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
    if (static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts())) {
        // 多流子图前后需加空拷贝
        CHK_RET(AlgTemplateBase::ExecEmptyTask(algResResp_->cclInputMem,
            algResResp_->cclOutputMem, param.stream, dispatcher_));
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvExecutor::RunLoopInAicpuUnfoldMode

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L310)

```cpp
HcclResult CollBatchSendRecvExecutor::RunLoopInAicpuUnfoldMode(OpParam& param)
{
    CHK_RET(MainPostSubWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    u32 loopInOnceLaunch = 0;
    // 每隔200个loop launch一次
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if(!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(param.stream, false, false));
            sendDataSilces_.pop_front();
        }
        if(!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false));
            recvDataSilces_.pop_front();
        }
        loopInOnceLaunch++;
        if (loopInOnceLaunch == MAX_LOOP_IN_ONCE_LAUNCH) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
            loopInOnceLaunch = 0;
            HCCL_INFO("[BatchSendRecv] LaunchTaskExtend, unprocessed send slices[%u], recv slices[%u].",
                sendDataSilces_.size(), recvDataSilces_.size());
        }
    }
    CHK_RET(SubPostMainWait(param.stream, algResResp_->slaveStreams[STREAM_INDEX_0]));
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvExecutor::SendKernelRun

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L447)

```cpp
HcclResult CollBatchSendRecvExecutor::SendKernelRun(Stream& stream, ExecMem &execMem, u32 remoteUserRank,
    bool retryEnable)
{
    LINK targetLink;
    CHK_RET(GetSendTargetLink(remoteUserRank, targetLink));
    SendReceive executor(dispatcher_, targetLink, INVALID_VALUE_RANKID, HCCL_CHUNK_SIZE, retryEnable);
    CHK_RET(executor.SendPrepare(execMem.inputMem, remoteUserRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.BatchSendRunAsync());

    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvExecutor::RecvKernelRun

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L460)

```cpp
HcclResult CollBatchSendRecvExecutor::RecvKernelRun(Stream& stream, ExecMem &execMem, u32 remoteUserRank,
    bool retryEnable)
{
    LINK targetLink;
    CHK_RET(GetRecvTargetLink(remoteUserRank, targetLink));
    SendReceive executor(dispatcher_, targetLink, INVALID_VALUE_RANKID, HCCL_CHUNK_SIZE, retryEnable);
    CHK_RET(executor.ReceivePrepare(execMem.outputMem, remoteUserRank, stream));
    CHK_RET(executor.RegisterProfiler(0, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));
    CHK_RET(executor.BatchReceiveRunAsync());
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvExecutor::CalcSendLoopMaxCount

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L495)

```cpp
u64 CollBatchSendRecvExecutor::CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的input count
    u64 maxCountPerLoop = inCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcSendLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollBatchSendRecvExecutor::CalcRecvLoopMaxCount

[coll_batch_send_recv_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L504)

```cpp
u64 CollBatchSendRecvExecutor::CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = outCCLBuffer.size() / unitSize;
    HCCL_WARNING("[CollBatchSendRecvExecutor][CalcRecvLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}
```
