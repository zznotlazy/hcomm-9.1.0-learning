# coll_batch_send_recv_retry_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollBatchSendRecvRetryExecutor::Orchestrate

[coll_batch_send_recv_retry_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_retry_executor.cc#L99)

```cpp
HcclResult CollBatchSendRecvRetryExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algResource)
{
    HcclUs startut = TIME_NOW();
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBatchSendRecvRetryExecutor] batchsendrecv retry starts.");
    algResResp_ = &algResource;
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_SIZE_TWO));

    // 校验当前sendRecvPair
    std::vector<HcclSendRecvItem*> sendRecvPair;
    if (param.BatchSendRecvDataDes.curIterNum < sendRecvPairList_.size()) {
        sendRecvPair = sendRecvPairList_[param.BatchSendRecvDataDes.curIterNum];
    } else {
        HCCL_ERROR("[CollBatchSendRecvRetryExecutor] the curIterNum[%u] is out of range[0, %zu].",
            param.BatchSendRecvDataDes.curIterNum, sendRecvPairList_.size());
        return HCCL_E_PARA;
    }
    CHK_RET(CheckSendRecvPair(sendRecvPair));

    // 自发自收场景
    if (sendRecvPair.size() == PAIRSIZE_TWO && sendRecvPair[0]->remoteRank == topoAttr_.userRank &&
        sendRecvPair[1]->remoteRank == topoAttr_.userRank) {
        if (sendRecvPair[0]->count == sendRecvPair[1]->count && sendRecvPair[0]->dataType == sendRecvPair[1]->dataType) {
            u64 dataSize = sendRecvPair[0]->count * SIZE_TABLE[sendRecvPair[0]->dataType];
            DeviceMem inUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[0]->buf), dataSize);
            DeviceMem outUserMem = DeviceMem::create(static_cast<u8*>(sendRecvPair[1]->buf), dataSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outUserMem, inUserMem, param.stream));
            return HCCL_SUCCESS;
        } else {
             HCCL_ERROR("[HcclBatchSendRecvRetry] Send task and recv task to self : data size do not equal, please"\
                "check the task list.");
            return HCCL_E_PARA;
        }
    }

    // 重执行正常执行场景，前后需和控制流做同步
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: main stream record, subStream wait.");
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesAux[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(param.stream, dispatcher_, algResResp_->notifiesAux[STREAM_INDEX_1], PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesAux[STREAM_INDEX_1], PROF_STAGE_1));
    }
    // run sendrecv
    CHK_RET(RunLoop(param, algResource, sendRecvPair));

    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND_RECV) {
        HCCL_INFO("[BatchSendRecv] Stream sync: subStream record, main stream wait.");
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_0],
            PROF_STAGE_0));
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesMain[STREAM_INDEX_1], PROF_STAGE_1));
        CHK_RET(LocalNotify::Wait(param.stream, dispatcher_, algResResp_->notifiesMain[STREAM_INDEX_1],
            PROF_STAGE_1));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_0], dispatcher_,
            algResResp_->notifiesMain[STREAM_INDEX_0], PROF_STAGE_0));
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        CHK_RET(LocalNotify::Post(algResResp_->slaveStreams[STREAM_INDEX_1], dispatcher_,
            algResResp_->notifiesMain[STREAM_INDEX_1], PROF_STAGE_1));
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    HCCL_INFO("[info][print] LaunchTaskExtend success.");
    HCCL_INFO("tag[%s] BatchSendRecv Executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollBatchSendRecvRetryExecutor::RunLoop

[coll_batch_send_recv_retry_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_retry_executor.cc#L171)

```cpp
HcclResult CollBatchSendRecvRetryExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes,
    const std::vector<HcclSendRecvItem*> &sendRecvPair)
{
    // 判断当前需执行的算子
    std::vector<HcclSendRecvItem*> curSendRecvPair;
    if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
        curSendRecvPair.push_back(sendRecvPair[0]);
    } else if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::RECV) {
        curSendRecvPair.push_back(sendRecvPair[sendRecvPair.size() - 1]);
    } else {
        curSendRecvPair = sendRecvPair;
    }

    // 执行当前需执行的算子
    for (const auto& itemPtr : curSendRecvPair) {
        if (static_cast<bool>(param.BatchSendRecvDataDes.isDirectRemoteRank[itemPtr->remoteRank])) {
            // device direct链路的任务会在host侧下发，此处需要跳过
            continue;
        }
        HCCL_INFO("[CollBatchSendRecvRetryExecutor][RunLoop] remoteRank %u", itemPtr->remoteRank);
        if (itemPtr->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            CHK_RET(CalcSendSlices(algRes, itemPtr));
        } else if (itemPtr->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            CHK_RET(CalcRecvSlices(algRes, itemPtr));
        } else {
            HCCL_ERROR("[CollBatchSendRecvRetryExecutor][RunLoop] sendRecvType is Wrong.");
            return HCCL_E_PARA;
        }
    }

    u32 loopInOnceLaunch = 0;
    // 每隔200个loop launch一次
    while (!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if(!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(algResResp_->slaveStreams[STREAM_INDEX_0], false, true)); 
            sendDataSilces_.pop_front();
        }
        if(!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(algResResp_->slaveStreams[STREAM_INDEX_1], true));
            recvDataSilces_.pop_front();
        }
        loopInOnceLaunch++;
        if (loopInOnceLaunch == MAX_LOOP_IN_ONCE_LAUNCH || (sendDataSilces_.empty() && recvDataSilces_.empty())) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
            HCCL_INFO("[BatchSendRecv] LaunchTaskExtend, unprocessed send slices[%u], recv slices[%u].",
                sendDataSilces_.size(), recvDataSilces_.size());
            loopInOnceLaunch = 0;
        }
    }
    return HCCL_SUCCESS;
}
```
