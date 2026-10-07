# coll_send_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollSendExecutor::Orchestrate

[coll_send_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_send_executor.cc#L22)

```cpp
HcclResult CollSendExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DeviceMem InputMem = algRes.paramInputMem;
        ret = RunTemplate(param, InputMem);
    } else {
        ret = RunLoop(param, algRes);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollSendExecutor][Orchestrate]errNo[0x%016llx]send executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s] Send Executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollSendExecutor::RunLoop

[coll_send_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_send_executor.cc#L130)

```cpp
HcclResult CollSendExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    HcclResult ret;

    u64 commInputSize = algRes.cclInputMem.size();

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    auto meta = HcclOpMetaInfo::GetOneForSend();
    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    CHK_PTR_NULL(curInputPtr);

    u64 inputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), meta.isEnableCache, meta.GetCacheKey()));
        curInputPtr += inputOffset;

        HCCL_DEBUG("SendOutPlace:inputOffset[%llu]", inputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("SendOutPlace:curInputPtr[%p], curCount[%llu], curSize[%llu]", curInputPtr, curCount, curSize);

        // 非A3场景，或A3的SDMA消减场景，send端不消减
        DeviceMem inCommMem(algRes.cclInputMem.ptr(), curSize);
        DeviceMem inMem(curInputPtr, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, inCommMem, inMem, const_cast<Stream&>(param.stream)));
        HCCL_INFO("[CollSendExecutor] inCommMem[%p]size[%lld], inMem[%p]size[%lld]", inCommMem.ptr(), inCommMem.size(), inMem.ptr(), inMem.size());
        ret = RunTemplate(param, inCommMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("errNo[0x%016llx] SendOutPlace: send error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), curInputPtr, curCount, param.DataDes.dataType),
            ret);
        HCCL_DEBUG("[CollSendExecutor][RunLoop]copy from user input to ccl input.");

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        inputOffset = curSize;

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}
```
