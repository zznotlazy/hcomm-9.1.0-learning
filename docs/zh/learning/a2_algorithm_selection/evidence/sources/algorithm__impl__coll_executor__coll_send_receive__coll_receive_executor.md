# coll_receive_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReceiveExecutor::Orchestrate

[coll_receive_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_receive_executor.cc#L22)

```cpp
HcclResult CollReceiveExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    // 图模式场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DeviceMem outputMem = algRes.paramOutputMem;
        ret = RunTemplate(param, outputMem);
    } else {
        ret = RunLoop(param, algRes);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReceiveExecutor][Orchestrate]errNo[0x%016llx]recv executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], Receive Executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollReceiveExecutor::RunLoop

[coll_receive_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_receive_executor.cc#L110)

```cpp
HcclResult CollReceiveExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    HcclResult ret;

    u64 commOutputSize = algRes.cclOutputMem.size();

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    auto meta = HcclOpMetaInfo::GetOneForRecieve();
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    CHK_PTR_NULL(curOutputPtr);

    u64 outputOffset = 0;
    u64 countLeft = param.DataDes.count;
    while (countLeft > 0) {
        CHK_RET(InitTask(dispatcher_, param.stream, meta.isEnableCache, meta.GetCacheKey())); 
        curOutputPtr += outputOffset;
        HCCL_DEBUG("RecvOutPlace:outputOffset[%llu]", outputOffset);
        u64 curCount = ((countLeft * unitSize) > commOutputSize) ? (commOutputSize / unitSize) : countLeft;
        u64 curSize = curCount * unitSize; // 单位 byte
        HCCL_DEBUG("RecvOutPlace:curOutputPtr[%p], curCount[%llu], curSize[%llu]", curOutputPtr, curCount, curSize);

        if(topoAttr_.deviceType != DevType::DEV_TYPE_910_93 || topoAttr_.isDiffDeviceType ||
            (topoAttr_.superPodNum > 1 || (topoAttr_.moduleNum > 1 && static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable())))) {
            // 非A3场景不做DMA消减；A3的RDMA场景，也不做DMA消减
            DeviceMem outCommMem(algRes.cclOutputMem.ptr(), curSize);
            DeviceMem outMem(curOutputPtr, curSize);
            ret = RunTemplate(param, outCommMem);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("errNo[0x%016llx] RecvOutPlace: recv error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), curOutputPtr, curCount, param.DataDes.dataType),
                ret);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, param.stream));
            HCCL_DEBUG("[CollReceiveExecutor][RunLoop]copy from ccl output to user output.");
        } else {
            // A3的SDMA场景，recv端做DMA消减
            DeviceMem outMem(curOutputPtr, curSize);
            ret = RunTemplate(param, outMem);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("errNo[0x%016llx] RecvOutPlace: recv error, tag[%s], ptr[%p], count[%llu], dataType[%d]",
                HCCL_ERROR_CODE(ret), param.tag.c_str(), curOutputPtr, curCount, param.DataDes.dataType),
                ret);
            HCCL_DEBUG("[CollReceiveExecutor][RunLoop]copy from ccl input to user output.");
        }

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("In OP_BASE curCount is zero"), HCCL_E_PARA);
        countLeft -= curCount;
        outputOffset = curSize;

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }
    return HCCL_SUCCESS;
}
```
