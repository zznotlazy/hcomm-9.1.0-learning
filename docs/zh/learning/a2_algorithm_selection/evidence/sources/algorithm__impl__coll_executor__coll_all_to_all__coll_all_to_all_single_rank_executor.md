# coll_all_to_all_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAlltoAllSingleRankExecutor::Orchestrate

[coll_all_to_all_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_single_rank_executor.cc#L20)

```cpp
HcclResult CollAlltoAllSingleRankExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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

    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllSingleRankExecutor][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_INFO("tag[%s], AlltoAllSingleRankExecutor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAlltoAllSingleRankExecutor::KernelRun

[coll_all_to_all_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_single_rank_executor.cc#L50)

```cpp
HcclResult CollAlltoAllSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAlltoAllSingleRankExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u64 sendCount  = 0 ;
    u64 totalSize = 0 ;
    u64 unitSize = SIZE_TABLE[param.All2AllDataDes.sendType] ;
    CopyPattern copyPattern = (execMem.inputPtr == execMem.outputPtr) ? CopyPattern::ZCOPY : CopyPattern::BCOPY ;
    HcclOpMetaInfo opMeta ;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC){
        sendCount = *(static_cast<u64 *>(param.All2AllDataDes.sendCountMatrix));
        totalSize = sendCount * unitSize ; 
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE); 
    } else {
        sendCount = *(static_cast<u64 *>(param.All2AllDataDes.sendCounts))  ;
        totalSize = sendCount * unitSize ; 
        opMeta = HcclOpMetaInfo::GetOneForAllToAllVC(copyPattern, totalSize, totalSize > SDMA_SEND_MAX_SIZE); 
    }
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    
    if (execMem.inputPtr != execMem.outputPtr) {
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    } 
    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams));
    return HCCL_SUCCESS;
}
```
