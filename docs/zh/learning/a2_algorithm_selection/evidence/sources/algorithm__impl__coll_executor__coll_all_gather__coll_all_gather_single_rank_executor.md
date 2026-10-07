# coll_all_gather_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherSingleRankExecutor::KernelRun

[coll_all_gather_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_single_rank_executor.cc#L20)

```cpp
HcclResult CollAllGatherSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherSingleRankExecutor][KernelRun]AllGather single rank");
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = (execMem.count * unitSize) > SDMA_SEND_MAX_SIZE;
    if (execMem.inputPtr == execMem.outputPtr) {
        // 通过CopyPattern字段区分不同的子图
        auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, false, CopyPattern::ZCOPY);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllGather(originalAlgTypeLevel1, hugeData, false, CopyPattern::BCOPY);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        // ranksize = 1; input、output地址不同，input->output
        DeviceMem srcMem(execMem.inputPtr, execMem.count * unitSize);
        DeviceMem dstMem(execMem.outputPtr, execMem.count * unitSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams));

    return HCCL_SUCCESS;
}
```
