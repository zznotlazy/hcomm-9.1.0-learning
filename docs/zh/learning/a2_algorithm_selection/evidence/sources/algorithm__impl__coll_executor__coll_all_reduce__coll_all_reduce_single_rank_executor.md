# coll_all_reduce_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceSingleRankExecutor::KernelRun

[coll_all_reduce_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_single_rank_executor.cc#L21)

```cpp
HcclResult CollAllReduceSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceSingleRankExecutor][KernelRun] AllReduce single rank");
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType =
        ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;
    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = totalSize > SDMA_SEND_MAX_SIZE;
    if (execMem.inputPtr == execMem.outputPtr) {
        auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            totalSize <= HCCL_SMALL_COUNT_128_KB, 1, hugeData, CopyPattern::ZCOPY); // 通过CopyPattern字段区分不同子图
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else {
        auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            totalSize <= HCCL_SMALL_COUNT_128_KB, 1, hugeData, CopyPattern::BCOPY); // 通过CopyPattern字段区分不同子图
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        // ranksize = 1; input、output地址不同，input->output
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    return HCCL_SUCCESS;
}
```
