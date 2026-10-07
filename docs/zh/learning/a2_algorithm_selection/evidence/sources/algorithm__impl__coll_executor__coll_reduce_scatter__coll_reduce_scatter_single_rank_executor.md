# coll_reduce_scatter_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterSingleRankExecutor::KernelRun

[coll_reduce_scatter_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_single_rank_executor.cc#L21)

```cpp
HcclResult CollReduceScatterSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterSingleRankExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType =
        ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE : ReduceType::TBE_REDUCE;

    auto originalAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = totalSize > SDMA_SEND_MAX_SIZE;
    bool smallData = totalSize <= HCCL_SMALL_COUNT_32_KB;
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    if (execMem.inputPtr == execMem.outputPtr) {
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            hugeData, smallData, CopyPattern::ZCOPY, false, deterministic, false); // 通过CopyPattern字段区分不同的子图
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    } else { // ranksize = 1; input、output地址不同，input->output
        auto opMeta = HcclOpMetaInfo::GetOneForReduceScatter(originalAlgTypeLevel1, param.DataDes.dataType, reduceType,
            hugeData, smallData, CopyPattern::BCOPY, false, deterministic, false);
        CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
        DeviceMem srcMem(execMem.inputPtr, totalSize);
        DeviceMem dstMem(execMem.outputPtr, totalSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams));

    return HCCL_SUCCESS;
}
```
