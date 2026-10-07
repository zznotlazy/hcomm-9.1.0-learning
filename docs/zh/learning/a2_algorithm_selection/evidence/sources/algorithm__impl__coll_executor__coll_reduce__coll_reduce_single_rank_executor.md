# coll_reduce_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceSingleRankExecutor::KernelRun

[coll_reduce_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_single_rank_executor.cc#L21)

```cpp
HcclResult CollReduceSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceSingleRankExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    ReduceType reduceType =
        ((param.reduceType != HCCL_REDUCE_PROD) && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)) ?
        ReduceType::INLINE_REDUCE :
        ReduceType::TBE_REDUCE;
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool isRootRank = param.root == topoAttr_.realUserRank ? true : false;
    bool hugeData = IsHugeData(totalSize); // override
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();

    auto opMeta = HcclOpMetaInfo::GetOneForReduce(isRootRank, param.root, autoSelectedAlgTypeLevel1,
        param.DataDes.dataType, reduceType, hugeData, deterministic);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    DeviceMem srcMem(execMem.inputPtr, totalSize);
    DeviceMem dstMem(execMem.outputPtr, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream &>(param.stream)));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream &>(param.stream)));
    return HCCL_SUCCESS;
}
```
