# coll_scatter_single_rank_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollScatterSingleRankExecutor::KernelRun

[coll_scatter_single_rank_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_single_rank_executor.cc#L20)

```cpp
HcclResult CollScatterSingleRankExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollScatterSingleRankExecutor][KernelRun] starts.");
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    bool hugeData = IsHugeData(totalSize); // override
    auto opMeta = HcclOpMetaInfo::GetOneForScatter( param.root, hugeData);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    DeviceMem srcMem(execMem.inputPtr, totalSize);
    DeviceMem dstMem(execMem.outputPtr, totalSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream &>(param.stream)));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream &>(param.stream)));
    return HCCL_SUCCESS;
}
```
