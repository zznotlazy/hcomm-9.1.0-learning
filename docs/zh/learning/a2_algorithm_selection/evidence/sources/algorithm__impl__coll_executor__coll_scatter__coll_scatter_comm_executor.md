# coll_scatter_comm_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollScatterCommExecutor::KernelRun

[coll_scatter_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_comm_executor.cc#L20)

```cpp
HcclResult CollScatterCommExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollScatterCommExecutor] scatter starts.");
    DeviceMem& inputMem = execMem.inputMem;
    DeviceMem& outputMem = execMem.outputMem;
    u64 count = execMem.count;
    auto root = param.root;
    auto dataType = param.DataDes.dataType;
    Stream& stream = const_cast<Stream&>(param.stream);
    u32 userRank = topoAttr_.userRank;

    u32 commIndex = COMM_INDEX_0;
    // 统一走server间
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CHK_RET(CheckCommSize(commPlane, COMM_INDEX_0 + 1));
    SubCommInfo combinedCommInfo = GetSubCommInfo(commPlane, COMM_INDEX_0);

    CHK_RET(KernelRunLevel1(inputMem, count, dataType, commIndex, root, userRank, commPlane, stream));

    // 将scratchMem赋值给outputMem
    u8 *inputMemPtr = static_cast<u8 *>(inputMem.ptr());
    CHK_PTR_NULL(inputMemPtr);
    DeviceMem resultMem(inputMemPtr + outputMem.size() * combinedCommInfo.localRank, outputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem, resultMem, stream));
    return HCCL_SUCCESS;
}
```

### CollScatterCommExecutor::CalcCombinedCommInfo

[coll_scatter_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_comm_executor.cc#L60)

```cpp
HcclResult CollScatterCommExecutor::CalcCombinedCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommPlane commPlane = COMM_COMBINE;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        commPlane = COMM_COMBINE_ORDER;
    }

    CommParaInfo commParaInfo(commPlane, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaInfo.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
    } else {
        commParaInfo.commType = CommType::COMM_TAG_RING_INNER;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[commPlane], inputType, outputType));
    return HCCL_SUCCESS;
}
```
