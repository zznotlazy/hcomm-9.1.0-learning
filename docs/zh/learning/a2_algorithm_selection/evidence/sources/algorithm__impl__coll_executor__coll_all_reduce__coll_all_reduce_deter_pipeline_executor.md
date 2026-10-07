# coll_all_reduce_deter_pipeline_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceDeterPipelineExecutor::RunLoopInner

[coll_all_reduce_deter_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_deter_pipeline_executor.cc#L92)

```cpp
HcclResult CollAllReduceDeterPipelineExecutor::RunLoopInner(OpParam &param, const ReduceType &reduceType, ExecMem &execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    HCCL_DEBUG("[CollAllReduceDeterPipelineExecutor][RunLoopInner]inputMem[%p][%llu], outputMem[%p][%llu], " \
        "intputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET((execMem.count == 0),
        HCCL_ERROR("[CollAllReduceDeterPipelineExecutor][RunLoop]In OP_BASE curCount is zero."), HCCL_E_PARA);

    /* 设置子图复用标志 */
    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(curSize);    // override
    bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);  // override
    constexpr s64 HCCL_MEDIUM_COUNT_2_MB = 2 * 1024 * 1024;
    u64 sliceNum = (curSize / topoAttr_.userRankSize) < HCCL_MEDIUM_COUNT_2_MB ? 1 : 0 ;
    bool dataSplit = false;
    u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(autoSelectedAlgTypeLevel1,
        param.DataDes.dataType, reduceType, smallData, 1, hugeData, CopyPattern::ZCOPY, sliceNum,
        false, true, dataSplit, deterministic);
    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey(), false));

    execMem.inputMem = DeviceMem::create(execMem.inputMem.ptr(), curSize);
    execMem.outputMem = DeviceMem::create(execMem.outputMem.ptr(), curSize);

    // 执行
    HcclResult ret =  KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllReduceDeterPipelineExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
        "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d], reduce op type[%d]",
        HCCL_ERROR_CODE(ret), param.tag.c_str(), execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        execMem.count, param.DataDes.dataType, param.reduceType), ret);

    CHK_RET(LaunchTaskExtend(dispatcher_, const_cast<Stream &>(param.stream),
        const_cast<std::vector<Stream> &>(algResResp_->slaveStreams)));
    return ret;
}
```

### CollAllReduceDeterPipelineExecutor::KernelRun

[coll_all_reduce_deter_pipeline_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_deter_pipeline_executor.cc#L149)

```cpp
HcclResult CollAllReduceDeterPipelineExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] starts.",
        tag_.c_str(), topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    std::vector<Slice> bufferSlices; // 数据分成ranksize份，每份的起始偏移和大小
    CHK_RET(PrepareDataSlice(execMem, unitSize, bufferSlices));

    CHK_RET(ActiveSlaveStreams(param.stream));

    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_REDUCE_MULTI_DETERMINISTIC_PIPELINE, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType};

    CHK_RET(tempAlg->Prepare(&opInfo, execMem.inputMem, execMem.outputMem, execMem.count, bufferSlices, level0CommInfo,
        level1CommInfo, const_cast<Stream&>(param.stream), algResResp_->slaveStreams, algResResp_->notifiesMain,
        algResResp_->notifiesAux));
    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollAllReduceDeterPipelineExecutor][KernelRun] tag[%s], userRank[%u] run success.",
        tag_.c_str(), topoAttr_.userRank);
    return HCCL_SUCCESS;
}
```
