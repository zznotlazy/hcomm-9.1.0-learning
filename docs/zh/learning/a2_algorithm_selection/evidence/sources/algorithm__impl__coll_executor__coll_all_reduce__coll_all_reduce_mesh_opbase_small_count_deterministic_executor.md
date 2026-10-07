# coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::CalcLevel1CommInfo

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L86)

```cpp
HcclResult CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] start", __func__, tag_.c_str());
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX);
    if (IsPowerOfTwo(topoAttr_.moduleNum) || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc HDCommInfo", __func__, tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc RingCommInfo", __func__, tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc NHRCommInfo", __func__, tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[commParaLevel1.commPlane], inputType, outputType));
    HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc CommInfo Finish", __func__, tag_.c_str());

    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::CalcLoopMaxCount

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L108)

```cpp
u64 CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop;
    if (IsPowerOfTwo(topoAttr_.deviceNumPerAggregation)) {
        // local doubling SDMA cclin->cclout 一定是字节对齐的
        maxCountPerLoop = cclBuffSize / unitSize;
    } else {
        // template-localreduce_bcast 没有 128B对齐
        maxCountPerLoop = cclBuffSize / unitSize / (topoAttr_.deviceNumPerAggregation - 1);
    }
    return maxCountPerLoop;
}
```

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::KernelRun

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L134)

```cpp
HcclResult CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllReduceMeshOpbaseSmallCountDeterministicExecutor][Run]CollAllReduceMeshOpbaseSmallCountDeterministicExecutor begins.");

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    // Run level0
    if (IsPowerOfTwo(level0CommInfo.localRankSize)) {
        // userin->cclin
        DeviceMem userInMem(execMem.inputPtr, curSize);
        DeviceMem cclInMem = execMem.inputMem.range(0, curSize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, cclInMem, userInMem, const_cast<Stream&>(param.stream)));
        CHK_RET(RunDoublingSingleLevel(param, reduceAttr, execMem, level0CommInfo));
        HCCL_INFO("allreduce small count deterministic: using doubling algo intra-server.");
    } else {
        HcomCollOpInfo opInfo = {
            "", execMem.inputPtr, execMem.inputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType};
        CHK_RET(RunReduceBcastSingleLevel(param, opInfo, reduceAttr, execMem, level0CommInfo));
        HCCL_INFO("allreduce small count deterministic: using reduce bcast algo intra-server.");
    }
    // Run level1
    if (IsPowerOfTwo(level1CommInfo.localRankSize)) {
        CHK_RET(RunDoublingSingleLevel(param, reduceAttr, execMem, level1CommInfo));
        HCCL_INFO("allreduce small count deterministic: using doubling algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        CHK_RET(RunTempLevel1(TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, param, reduceAttr, execMem, 
            level1CommInfo));
        HCCL_INFO("allreduce small count deterministic: using rhd algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        CHK_RET(RunTempLevel1(TemplateType::TEMPLATE_ALL_REDUCE_RING, param, reduceAttr, execMem, level1CommInfo));
        HCCL_INFO("allreduce small count deterministic: using default ring algo inter-server.");
    } else {
        // 默认nhr
        CHK_RET(RunTempLevel1(TemplateType::TEMPLATE_ALL_REDUCE_NHR, param, reduceAttr, execMem, level1CommInfo));
        HCCL_INFO("allreduce small count deterministic: using nhr algo inter-server.");
    }
    DeviceMem dstMem(execMem.outputPtr, curSize);
    DeviceMem srcMem;
    if (IsPowerOfTwo(level1CommInfo.localRankSize)) {
        // cclin->userout
        srcMem = execMem.inputMem.range(0, curSize);
    } else {
        // cclout->userout
        srcMem = execMem.outputMem.range(0, curSize);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunDoublingSingleLevel

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L192)

```cpp
HcclResult CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunDoublingSingleLevel(const OpParam &param, u64 reduceAttr, ExecMem &execMem,
                                                        SubCommInfo &levelCommInfo)
{
    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_DOUBLING_LOCAL_REDUCE,
        dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));
    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
        (levelCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + levelCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, levelCommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunReduceBcastSingleLevel

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L212)

```cpp
HcclResult CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunReduceBcastSingleLevel(const OpParam &param, HcomCollOpInfo& opInfo, u64 reduceAttr, ExecMem &execMem,
                                                        SubCommInfo &levelCommInfo)
{
    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_LOCAL_REDUCE_BCAST, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        levelCommInfo.localRank, levelCommInfo.localRankSize, topoAttr_.userRank, &opInfo));
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(tempAlg->RegisterProfiler(
            (levelCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + levelCommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(tempAlg, levelCommInfo));
    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunTempLevel1

[coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L231)

```cpp
HcclResult CollAllReduceMeshOpbaseSmallCountDeterministicExecutor::RunTempLevel1(const TemplateType type, 
    const OpParam &param, u64 reduceAttr, ExecMem &execMem, SubCommInfo &level1CommInfo)
{
    std::unique_ptr<AlgTemplateBase> tempAlg;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(type, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(reduceAttr));

    CHK_RET(tempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));
    CHK_RET(tempAlg->RegisterProfiler((level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        level1CommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, level1CommInfo));
    if (type == TemplateType::TEMPLATE_ALL_REDUCE_NHR) {
        tempAlg->CloseBarrier();
    }
    return HCCL_SUCCESS;
}
```
