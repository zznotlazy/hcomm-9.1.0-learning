# coll_reduce_scatter_ring_for_910_93_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterRingFor91093Executor::CalcLevel2CommInfo

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L145)

```cpp
HcclResult CollReduceScatterRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollReduceScatterRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollReduceScatterRingFor91093Executor::CalcLoopMaxCount

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L170)

```cpp
u64 CollReduceScatterRingFor91093Executor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}
```

### CollReduceScatterRingFor91093Executor::IsHugeData

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L178)

```cpp
bool CollReduceScatterRingFor91093Executor::IsHugeData(const u64 curSize, OpParam *param)
{
    u32 level2RankSize;
    if ((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)) {
        //AHC非对称场景下没有L2
        level2RankSize =1;
    } else {
        // 多QP哈希散列开启且RDMA通信下，强制刷新子图
        // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        level2RankSize = level2CommInfo.localRankSize;
    }

    const u64 TBE_REDUCE_MAX_COUNT = INT32_MAX;

    u64 curCount = curSize / SIZE_TABLE[param->DataDes.dataType];
    bool issupportRDMAInlineReduce = IsSupportRDMAReduce(param->DataDes.dataType, param->reduceType);
    bool hugeData =
        (curSize * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
        (curSize > SDMA_SEND_MAX_SIZE) ||
        ((!isSupportSDMAReduce_) && (curCount > TBE_REDUCE_MAX_COUNT)) ||
        ((!issupportRDMAInlineReduce) && (curCount * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > TBE_REDUCE_MAX_COUNT));
    return hugeData;
}
```

### CollReduceScatterRingFor91093Executor::GetLevelCommInfo

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L343)

```cpp
HcclResult CollReduceScatterRingFor91093Executor::GetLevelCommInfo()
{
    logicalLevel0plane_ = COMM_LEVEL0;
    CHK_RET(CheckCommSize(logicalLevel0plane_, COMM_INDEX_0 + 1));
    logicalLevel0CommInfo_ = GetSubCommInfo(logicalLevel0plane_, COMM_INDEX_0);
    u32 commIndex = logicalLevel0CommInfo_.localRank;
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    logicalLevel1plane_ = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(logicalLevel1plane_, commIndex + 1));
    logicalLevel1CommInfo_ = GetSubCommInfo(logicalLevel1plane_, commIndex);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterRingFor91093Executor::KernelRun

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L572)

```cpp
HcclResult CollReduceScatterRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] executor starts, rsv[%u]", __func__, isReduceScatterV_);
    CHK_RET(GetLevelCommInfo()); // 获取通信域
    u32 perDataSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    u32 ringNum;
    u32 level0RankSize = logicalLevel0CommInfo_.localRankSize;
    bool ARSFlag = topoMatcher_->GetARSFlag();
    bool ARSDoubleRing = (ARSFlag && (level0RankSize > FACTOR_TWO) && topoAttr_.isARSDoubleRing);
    u32 sliceNum = logicalLevel0CommInfo_.localRankSize;
    u32 commIndex = logicalLevel0CommInfo_.localRank;
 
    if ((topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING && !IsUnifiedMarch(param) && !ARSFlag) || ARSDoubleRing) {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }

    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = logicalLevel1CommInfo_;
        level2CommInfo.localRankSize = 1;   // AHC bypass level2
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    }
    const u32 level2RankSize = level2CommInfo.localRankSize;
    const u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));

    // 计算slice
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    std::vector<std::vector<Slice>> level0DataSegsSlice;
    CalLevel0DataSegsSlice(execMem, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize,
        dataType, level0DataSegsSlice);

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable &&
        (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 ||
        algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CalUserMemDataSegsSlice(execMem, level0DataSegsSlice, multiStreamSlice, param, ringNum, sliceNum, level1RankSize,
        level2RankSize, dataType, perDataSize, opInfoPtr, disableDMAReduce, multRingsUserMemSlice);

    // 区分消减拷贝场景
    if ((topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || ARSDoubleRing) &&
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        // 图模式opinfo不为空
        HcomCollOpInfo graphModeOpInfo = {"", execMem.inputMem.ptr(), nullptr, param.GetDataCount(topoAttr_.userRank),
            dataType, param.root, param.reduceType, param.GetStrideCount()};
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, dataType,
            param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, &graphModeOpInfo,
            multRingsUserMemSlice, disableDMAReduce));
    } else if (opInfoPtr != nullptr && (level1RankSize > 1 || level2RankSize > 1)) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr = nullptr;
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            dataType, param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_1, 0,
            &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice, disableDMAReduce));
    } else {
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            dataType, param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr,
            multRingsUserMemSlice, disableDMAReduce));
    }
    // 对于单server图模式的最后一步需要把数据从ccl input拷贝到ccl output上
    if (level1RankSize == 1 && level2RankSize == 1 && opInfoPtr == nullptr) {
        const u64 offset = CalcSrcMemOffset(execMem, param, perDataSize);
        DeviceMem srcMem = execMem.inputMem.range(offset, execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if  (level1RankSize > 1) {
        // 节点间做reduce scatter(ring/NHR/NB)
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;

        // 计算slice
        std::vector<Slice> level1DataSegsSlice;
        CHK_RET(CalLevel1DataSegsSlice(execMem, param, logicalLevel1plane_, commIndex, sliceNum, level1RankSize,
            level2RankSize, perDataSize, level1DataSegsSlice));

        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        } else if (isSelectAHC) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(logicalLevel1plane_, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_AHC in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            HCCL_DEBUG("[CollReduceScatterRingFor91093Executor]runAsync for COMM_LEVEL1 ends");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
        }

        CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
            dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice));
        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1TempAlg, logicalLevel1CommInfo_));
    }

    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);

        // 计算slice
        std::vector<Slice> level2DataSegsSlice;
        CHK_RET(CalLevel2DataSegsSlice(execMem, param, level2RankSize, perDataSize, level2DataSegsSlice));

        std::unique_ptr<AlgTemplateBase> level2TempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr, false));
            if (algoAttr_.isSupportAtomicWrite) {
                level2TempAlg->CloseBarrier();
            }
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL2", __func__);
        } else {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL2", __func__);
        }

        CHK_RET(level2TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType,
            param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, level2DataSegsSlice));
        CHK_RET(level2TempAlg->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    }

    if (level1RankSize > 1 || level2RankSize > 1) {
        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        const u64 offset = CalcSrcMemOffset(execMem, param, perDataSize);
        DeviceMem srcMem = execMem.inputMem.range(offset, execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    HCCL_INFO("ReduceScatter ring run success, rsv[%u]", isReduceScatterV_);
    return HCCL_SUCCESS;
}
```

### CollReduceScatterRingFor91093Executor::Getlevel1CommRank

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L767)

```cpp
HcclResult CollReduceScatterRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    if (isSelectAHC) {
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

        u32 commIndex = level0CommInfo.localRank;

        CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
        CHK_RET(CheckCommSize(commPlaneLevel1, commIndex + 1));
        level1CommInfo = GetSubCommInfo(commPlaneLevel1, commIndex);
        return HCCL_SUCCESS;
    }

    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}
```

### CollReduceScatterRingFor91093Executor::SelectTempAlg

[coll_reduce_scatter_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_for_910_93_executor.cc#L792)

```cpp
HcclResult CollReduceScatterRingFor91093Executor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    HCCL_DEBUG("[CollReduceScatterRingFor91093Executor]SelectTempAlg begins");
    if (isSelectAHC) {
        CommPlane commPlaneLevel1 = COMM_LEVEL1_AHC;
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_AHC, dispatcher_);
            HCCL_INFO("reducescatter ring: using ahc algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_AHC_BROKE, dispatcher_);
            HCCL_INFO("reducescatter ring: using ahc-broke algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(NSLBDP_MIN_COUNT, globalSubGroups, ahcAlgOption));
        return HCCL_SUCCESS;
    }
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        }
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
