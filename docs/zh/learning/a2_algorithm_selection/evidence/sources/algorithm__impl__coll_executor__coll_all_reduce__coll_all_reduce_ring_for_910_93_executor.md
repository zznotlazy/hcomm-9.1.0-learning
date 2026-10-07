# coll_all_reduce_ring_for_910_93_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceRingFor91093Executor::CalcLevel2CommInfo

[coll_all_reduce_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L103)

```cpp
HcclResult CollAllReduceRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollAllReduceRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }
    
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s]Calc HDCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollAllReduceRingFor91093Executor::GetLevelCommInfo

[coll_all_reduce_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L154)

```cpp
HcclResult CollAllReduceRingFor91093Executor::GetLevelCommInfo()
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

### CollAllReduceRingFor91093Executor::KernelRun

[coll_all_reduce_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L209)

```cpp
HcclResult CollAllReduceRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The CollAllReduceRingFor91093Executor starts", __func__);
    CHK_RET(ActiveSlaveStreams(param.stream));
    CHK_RET(GetLevelCommInfo()); // 获取通信域
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multRingsSliceZero; // 数据基于该rank上环0的偏移
    u32 sliceNum = logicalLevel0CommInfo_.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // 构造ring algorithm对应的reduce-scatter实例
    std::vector<u32> mockNicList = topoAttr_.nicList;
    CHK_RET(GetNicList(mockNicList));
    u32 level0RankSize = logicalLevel0CommInfo_.localRankSize;
    bool ARSFlag = topoMatcher_->GetARSFlag();
    bool ARSDoubleRing = (ARSFlag && (level0RankSize > FACTOR_TWO) && topoAttr_.isARSDoubleRing);

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || ARSDoubleRing) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, mockNicList, logicalLevel0plane_);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }

    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo reduceScatterGraphModeOpInfo = {
        "", execMem.inputMem.ptr(), nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        reduceScatterOpInfoPtr = &reduceScatterGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable &&
        (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 ||
        algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);
    const std::vector<std::vector<Slice>> multRingsUserMemSliceDefault = std::vector<std::vector<Slice>>(0);
    CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr, multRingsUserMemSliceDefault, disableDMAReduce));
    HCCL_INFO("AllReduce double ring stage0 run success.");

    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    if(ARSFlag && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) ){
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }

    /* 三步算法step2: 内层 - 节点间 allreduce */
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    if (ARSFlag) {
        CHK_RET(PrepareARSLevel1CommInfo(segmentIdx, commIndex, hdSize, logicalLevel0CommInfo_, multRingsSliceZero, param.tag, mockNicList));
    } else {
        CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize, logicalLevel0CommInfo_, multRingsSliceZero, param.tag));
    }
    if (ARSDoubleRing && reduceScatterOpInfoPtr == nullptr) {
        DeviceMem srcMem = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        DeviceMem dstMem = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
    }

    u64 hdCount = hdSize / perDataSize;
    if (topoAttr_.superPodNum <= 1 || isSelectAHC) {
        DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_SMART_PTR_NULL(allreduceOutput);

        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, 
                dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL1", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_V1 in COMM_LEVEL1", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (isSelectAHC) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(logicalLevel1plane_, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_AHC in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, 
                dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NB in COMM_LEVEL1", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
            HCCL_DEBUG("allreduce ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]",
                curSize, logicalLevel0CommInfo_.localRankSize, logicalLevel0CommInfo_.localRankSize);                    
            if (curSize / logicalLevel0CommInfo_.localRankSize <= NHR_ALLREDUCE_SMALL_SIZE) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR in COMM_LEVEL1", __func__);
            }
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else {
            HCCL_ERROR("AllReduce ring: algType_[%u] is not supported.", algType_.algoLevel1);
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        u32 rankSize = logicalLevel1CommInfo_.localRankSize;
        // 节点间的hd 使用环0来记录
        CHK_RET(level1TempAlg->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, hdCount,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_RET(level1TempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1TempAlg, logicalLevel1CommInfo_));

        HCCL_INFO("AllReduce double ring stage1 run success");
    } else {
        // 超节点内做reducescatter
        CHK_RET(CheckCommSize(logicalLevel1plane_, commIndex + 1));
        u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;
        u64 level1Offset = dataSegsSlice[segmentIdx].offset;

        // 根据数据量计算每个环上数据的偏移和大小
        CHK_RET(AlgTemplateBase::PrepareSliceData(hdCount, perDataSize, level1RankSize, 0, dataSegsSlice));
        DeviceMem reducescatterInput = execMem.inputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterInput);
        DeviceMem reducescatterOutput = execMem.outputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterOutput);
        if (level1RankSize > 1) {
            u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
                param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level1RSTempAlg;

            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr, false));
            } else {
                HCCL_ERROR("ReduceScatter ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_RET(level1RSTempAlg->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, hdCount, param.DataDes.dataType,
                param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));

            CHK_RET(level1RSTempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1RSTempAlg, logicalLevel1CommInfo_));
            HCCL_INFO("AllReduce double ring [superpod] level1 ReduceScatter run success");
        }

        // 超节点间做allreduce
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        u32 rankSize = level2CommInfo.localRankSize;
        u32 localRank = logicalLevel1CommInfo_.localRank;

        DeviceMem allreduceInput =
            reducescatterInput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput =
            reducescatterOutput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(allreduceOutput);

        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

        std::unique_ptr<AlgTemplateBase> level2ARTempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR in COMM_LEVEL2", __func__);
            if (algoAttr_.isSupportAtomicWrite) {
                CHK_SMART_PTR_NULL(level2ARTempAlg);
                level2ARTempAlg->CloseBarrier();
            }
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL2", __func__);
        } else {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING in COMM_LEVEL2", __func__);
        }
        CHK_SMART_PTR_NULL(level2ARTempAlg);
        CHK_RET(level2ARTempAlg->Prepare(reduceAttr));

        u64 arCount = dataSegsSlice[localRank].size / perDataSize;
        CHK_RET(level2ARTempAlg->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, arCount,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[localRank].offset + level1Offset));
        CHK_RET(level2ARTempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2ARTempAlg, level2CommInfo));
        HCCL_INFO("AllReduce double ring [superpod] level2 AllReduce run success");

        // 超节点内做allgather
        if (level1RankSize > 1) {
            std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
            DeviceMem allgatherInput = execMem.outputMem.range(level1Offset, hdSize);
            DeviceMem allgatherOutput = execMem.outputMem.range(level1Offset, hdSize);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, 
                    dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
            } else {
                HCCL_ERROR("AllGather ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1AGTempAlg);
            CHK_RET(level1AGTempAlg->Prepare(allgatherInput, allgatherOutput, allgatherOutput, arCount,
                param.DataDes.dataType, param.stream,
                HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
            CHK_RET(level1AGTempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1AGTempAlg, logicalLevel1CommInfo_));
            HCCL_INFO("AllReduce double ring [superpod] level1 AllGather run success");
        }
    }
    /* 三步算法step3：外层 - 节点内 allgather */
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo allgatherOpInfoGraphModeOpInfo = {
        "", nullptr, execMem.outputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        allgatherOpInfoPtr = &allgatherOpInfoGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount,
        param.DataDes.dataType, multRingsSliceZero, param.stream,
        PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("AllReduce double ring stage2 run success");
    return HCCL_SUCCESS;
}
```

### CollAllReduceRingFor91093Executor::Getlevel1CommRank

[coll_all_reduce_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L504)

```cpp
HcclResult CollAllReduceRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
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

### CollAllReduceRingFor91093Executor::SelectTempAlg

[coll_all_reduce_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L527)

```cpp
HcclResult CollAllReduceRingFor91093Executor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isSelectAHC) {
        CommPlane commPlaneLevel1 = COMM_LEVEL1_AHC;
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
            HCCL_INFO("allreduce ring: using ahc algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
            HCCL_INFO("allreduce ring: using ahc-broke algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(NSLBDP_MIN_COUNT, globalSubGroups, ahcAlgOption));
        return HCCL_SUCCESS;
    }
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_INFO("AllReduce ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_INFO("AllReduce ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("AllReduce ring: using ring algo inter-superPod.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllReduce ring: using halving-doubling algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
