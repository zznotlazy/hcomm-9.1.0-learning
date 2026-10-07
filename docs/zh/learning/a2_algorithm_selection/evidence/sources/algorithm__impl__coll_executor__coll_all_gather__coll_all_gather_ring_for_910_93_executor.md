# coll_all_gather_ring_for_910_93_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherRingFor91093Executor::CalcLevel2CommInfo

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L82)

```cpp
HcclResult CollAllGatherRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    HCCL_DEBUG("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo]Level2CommInfo start set");
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo.", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo.", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo.", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}
```

### CollAllGatherRingFor91093Executor::CalcLoopMaxCount

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L107)

```cpp
u64 CollAllGatherRingFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}
```

### CollAllGatherRingFor91093Executor::GetLevelCommInfo

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L263)

```cpp
HcclResult CollAllGatherRingFor91093Executor::GetLevelCommInfo()
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

### CollAllGatherRingFor91093Executor::KernelRun

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L277)

```cpp
HcclResult CollAllGatherRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] The AllGatherRingExecutor starts, topoType_[%u], agv[%u]",
        __func__, topoType_, isAllGatherV_);
    CHK_RET(GetLevelCommInfo()); // 设置逻辑通信域
    CHK_RET(ActiveSlaveStreams(param.stream));
    const HcclDataType dataType = param.GetDataType();
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str()), HCCL_E_PARA);

     bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    
    u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;

    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = logicalLevel1CommInfo_;
        level2CommInfo.localRankSize = 1;   // AHC bypass level2
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    }
    const u32 level2RankSize = level2CommInfo.localRankSize;

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = CalcDstMemOffset(param, perDataSize, inputMemSize);
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo *opInfoPtr = (DMAReduceFlag_ || (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)) ? &opInfo :
        nullptr;

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]AllGather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    } else {
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1 || level2RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]AllGather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
        }
    }
    if (level2RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level2AGExecutor;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL2", __func__);
        } else {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL2", __func__);
        }
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice = PrepareSlicesL2(param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_,
            perDataSize, inputMemSize);
        CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            dataType, param.stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler((
            level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO("AllGather ring [superpod] level2 AllGather run successtopoType_[%u], agv[%u]",
            topoType_, isAllGatherV_);
    }
    if (level1RankSize > 1) {
        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice = PrepareSlicesL1(param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_,
            perDataSize, inputMemSize);

        std::unique_ptr<AlgTemplateBase> level1AGExecutor;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else if (isSelectAHC) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(logicalLevel1plane_, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC in COMM_LEVEL1", __func__);
            } else {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            CHK_SMART_PTR_NULL(level1AGExecutor);
            CHK_RET(level1AGExecutor->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
        } else {
            HCCL_ERROR("AllGather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);
        CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            dataType, param.stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler((
            level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level1AGExecutor, logicalLevel1CommInfo_));
        HCCL_INFO("AllGather ring [superpod] level1 AllGather run successtopoType_[%u], agv[%u]",
            topoType_, isAllGatherV_);
    }
    // 节点内做AllGather ring
    std::vector<std::vector<Slice>> multRingsSlice;
    CHK_RET(PrepareSlicesL0(multRingsSlice, param, level2CommInfo, logicalLevel1CommInfo_, logicalLevel0CommInfo_, perDataSize,
        inputMemSize));

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CHK_RET(PrepareUserMemSlices(multRingsUserMemSlice, multRingsSlice, param, level2CommInfo, logicalLevel1CommInfo_,
        logicalLevel0CommInfo_, perDataSize, inputMemSize));

    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, dataType,
        multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("AllGather ring run success. topoType_[%u], agv[%u]", topoType_, isAllGatherV_);
    return HCCL_SUCCESS;
}
```

### CollAllGatherRingFor91093Executor::Getlevel1CommRank

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L430)

```cpp
HcclResult CollAllGatherRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    HCCL_INFO("[CollAllGatherRingFor91093Executor][Getlevel1CommRank] Entry Getlevel1CommRank.");
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isSelectAHC) {
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
        u32 level0ServerIndex = level0CommInfo.localRank;

        CommPlane commPlaneLevel1 = COMM_LEVEL1;
        CHK_RET(CheckCommSize(commPlaneLevel1, level0ServerIndex + 1));
        level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0ServerIndex);
        u32 level1RankSize = level1CommInfo.localRankSize;
        HCCL_INFO("Getlevel1CommRank. level1RankSize[%u]", level1RankSize);
        return HCCL_SUCCESS;
    }
    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        HCCL_INFO("[nslbdp] Getlevel1CommRank size not match.");
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}
```

### CollAllGatherRingFor91093Executor::SelectTempAlg

[coll_all_gather_ring_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_for_910_93_executor.cc#L455)

```cpp
HcclResult CollAllGatherRingFor91093Executor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    HCCL_INFO("[nslbdp] Entry SelectTempAlg, level1RankSize = [%u].", level1RankSize);
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
                        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    if (isSelectAHC) {
        CommPlane commPlaneLevel1 = COMM_LEVEL1;
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
            HCCL_INFO("allgather comm: using ahc algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
            HCCL_INFO("allgather comm: using ahc-broke algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(NSLBDP_MIN_COUNT, globalSubGroups, ahcAlgOption));
        return HCCL_SUCCESS;
    }
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather ring: using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
