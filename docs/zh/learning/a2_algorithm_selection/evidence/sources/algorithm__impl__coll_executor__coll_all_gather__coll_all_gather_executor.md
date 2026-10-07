# coll_all_gather_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherExecutor::Orchestrate

[coll_all_gather_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L21)

```cpp
HcclResult CollAllGatherExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    const u64 count = param.GetDataCount(topoAttr_.userRank);
    const HcclDataType dataType = param.GetDataType();
    bool needLaunchAtTheEnd = !is310P3Common_; // 是否需要在Orchestrate()结束时launch任务

    HcclResult ret = HCCL_SUCCESS;
    // 图模式和单卡场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 countSize = count * SIZE_TABLE[dataType];
        u64 totalSize = CalcTotalCount(param) * SIZE_TABLE[dataType];
        ExecMem execMem;
        execMem.count = count;
        execMem.inputMem = DeviceMem::create(algRes.paramInputMem.ptr(), countSize);
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        HCCL_DEBUG("[CollAllGatherExecutor][Orchestrate]offload inputMem[%p][%llu], outputMem[%p][%llu]," \
            "scratchMem[%p][%llu], inputPtr[%p] outputPtr[%p], count[%llu].",
            execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
            execMem.scratchMem.ptr(), execMem.scratchMem.size(), execMem.inputPtr, execMem.outputPtr, execMem.count);
        ret = KernelRun(param, execMem);
    } else if (topoAttr_.userRankSize == 1) {
        ExecMem execMem;
        execMem.count = count;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        ret = KernelRun(param, execMem);
        needLaunchAtTheEnd = false;
    } else if (desc_.isZeroCopy) {
        u64 totalSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        // 在Level1和Level2执行RunLoop
        if (topoAttr_.serverNum > 1) {
            ret = RunLoop(param, algRes);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherExecutor][Orchestrate]errNo[0x%016llx]AllGather executor run loop failed",
                    HCCL_ERROR_CODE(ret)), ret);
        } else {        // 单机场景，数据直接从UserInput搬到UserOutput
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(algRes.paramOutputMem.ptr()) + totalSize * topoAttr_.userRank, totalSize);
            DeviceMem srcMem = DeviceMem::create(algRes.paramInputMem.ptr(), totalSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
        }
        // 在Level0执行KernelRun
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputMem = DeviceMem::create(algRes.paramInputMem.ptr(), totalSize);
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize * topoAttr_.userRankSize);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        ret = KernelRunIntraServerPost(param, execMem);
    } else {
        if (isAllGatherV_) {
            ret = RunLoopV(param, algRes);
        } else {
            ret = RunLoop(param, algRes);
        }
        needLaunchAtTheEnd = false;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherExecutor][Orchestrate]errNo[0x%016llx]AllGather executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    if (needLaunchAtTheEnd) {
        HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
    }

    HCCL_INFO("tag[%s], Allgather executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
```

### CollAllGatherExecutor::CalcLoopMaxCount

[coll_all_gather_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L104)

```cpp
u64 CollAllGatherExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_WARNING("[CollAllGatherExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize", maxCountPerLoop);
    return maxCountPerLoop;
}
```

### CollAllGatherExecutor::RunLoop

[coll_all_gather_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L219)

```cpp
HcclResult CollAllGatherExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    void *commInputPtr = algRes.cclInputMem.ptr();
    u8 *commOutputPtr = static_cast<u8 *>(algRes.cclOutputMem.ptr());
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[CollAllGatherExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
            param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);

    bool smallData = IsSmallData(param.DataDes.count * unitSize);
    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d]",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        if (!is310P3Common_) {
            /* 设置子图复用标志 */
            auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
            bool hugeData = IsHugeData(curSize);    // override
            bool dataSplit = false;
            auto opMeta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData, smallData,
                CopyPattern::BCOPY, dataSplit);
            CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        }

        // 执行
        if (!DMAReduceFlag_) {
            // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
            DeviceMem srcMem = DeviceMem::create(curInputPtr, curSize);
            DeviceMem dstMem = DeviceMem::create(commInputPtr, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]copy from user in to ccl in.");
        }

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = DeviceMem::create(commInputPtr, curSize);
        u32 sliceNum = desc_.isZeroCopy ? topoAttr_.serverNum : topoAttr_.userRankSize;
        execMem.outputMem = DeviceMem::create(commOutputPtr, curSize * sliceNum);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        HcclResult ret = HCCL_SUCCESS;
        if (!desc_.isZeroCopy) {
            ret = KernelRun(param, execMem);
        } else {
            ret = KernelRunInterServer(param, execMem);
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d].",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), commInputPtr, commOutputPtr,
            curCount, param.DataDes.dataType),
            ret);

        if (!DMAReduceFlag_) {
            // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
                DeviceMem dstMem = DeviceMem::create(curOutputPtr + param.DataDes.count * unitSize * i, curSize);
                DeviceMem srcMem = DeviceMem::create(commOutputPtr + curSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            }
        }

        if (!is310P3Common_) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        }

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}
```

### CollAllGatherExecutor::RunLoopV

[coll_all_gather_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L312)

```cpp
HcclResult CollAllGatherExecutor::RunLoopV(OpParam &param, AlgResourceResponse &algRes)
{
    auto counts = GetCounts(param);
    auto displs = GetDispls(param);
    const HcclDataType dataType = param.GetDataType();
    u32 unitSize = SIZE_TABLE[dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    u8 *commInputPtr = static_cast<u8 *>(algRes.cclInputMem.ptr());
    u8 *commOutputPtr = static_cast<u8 *>(algRes.cclOutputMem.ptr());

    if (UNLIKELY(curInputPtr == nullptr)) {
        // 若本rank的input count为0，此时允许curInputPtr传入空指针，为保证后续流程正常执行，赋值为cclin的地址
        curInputPtr = commInputPtr;
        HCCL_DEBUG("Since the input count is 0, set curInputPtr to ccl input[%p]", curInputPtr);
    } else {
        CHK_PTR_NULL(curInputPtr);
    }
    CHK_PTR_NULL(curOutputPtr);
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[CollAllGatherExecutor][RunLoopV]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
            param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);

    bool finished = false;
    while (!finished) {
        auto curCounts = std::vector<u64>();
        auto curDispls = std::vector<u64>();
        // 每轮loop需要重新计算counts和displs
        finished = CalcCountsDispls(maxCountPerLoop, counts, displs, curCounts, curDispls);
        u64 curCount = GetCurrentCount(param, curCounts);
        u64 curSize = curCount * unitSize; // 单位：字节
        const u64 totalSize = CalcCurrentTotalCount(param, curCounts) * unitSize;

        HCCL_DEBUG("[CollAllGatherExecutor][RunLoopV]tag[%s], sendBuf[%p], recvBuf[%p], sendSize[%llu], "
            "recvSize[%llu], cclInputMem[%u], cclOutputMem[%u], dataType[%d]", param.tag.c_str(), curInputPtr,
            curOutputPtr, curSize, totalSize, algRes.cclInputMem.size(), algRes.cclOutputMem.size(), dataType);

        /* 设置子图复用标志 */
        auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
        bool hugeData = IsHugeData(curSize);    // override
        bool smallData = IsSmallData(curSize);
        bool dataSplit = false;
        auto opMeta = GetOpMetaInfo(autoSelectedAlgTypeLevel1, hugeData, smallData, dataSplit);
        CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = DeviceMem::create(commInputPtr, curSize);
        execMem.outputMem = DeviceMem::create(commOutputPtr, totalSize);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        UpdateOpParam(param, curCounts, curDispls);
        HcclResult ret = KernelRun(param, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherExecutor][RunLoopV]errNo[0x%016llx]kernel run error, tag[%s], "
                "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d]", HCCL_ERROR_CODE(ret),
                param.tag.c_str(), commInputPtr, commOutputPtr, curCount, dataType), ret);

        CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

        curInputPtr += curSize;
        // AllGatherV curOutputPtr不需要偏移，偏移由displs计算
    }
    return HCCL_SUCCESS;
}
```

### CollAllGatherExecutor::AllGatherLevel2

[coll_all_gather_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L438)

```cpp
HcclResult CollAllGatherExecutor::AllGatherLevel2(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, Stream &stream, HcomCollOpInfo *opInfo)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    u64 inputMemSize = inputMem.size();
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;
    u32 level0ServerIndex = level0CommInfo.localRank;
    u32 level1ServerIndex = level1CommInfo.localRank;

    std::unique_ptr<AlgTemplateBase> level2AGExecutor;
    level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    HCCL_INFO("AllGather ring: using ring algo inter-server.");
    CHK_SMART_PTR_NULL(level2AGExecutor);

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level2DataSegsSlice;
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    //  outputMem传整块，通过baseOffset偏移
    u64 level2BaseOffset = (level0ServerIndex + level1ServerIndex * level1RankSize) * inputMemSize;
    CHK_RET(level2AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, level2BaseOffset));

    CHK_RET(level2AGExecutor->RegisterProfiler((
        level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
    HCCL_INFO("AllGather double ring [superpod] level2 AllGather run success");

    // 第二步，各个AI Server 间 AllGather (ring/NHR)
    HCCL_INFO("commIdx:%u Tag[%s].commLevel1.size():%u", commIndex, tag.c_str(),
        level1RankSize);

    if (level1RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level1AGExecutor;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather ring: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-bruck algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather ring: using nonuniform-hierarchical-ring algo inter-server.");
        } else {
            HCCL_ERROR("AllGather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);

        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice;
        for (u32 j = 0; j < level2RankSize; j++) {
            for (u32 i = 0; i < level1RankSize; i++) {
                sliceTemp.size = inputMemSize;
                sliceTemp.offset =
                    (i * level0RankSize +  j * level1RankSize * level0RankSize + level0ServerIndex) *inputMemSize;
                level1DataSegsSlice.push_back(sliceTemp);
            }
        }

        CHK_RET(level1AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler((
            level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
        HCCL_INFO("AllGather double ring [superpod] level1 AllGather run success");
    }

    // 节点内做AllGather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> level2UserMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / inputMemSize) * count * perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                level2UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level2UserMemSlice);
        }
    }

    CHK_RET(ActiveSlaveStreams(stream));
    if (DMAReduceFlag_ && level1RankSize > 1) {
        // AllGather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo->inputAddr = nullptr;
    }
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count,
        dataType, multRingsSlice, stream, PROF_STAGE_2, 0, opInfo, multRingsUserMemSlice));

    HCCL_INFO("AllGather double ring [superpod] level2 AllGather run success");
    return HCCL_SUCCESS;
}
```
