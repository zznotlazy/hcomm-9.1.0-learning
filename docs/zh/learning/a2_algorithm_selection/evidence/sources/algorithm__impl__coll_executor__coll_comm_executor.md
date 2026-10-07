# coll_comm_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollCommExecutor::MultiRingAllReduce

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L49)

```cpp
HcclResult CollCommExecutor::MultiRingAllReduce(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;
        std::unique_ptr<AlgTemplateBase> tempAlg;
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr));

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }

            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed",
                ringIndex), ret);
            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero,
                baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex],
                profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] record failed", ringIndex), ret);
        } else {  // 主环
            tempAlg =
                AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(reduceAttr));

            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u], AllReduce(ring) run failed, "\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 */
            ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllReduce]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiRingAllGather

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L179)

```cpp
HcclResult CollCommExecutor::MultiRingAllGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const CommPlane leveIndex)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(leveIndex, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(leveIndex, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    TopoType topoType = topoType_;
 
    if (leveIndex == COMM_LEVEL0_LOGICAL) {
        std::vector<u32> mockNicList;
        mockNicList.reserve(level0ZeroCommInfo.localRankSize);
        for (u32 rankIndex = 0; rankIndex < level0ZeroCommInfo.localRankSize; rankIndex++) {
            mockNicList.push_back(rankIndex);
        }
        nicList = mockNicList;
        u32 ARSRankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0_LOGICAL)[0].size();
        bool ARSDoubleRing = ((ARSRankSize > FACTOR_TWO) && (ARSRankSize % FACTOR_TWO == 0) && topoAttr_.isARSDoubleRing);
 
        if (ARSDoubleRing) {
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
        } else {
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
        }
    }
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex];
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo level0RingCommInfo = GetSubCommInfo(leveIndex, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        level0RingCommInfo, algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING_DIRECT, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        level0RingCommInfo, algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                }

                ret = tempAlg->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                    mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
            }

            ret = tempAlg->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u],AllGather(ring) prepare failed,"\
                "return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], AllGather(ring) register Profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u], AllGather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGather]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiRingAllGatherConcurrent

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L365)

```cpp
HcclResult CollCommExecutor::MultiRingAllGatherConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
    Stream stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size(); // 环数, 当前为4环

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    auto halfRingSize = ringNum;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / 2; // 2环
    }
    // 拿到ring环映射关系
    CHK_RET(CheckCommSize(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_1));
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderForAnyPath(level0ZeroCommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second; // 取出sdma/rdma的数据块
        CHK_PRT_RET(singleRingSliceZero.empty(), HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]"\
            "singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 910_93场景 生成userMemOut_上对应的slices
        std::vector<Slice> userMemOutputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemOutputSlices));
        } else {
            userMemOutputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        u32 commIndex = ringIndex % halfRingSize;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo level0RingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, commIndex) : GetSubCommInfo(COMM_LEVEL0_ANYPATH_RDMA, commIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        // 910_93场景 准备环中的从流
        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) { // 最后一个环是主stream，所以这里减1，符合条件的走从stream
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ?
                        ExecutorType::ALLGATHER_RING_DIRECT : ExecutorType::ALLGATHER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        level0RingCommInfo, algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex],
                        ringIndex, type, 0, opInfo, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(outputMem, outputMem, inputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                        singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize], tag, profStage,
                        level0RingCommInfo, algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex],
                        ringIndex, ExecutorType::ALLGATHER_RING);
                }
                algResResp_->threadManage[ringIndex]->NotifyStart();    // 给线程发信号启动处理
            } else {
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed",
                    ringIndex), ret);
                // 如何判断是否环内是否有数据, 以ring的第一个rank的 size为判断依据
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, 
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemOutputSlices, isSdma));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                }
                ret = tempAlg->Prepare(outputMem, outputMem, inputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) prepare "\
                    "failed,return[%d]", ringIndex, ret), ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) register "\
                    "Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed",
                    ringIndex), ret);
            }

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] record failed", ringIndex), ret);
        } else { // 主环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemOutputSlices, isSdma));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
            }
            ret = tempAlg->Prepare(outputMem, outputMem, inputMem, count, dataType, stream, HCCL_REDUCE_RESERVED,
                LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex%halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) prepare"\
                " failed,return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) register "\
                "Profiler failed, return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u],AllGather(ring) run failed,"\
                "return[%d]", ringIndex, ret), ret);

            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone(); // 单算子模式，等待线程处理完成信号
                }
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingAllGatherConcurrent]stream[%u] wait failed", ring), ret);
            }
        }
    }
    // 添加空task,保证执行时不乱序
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::Level1AllGatherConcurrent

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L543)

```cpp
HcclResult CollCommExecutor::Level1AllGatherConcurrent(DeviceMem inputMem, DeviceMem outputMem,const u64 count,
    const HcclDataType dataType, Stream stream, s32 profStage,std::vector<Slice> &level1DataSegsSlice, u32 syncTrans)
{
    std::vector<std::pair<bool, std::vector<Slice>>> level1MultSlice;
    std::vector<Slice> level1DataSegsSliceSdma;
    std::vector<Slice> level1DataSegsSliceRdma;
    bool isAnyPathCommLevel0 = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ? true : false;
    CommPlane commplane = (isAnyPathCommLevel0) ? COMM_LEVEL0_ANYPATH_SDMA : COMM_LEVEL0;
    CHK_RET(CheckCommSize(commplane, COMM_INDEX_1));
    SubCommInfo level0CommInfo = GetSubCommInfo(commplane, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, level0ServerIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    HcclResult ret = HCCL_SUCCESS;
    level1MultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);

    for (u32 i = 0; i < level1CommInfo.localRankSize; i++) {
        Slice sdmaSlice;
        Slice rdmaSlice;
        u64 sdmaSliceSize =
            ((level1DataSegsSlice[i].size <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE))
                ? level1DataSegsSlice[i].size
                : ((syncTrans * level1DataSegsSlice[i].size / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) *
                      HCCL_MIN_SLICE_ALIGN_910_93;
        sdmaSlice.size = sdmaSliceSize;
        sdmaSlice.offset = level1DataSegsSlice[i].offset;
        rdmaSlice.size = level1DataSegsSlice[i].size - sdmaSliceSize;
        rdmaSlice.offset = level1DataSegsSlice[i].offset + sdmaSliceSize;
        level1DataSegsSliceSdma.push_back(sdmaSlice);
        level1DataSegsSliceRdma.push_back(rdmaSlice);
        HCCL_DEBUG("Level1 index:[%u], Original [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                   "rdma [offset %llu, size %llu]", i, level1DataSegsSlice[i].offset, level1DataSegsSlice[i].size,
            sdmaSlice.offset, sdmaSlice.size, rdmaSlice.offset, rdmaSlice.size);
    }
    level1MultSlice[0] = std::make_pair(true, level1DataSegsSliceSdma);
    level1MultSlice[1] = std::make_pair(false, level1DataSegsSliceRdma);

    u32 commPlaneNum = level1MultSlice.size();
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice> &singleSlice = level1MultSlice[planeIndex].second;
        SubCommInfo level1RdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_RDMA, level0ServerIndex);
        SubCommInfo level1TempCommInfo = level1MultSlice[planeIndex].first ? level1CommInfo : level1RdmaCommInfo;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesAux[planeIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(level1TempAlg->Prepare(outputMem, outputMem, inputMem, count,
                dataType, algResResp_->slaveStreams[planeIndex], HCCL_REDUCE_RESERVED,
                INVALID_VALUE_RANKID, singleSlice, 0));

            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesMain[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[planeIndex], profStage);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(level1TempAlg->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
                HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, singleSlice, 0));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1AllGatherConcurrent run success");
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::CalcOptimalIntraRingsize

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L640)

```cpp
u32 CollCommExecutor::CalcOptimalIntraRingsize(u64 count, HcclDataType dataType, HcclCMDType opType)
{
    if (!topoMatcher_->GetARSFlag()) return 0;
 
    u32 level0RankSize   = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0)[0].size();
    u32 rankSizeInSuperPod = topoMatcher_->GetCommPlaneRanks(COMM_ARS)[0].size();
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    // 不支持 ARS 或环内卡数不是 2 的倍数
    u32 level0RingSize = 1;
    if (!topoAttr_.isARSDoubleRing || (level0RankSize % FACTOR_TWO != 0)) {
        HCCL_INFO("not Support ARS doubleRing, level0RingSize:[%u], level0RankSize[%u].", level0RingSize, level0RankSize);
        return level0RingSize;
    }
    // --- 1. 带宽 & 基本参数 ---
    float bwHCCS, bwHBM, bwSIO;
    constexpr u32 level0 = 0;
    constexpr u32 level2 = 2;
    constexpr u32 level3 = 3;
    CHK_RET(GetBandWidthPerNPU(level0, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwHCCS));
    CHK_RET(GetBandWidthPerNPU(level2, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwHBM));
    CHK_RET(GetBandWidthPerNPU(level3, topoAttr_.userRankSize, topoAttr_.deviceNumPerAggregation, bwSIO));
    float latency = BASE_COMM_LATENCY / MULTIPLIER_MS2US;   // ms
    // --- 2. 数据总量 (GB) ---
    float baseSizeGB = static_cast<double>(count) * perDataSize / (1024 * 1024 * 1024);
    float totalSize  = baseSizeGB;
    HCCL_INFO("CalcOptimalIntraRingsize: count[%u], totalSize:[%lf]GB, perDataSize[%u].", count, totalSize, perDataSize);
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        totalSize *= rankSizeInSuperPod;
    }
    // --- 3. 枚举可能的环大小 ---
    std::vector<u32> factors;
    for (u32 i = 1; i <= rankSizeInSuperPod / i; ++i) {
        if (rankSizeInSuperPod % i == 0) {
            factors.push_back(i);
            if (i != rankSizeInSuperPod / i) {
                factors.push_back(rankSizeInSuperPod / i);
            }
        }
    }
    std::sort(factors.begin(), factors.end());
    // --- 4. 计算最优带宽 ---
    double maxBwARS = 0.0;
    for (u32 N1 : factors) {
        u32 N2 = rankSizeInSuperPod / N1;
        // 静态时延 (ms)
        double interStep = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) ? (N2 - 1) : log2(N2);
        double latencyStep = (interStep + (N1 - 1)) * latency;
        // 传输时延 (ms)
        double latencyIntra;
        if ((N1 % FACTOR_TWO == 0) && (N1 > FACTOR_TWO)) {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS / FACTOR_TWO;
        } else if (N1 == FACTOR_TWO) {
            latencyIntra = totalSize * MULTIPLIER_S2MS / FACTOR_TWO / bwSIO;
        } else {
            latencyIntra = (N1 - 1) * totalSize * MULTIPLIER_S2MS / N1 / bwHCCS;
        }
        double latencyInter = (N2 - 1) * totalSize * MULTIPLIER_S2MS / N1 / N2 / bwHCCS;
        // HBM 拷贝时延 (ms)
        double latencyCopy = totalSize * MULTIPLIER_S2MS / bwHBM;
        u8 mul = (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) ? FACTOR_TWO : 1;
        double timeCost = mul * (latencyStep + latencyIntra + latencyInter) + latencyCopy;
        double bwARS = totalSize / timeCost;  // GB/ms
        if (bwARS > maxBwARS) {
            maxBwARS = bwARS;
            level0RingSize = N1;
        }
    }
    HCCL_INFO("level0RingSize:[%u], totalSize:[%lf]GB, level0RankSize[%u].", level0RingSize, totalSize, level0RankSize);
    return level0RingSize;
}
```

### CollCommExecutor::MultiRingReduceScatter

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L753)

```cpp
HcclResult CollCommExecutor::MultiRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const CommPlane levelIndex)    
{
    HCCL_INFO("[MultiRingReduceScatter] MultiRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(levelIndex, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(levelIndex, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;

    TopoType topoType = topoType_;
 
    if (levelIndex == COMM_LEVEL0_LOGICAL) {
        std::vector<u32> mockNicList;
        mockNicList.reserve(level0ZeroCommInfo.localRankSize);
        for (u32 rankIndex = 0; rankIndex < level0ZeroCommInfo.localRankSize; rankIndex++) {
            mockNicList.push_back(rankIndex);
        }
        nicList = mockNicList;
        u32 ARSRankSize = topoMatcher_->GetCommPlaneRanks(COMM_LEVEL0_LOGICAL)[0].size();        
        bool ARSDoubleRing = ((ARSRankSize > FACTOR_TWO) && (ARSRankSize % FACTOR_TWO == 0) && topoAttr_.isARSDoubleRing);
        if (ARSDoubleRing) {
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
        } else {
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
        }
    }
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex];
        }

        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));

        SubCommInfo level0RingCommInfo = GetSubCommInfo(levelIndex, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]active stream[%u], failed",
                    ringIndex), ret);
            }
            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                        LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex], tag, profStage,
                        level0RingCommInfo, algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex],
                        ringIndex, ExecutorType::REDUCE_SCATTER_RING_DIRECT, reduceAttr, opInfo,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr));
                }
                CHK_SMART_PTR_NULL(tempAlg);

                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ringIndex), ret);
                ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) run "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] record failed", ringIndex), ret);
        } else { // 主环 最后一个环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                    mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr));
            }
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) prepare "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) register "\
                "Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u],ReduceScatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);

                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiRingGather

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L953)

```cpp
HcclResult CollCommExecutor::MultiRingGather(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    HcclReduceOp op, u32 root, Stream stream, s32 profStage)
{
    u32 ringNum = multRingsSliceZero.size();
    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    HcclResult ret;

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CommonOperator][MultiRingGather]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CommonOperator][MultiRingGather]invalid root rank[%u] to get user rank", root), ret);

        std::unique_ptr<AlgTemplateBase> tempAlg = nullptr;
        EXCEPTION_CATCH(
            (tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_GATHER_RING, dispatcher_)),
            return HCCL_E_PTR);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_GATHER_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]in stream[%u] wait failed", \
                ringIndex), ret);
            if (singleRingSliceZero[0].size != 0) {
            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType,
                                    algResResp_->slaveStreams[ringIndex], op, rootRank, singleRingSliceZero, 0,
                                    ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(level0RingCommInfo.localRank, profStage, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register profiler "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed,return[%d]",
                ringIndex, ret), ret);
            }
            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex],
                profStage);

            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] record failed", \
                ringIndex), ret);
        } else {  // 主环
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_GATHER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);

            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                op, rootRank, singleRingSliceZero, 0, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) prepare failed, "\
                "return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u], gather(ring) register "\
                "profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u],gather(ring) run failed, "\
                "return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CommonOperator][MultiRingGather]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiRingReduceScatterConcurrent

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1054)

```cpp
HcclResult CollCommExecutor::MultiRingReduceScatterConcurrent(const std::string &tag, DeviceMem inputMem,
    DeviceMem outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));
    u32 halfRingSize = ringNum;
    u32 DoubleRing = 2;
    if (ringNum > RDMA_PLANE_NUM_IN_NPRING_DOUBLE) {
        halfRingSize = ringNum / DoubleRing;
    }

    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderForAnyPath(level0ZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex].second;
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]singleRingSliceZero is empty"),
            HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        u32 commIndex = ringIndex % halfRingSize;
        if (multRingsUserMemSlice.size() == 0) {
            CHK_RET(CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder,
                userMemInputSlices));
        } else {
            userMemInputSlices = multRingsUserMemSlice[ringIndex].second;
        }
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, commIndex, rankOrder));

        SubCommInfo level0RingCommInfo = multRingsSliceZero[ringIndex].first ?
            GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, commIndex) : GetSubCommInfo(COMM_LEVEL0_ANYPATH_RDMA, commIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;
        u32 ringIndexOp = ringIndex;

        std::vector<Stream>       subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        if (opInfo != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
        }
        bool isSdma = multRingsSliceZero[ringIndex].first;
        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]active stream[%u], failed",
                        ringIndex), ret);
            }

            if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                /* 更新线程参数 */
                if (opInfo != nullptr) {
                    ExecutorType type = isSdma ?
                       ExecutorType::REDUCE_SCATTER_RING_DIRECT : ExecutorType::REDUCE_SCATTER_RING_DIRECT_RDMA;
                    algResResp_->threadManage[ringIndex]->Prepare(
                        inputMem, inputMem, outputMem, count, dataType, algResResp_->slaveStreams[ringIndex], reductionOp,
                        LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize], tag,
                        profStage, level0RingCommInfo, algResResp_->notifiesAux[ringIndex],
                        algResResp_->notifiesMain[ringIndex], ringIndex, type,
                        reduceAttr, opInfo, subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder,
                        userMemInputSlices);
                } else {
                    algResResp_->threadManage[ringIndex]->Prepare(inputMem, inputMem, outputMem, count, dataType,
                        algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero,
                        baseOffset, ringNics[ringIndex % halfRingSize], tag, profStage, level0RingCommInfo,
                        algResResp_->notifiesAux[ringIndex], algResResp_->notifiesMain[ringIndex], ringIndex,
                        ExecutorType::REDUCE_SCATTER_RING, reduceAttr);
                }

                algResResp_->threadManage[ringIndex]->NotifyStart(); // 给线程发通知启动线程执行
            } else {
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                        mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
                    HCCL_DEBUG("[MultiRingReduceScatterConcurrent]run in COMM_LEVEL0 ends");
                } else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(reduceAttr));
                }
                HCCL_DEBUG("[MultiRingReduceScatterConcurrent]run in COMM_LEVEL0 ends");
                CHK_SMART_PTR_NULL(tempAlg);

                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesAux[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed", ringIndex),
                    ret);
                ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID,
                    singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "\
                    "prepare failed,return[%d]", ringIndex, ret), ret);
                ret = tempAlg->RegisterProfiler(
                    ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    profStage, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "\
                    "register Profiler failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring)"\
                    " run failed,return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesMain[ringIndex], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed",
                    ringIndex),
                    ret);
            }
            /* 主环record启动从环 */
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], profStage);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] record failed", ringIndex),
                ret);
        } else { // 主环 最后一个环
            std::unique_ptr<AlgTemplateBase> tempAlg;
            if (opInfo != nullptr) {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING_DIRECT, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr, opInfo, topoAttr_.userRank, subStreamsInOneRing,
                    mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices, isSdma));
            } else {
                tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL0", __func__);
                CHK_SMART_PTR_NULL(tempAlg);
                CHK_RET(tempAlg->Prepare(reduceAttr));
            }
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex % halfRingSize]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "\
                " prepare failed,return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                profStage, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) "\
                "register Profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u],ReduceScatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                if (!static_cast<bool>(topoMatcher_->GetExternalInputHcclEnableFfts()) &&
                    workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algResResp_->threadManage[ring]->WaitDone();
                }
                /* 等待executor执行完毕 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], profStage);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingReduceScatterConcurrent]stream[%u] wait failed",
                    ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::Level1ReduceScatterConcurrent

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1252)

```cpp
HcclResult CollCommExecutor::Level1ReduceScatterConcurrent(DeviceMem inputMem, DeviceMem scratchMem,const u64 count,
    const HcclDataType dataType, const HcclReduceOp reductionOp, Stream stream, s32 profStage,
    std::vector<Slice> &level1DataSegsSlice, u32 syncTrans, u64 reduceAttr)
{
    (void)profStage;
    std::vector<std::pair<bool, std::vector<Slice>>> level1MultSlice;
    level1MultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
    std::vector<Slice> sdmaSlice;
    std::vector<Slice> rdmaSlice;
    for (u32 segsIndex = 0; segsIndex < level1DataSegsSlice.size(); segsIndex++) {
        u64 totalSize = level1DataSegsSlice[segsIndex].size;
        u64 sdmaSliceOffset = level1DataSegsSlice[segsIndex].offset;
        u64 sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ? totalSize
                                 : ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) *
                                       HCCL_MIN_SLICE_ALIGN_910_93;
        Slice sdmaSliceTmp;
        sdmaSliceTmp.offset = sdmaSliceOffset;
        sdmaSliceTmp.size = sdmaSliceSize;
        Slice rdmaSliceTmp;
        rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
        rdmaSliceTmp.size = totalSize - sdmaSliceSize;
        sdmaSlice.push_back(sdmaSliceTmp);
        rdmaSlice.push_back(rdmaSliceTmp);
        HCCL_DEBUG("Level1 data segId:%u, Original [offset %llu, size %llu], sdma [offset %llu, size %llu], "
                   "rdma [offset %llu, size %llu]", segsIndex, sdmaSliceOffset, totalSize, sdmaSliceTmp.offset,
                   sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
    }
    level1MultSlice[0] = std::make_pair(true, sdmaSlice);   // true表示使用sdma
    level1MultSlice[1] = std::make_pair(false, rdmaSlice);  // false表示rdma

    u32 commPlaneNum = level1MultSlice.size();
    bool isAnyPathCommLevel0 = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ? true : false;
    CommPlane commplane = (isAnyPathCommLevel0) ? COMM_LEVEL0_ANYPATH_SDMA : COMM_LEVEL0;
    u32 commIndex = GetSubCommInfo(commplane, COMM_INDEX_0).localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_SDMA, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_RDMA, commIndex + 1));
    SubCommInfo level1RdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_RDMA, commIndex);
    for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
        std::vector<Slice> &singleSlice = level1MultSlice[planeIndex].second;
        SubCommInfo level1TempCommInfo = level1MultSlice[planeIndex].first ? level1CommInfo : level1RdmaCommInfo;
        std::unique_ptr<AlgTemplateBase> level1TempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HcclResult ret = HCCL_SUCCESS;

        if (planeIndex != (commPlaneNum - 1)) {
            ret = LocalNotify::Wait(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesAux[planeIndex], reductionOp);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

            CHK_RET(level1TempAlg->Prepare(inputMem, inputMem, scratchMem, count, dataType,
                algResResp_->slaveStreams[planeIndex], reductionOp, LEVEL0_BRIDGE_RANK_ID, singleSlice));

            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            ret = LocalNotify::Post(
                algResResp_->slaveStreams[planeIndex], dispatcher_, algResResp_->notifiesMain[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
            // 主环record启动从环
            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[planeIndex], reductionOp);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[collAllGather]level1 stream[%u] record failed", planeIndex), ret);
        } else {
            CHK_RET(level1TempAlg->Prepare(inputMem, inputMem, scratchMem, count, dataType, stream,
                reductionOp, LEVEL0_BRIDGE_RANK_ID, singleSlice));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1TempCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                reductionOp, HCCL_EXEC_STEP_NOT_SET, stream));

            CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
            for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], reductionOp);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed", ring), ret);
            }
        }
    }
    HCCL_INFO("Level1ReduceScatterConcurrent run success");
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, scratchMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiRingMultiRootScatter

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1348)

```cpp
HcclResult CollCommExecutor::MultiRingMultiRootScatter(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, u32 root, Stream stream, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);

        u32 rankSize = level0RingCommInfo.localRankSize;
        std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_MULTI_ROOT_SCATTER_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_MULTI_ROOT_SCATTER_RING in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(tempAlg);

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr()));
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]invalid root [%u] to get userrank", root), ret);

        if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
            ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]in stream[%u] wait failed", ringIndex), ret);

            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[ringIndex], HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID,
                singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "prepare failed,return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
                level0RingCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "\
                "register profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "failed,return[%d]", ringIndex, ret), ret);

            ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_, algResResp_->notifiesMain[ringIndex],
                PROF_STAGE_0);

            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] record failed", ringIndex), ret);
        } else {  // 主环
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_MULTI_ROOT_SCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_MULTI_ROOT_SCATTER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            ret = tempAlg->Prepare(inputMem, outputMem, outputMem, count, dataType, stream,
                HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) "\
                "prepare failed,return[%d]", ringIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + level0RingCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u], multirootscatter(ring) "\
                "register profiler failed,return[%d]", ringIndex, ret), ret);

            ret = RunTemplate(tempAlg, level0RingCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u],multirootscatter(ring) run "\
                "failed,return[%d]", ringIndex, ret), ret);
            for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingMultiRootScatter]stream[%u] wait failed", ring), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiStreamReduceScatterMeshAtomic

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1454)

```cpp
HcclResult CollCommExecutor::MultiStreamReduceScatterMeshAtomic(const std::string &tag, DeviceMem &inputMem,
    DeviceMem &outputMem, const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<Slice> &dataSliceVct, Stream &stream,
    const CommPlane commLevelIndex, const u64 baseOffset, HcomCollOpInfo *opInfo)
{
    (void) tag;
    u32 unitSize = SIZE_TABLE[dataType];

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);
    std::unique_ptr<AlgTemplateBase> tempAlg;
    DeviceMem deviceOutputMem = inputMem;
    if (topoAttr_.isSingleMeshAggregation && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        static_cast<bool>((reduceAttr & INLINE_REDUCE_BITMASK)) && (opInfo != nullptr)) {
        if (((opInfo -> count) * unitSize <= HCCL_SMALL_COUNT_32_KB) &&
            (topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT)) {
            deviceOutputMem = outputMem;
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_HDSTAGE in COMM_LEVEL0", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_DIRECT in COMM_LEVEL0", __func__);
        }
    } else {
        tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_ATOMIC in COMM_LEVEL0", __func__);
    }
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(CheckCommSize(commLevelIndex, COMM_INDEX_0 + 1));
    const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);
    CHK_RET(tempAlg->Prepare(inputMem, deviceOutputMem, outputMem, count, dataType, stream, reductionOp,
        LEVEL0_BRIDGE_RANK_ID, dataSliceVct, baseOffset, reduceAttr, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, opInfo));

    CHK_RET(tempAlg->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(tempAlg, subCommInfo));

    return HCCL_SUCCESS;
}
```

### CollCommExecutor::MultiStreamReduceScatterMesh

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1500)

```cpp
HcclResult CollCommExecutor::MultiStreamReduceScatterMesh(const std::string &tag,
    DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
    const CommPlane commLevelIndex, const u64 baseOffset)
{
    (void) tag;
    HcclResult ret = HCCL_SUCCESS;
    u64 streamNum = multStreamsSlice.size();
    HCCL_INFO("MultiStreamReduceScatterMesh streamNum[%llu]", streamNum);
    CHK_RET(CheckCommSize(commLevelIndex, streamNum));
    const SubCommInfo zeroCommInfo = GetSubCommInfo(commLevelIndex, COMM_INDEX_0);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    for (u32 streamIndex = 0; streamIndex < streamNum; streamIndex++) {
        std::vector<Slice> singleStreamSlice = multStreamsSlice[streamIndex];
        CHK_PRT_RET(singleStreamSlice.size() <= 0,
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]singleStreamSlice is empty"),
            HCCL_E_INTERNAL);

        const SubCommInfo subCommInfo = GetSubCommInfo(commLevelIndex, streamIndex);
        u32 commIndex = subCommInfo.localRank;
        CHK_PRT_RET(commIndex >= singleStreamSlice.size(), \
            HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]commIndex[%u] => " \
            "singleStreamSlice size[%zu]", commIndex, singleStreamSlice.size()), HCCL_E_INTERNAL);

        u32 rankSize = subCommInfo.localRankSize;
        u32 ringIndexOp = streamIndex;
        std::unique_ptr<AlgTemplateBase> tempAlg;

        if (topoAttr_.isDiffDeviceType) {
            HCCL_DEBUG("[CollCommExecutor][MultiStreamReduceScatterMesh]isDiffDeviceType");
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX_SS, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH_MIX_SS in COMM_LEVEL0", __func__);
        } else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_MESH in COMM_LEVEL0", __func__);
        }
        CHK_SMART_PTR_NULL(tempAlg);
        CHK_RET(tempAlg->Prepare(reduceAttr, streamIndex));

        if (streamIndex != (streamNum - 1)) {  // 0~ringNum-2的环
            HCCL_INFO("MultiStreamReduceScatterMesh step into subStream");
            ret = LocalNotify::Wait(algResResp_->slaveStreams[streamIndex], dispatcher_,
                algResResp_->notifiesAux[streamIndex], PROF_STAGE_0);
            // 等待executor执行完毕
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                streamIndex), ret);

            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType,
                algResResp_->slaveStreams[streamIndex], reductionOp,
                LEVEL0_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) "\
                "prepare failed,return[%d]", streamIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                algResResp_->slaveStreams[streamIndex]);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) "\
                "register Profiler failed,return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(tempAlg, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u],ReduceScatter(mesh) run "\
                "failed,return[%d]", streamIndex, ret), ret);

            ret  = LocalNotify::Post(algResResp_->slaveStreams[streamIndex], dispatcher_,
                algResResp_->notifiesMain[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);

            ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[streamIndex], PROF_STAGE_0);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] record failed",
                streamIndex), ret);
        } else { // 主环
            HCCL_INFO("MultiStreamReduceScatterMesh step into mainStream");

            ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                reductionOp, LEVEL0_BRIDGE_RANK_ID, singleStreamSlice, baseOffset);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "ReduceScatter(mesh) prepare failed, return[%d]", streamIndex, ret), ret);

            ret = tempAlg->RegisterProfiler(
                ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + \
                zeroCommInfo.localRank, PROF_STAGE_0,
                HCCL_EXEC_STEP_NOT_SET, stream);
            CHK_PRT_RET(ret != HCCL_SUCCESS,\
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], ReduceScatter(mesh) " \
                "register Profiler failed, return[%d]", streamIndex, ret), ret);

            ret = RunTemplate(tempAlg, subCommInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u], " \
                    "ReduceScatter(mesh) run failed, return[%d]", streamIndex, ret), ret);

            for (u32 streamIndex = 0; streamIndex < (streamNum - 1); streamIndex++) {
                //  等待executor执行完毕
                ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[streamIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiStreamReduceScatterMesh]stream[%u] wait failed",
                        streamIndex), ret);
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return ret;
}
```

### CollCommExecutor::MultiRingScatter

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2010)

```cpp
HcclResult CollCommExecutor::MultiRingScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
    u32 root, Stream stream, const HcomCollOpInfo *opInfo, const u64 baseOffset)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder = GetRingsOrderByTopoType(level0CommInfo.localRankSize, topoType_, nicList);

    // 空拷贝用于后续操作附着
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(singleRingSliceZero.empty(),
            HCCL_ERROR("[CollCommExecutor][MultiRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, ringIndex);
        u32 rankSize = level0RingCommInfo.localRankSize;

        std::vector<Stream> subStreamsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> mainSignalsInOneRing;
        std::vector<std::shared_ptr<LocalNotify>> subSignalsInOneRing;
        std::unique_ptr<AlgTemplateBase> tempAlg;
        if (opInfo == nullptr) {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
        }
        else if (opInfo->inputAddr != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
        }
        else {
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(
                const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, rankOrder, userMemInputSlices));
        }

        if (ringIndex != (ringNum - 1)) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
                ret = StreamActiveManager::GetInstance(topoAttr_.deviceLogicId).StreamActive(
                    algResResp_->slaveStreams[ringIndex].ptr(), stream.ptr());
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],active stream failed", ringIndex), ret);
            }
        }

        u32 rootRank = 0;
        ret = GetRankByUserRank(COMM_LEVEL0, ringIndex, root, rootRank);
        CHK_PRT_RET(ret == HCCL_E_PARA,
            HCCL_ERROR("[CollCommExecutor][MultiRingScatter]invalid root [%u] to get userrank", root), ret);

        if (ret == HCCL_SUCCESS) {
            if (ringIndex != (ringNum - 1)) {  // 0~ringNum-2的环
                ret = LocalNotify::Wait(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]in stream[%u] wait failed", ringIndex), ret);

                ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType,
                    algResResp_->slaveStreams[ringIndex], HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero,
                    baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = tempAlg->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                ret = LocalNotify::Post(algResResp_->slaveStreams[ringIndex], dispatcher_,
                    algResResp_->notifiesMain[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
                /* 主环record启动从环 */
                ret = LocalNotify::Post(stream, dispatcher_, algResResp_->notifiesAux[ringIndex], PROF_STAGE_0);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] record failed", ringIndex), ret);
            } else {  // 主环
                std::unique_ptr<AlgTemplateBase> tempAlg;
                if (opInfo == nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                }
                else if (opInfo->inputAddr != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                }
                else {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(
                        const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, rankOrder, userMemInputSlices));
                }

                ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, stream,
                    HCCL_REDUCE_RESERVED, rootRank, singleRingSliceZero, baseOffset, ringNics[ringIndex]);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) prepare failed, "\
                    "return[%d]", ringIndex, ret), ret);
                ret = tempAlg->RegisterProfiler(((ringIndex + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
                    (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
                    PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u], scatter(ring) register profiler "\
                    "failed,return[%d]", ringIndex, ret), ret);

                ret = RunTemplate(tempAlg, level0RingCommInfo);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u],scatter(ring) run failed, "\
                    "return[%d]", ringIndex, ret), ret);

                for (u32 ring = 0; ring < (ringNum - 1); ring++) {
                    /* 等待executor执行完毕 , 当前环没有分配数据，跳过此环处理，继续下一个环 */
                    ret = LocalNotify::Wait(stream, dispatcher_, algResResp_->notifiesMain[ring], PROF_STAGE_0);
                    CHK_PRT_RET(ret != HCCL_SUCCESS,
                        HCCL_ERROR("[CollCommExecutor][MultiRingScatter]stream[%u] wait failed", ring), ret);
                }
            }
        }
    }

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
```

### if

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2052)

```cpp
else if (opInfo->inputAddr != nullptr) {
            CHK_RET(GetSubStreamInfoOnOneRing(ringIndex, subStreamsInOneRing, mainSignalsInOneRing,
                                              subSignalsInOneRing));
            tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
            CHK_SMART_PTR_NULL(tempAlg);
            CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank, subStreamsInOneRing,
                mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
        }
```

### if

[coll_comm_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2127)

```cpp
else if (opInfo->inputAddr != nullptr) {
                    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                        TemplateType::TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT, dispatcher_);
                    HCCL_CONFIG_INFO(HCCL_ALG, "[%s][KernelRun] Run TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT in COMM_LEVEL0", __func__);
                    CHK_SMART_PTR_NULL(tempAlg);
                    CHK_RET(tempAlg->Prepare(const_cast<HcomCollOpInfo *>(opInfo), topoAttr_.userRank,
                        subStreamsInOneRing, mainSignalsInOneRing, subSignalsInOneRing, rankOrder, userMemInputSlices));
                }
```
