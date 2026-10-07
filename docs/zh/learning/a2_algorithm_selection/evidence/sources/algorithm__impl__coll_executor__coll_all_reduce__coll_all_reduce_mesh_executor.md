# coll_all_reduce_mesh_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllReduceMeshExecutor::KernelRun

[coll_all_reduce_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_executor.cc#L89)

```cpp
HcclResult CollAllReduceMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    std::unique_ptr<AlgTemplateBase> level0TempAlg;

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    // mesh算法stream数量为server内rank数减1

    CHK_RET(ActiveSlaveStreams(param.stream));

    if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE &&
        (param.DataDes.dataType != HCCL_DATA_TYPE_INT64) &&
        (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
        CHK_RET(MultiStreamReduceScatterMeshAtomic(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, param.reduceType, dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
    } else {
        CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
        CHK_RET(MultiStreamReduceScatterMesh(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
            param.DataDes.dataType, param.reduceType, multiStreamSlice,
            const_cast<Stream&>(param.stream), COMM_LEVEL0));
    }

    HCCL_INFO("AllReduce meshhd stage0 run success.");

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的allReduce计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    u32 commIndex = level0CommInfo.localRank;
    CHK_PRT_RET(commIndex >= dataSegsSlice.size(),
        HCCL_ERROR("[CollAllReduceMeshExecutor][Run]commIndex[%u] >= dataSegsSlice size[%zu]", commIndex,
        dataSegsSlice.size()), HCCL_E_INTERNAL);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_INFO("AllReduce mesh: using ring algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        u64 curSize = execMem.count * perDataSize; // 单位 byte
        HCCL_DEBUG("AllReduce mesh: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]",
            curSize, topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
        }
        HCCL_INFO("AllReduce mesh: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        level1TempAlg->CloseBarrier();
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
        HCCL_INFO("AllReduce mesh: using nhr_v1 algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_INFO("AllReduce mesh: using nb algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllReduce mesh: using Recursive halving-doubling algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    u32 rankSize = level1CommInfo.localRankSize;
    // 节点间的hd 使用环0来记录

    u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(level1TempAlg->Prepare(allreduceInput, allreduceOutput, allreduceOutput, hdCount,
        param.DataDes.dataType, param.stream, param.reduceType,
        LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    CHK_RET(level1TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

    HCCL_INFO("AllReduce meshhd stage1 run success.");

    /* 外层topo:all_gather */

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, 
            dispatcher_);
    } else {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH, 
            dispatcher_);
    }

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, 
        topoAttr_.userRank, nullptr, level0CommInfo.localRank, level0CommInfo.localRankSize));

    /* 节点内执行器 stage2 */
    {
        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType,
            LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

        CHK_RET(level0TempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    }

    HCCL_INFO("AllReduce meshhd stage2 run success");
    return HCCL_SUCCESS;
}
```

### CollAllReduceMeshExecutor::SelectTempAlg

[coll_all_reduce_mesh_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_executor.cc#L244)

```cpp
HcclResult CollAllReduceMeshExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("AllReduce mesh: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
            HCCL_INFO("AllReduce mesh: using ahc algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
            HCCL_INFO("AllReduce mesh: using ahc-broke algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_INFO("AllReduce mesh: using nb algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllReduce mesh: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
