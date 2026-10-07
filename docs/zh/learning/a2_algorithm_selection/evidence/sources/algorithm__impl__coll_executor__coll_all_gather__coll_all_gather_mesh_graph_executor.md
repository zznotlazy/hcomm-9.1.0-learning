# coll_all_gather_mesh_graph_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollAllGatherMeshGraphExecutor::KernelRun

[coll_all_gather_mesh_graph_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_graph_executor.cc#L60)

```cpp
HcclResult CollAllGatherMeshGraphExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
	HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherMeshGraphExecutor][KernelRun]AllGather start");
	
	// 获取子通信域信息
	u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
	u32 level1RankSize = level1CommInfo.localRankSize;
    u32 serverIndex = level1CommInfo.localRank;
    u64 inputMemSize = execMem.inputMem.size();
	CHK_RET(ActiveSlaveStreams(param.stream));

    //  第一步，将数据从input内存拷贝到output内存的对应位置
	u32 transposeRankId = commIndex * level1RankSize + serverIndex;
	u64 initOffset = transposeRankId * inputMemSize;
	DeviceMem dstMem = execMem.outputMem.range(initOffset, inputMemSize);
	CHK_SMART_PTR_NULL(dstMem);

    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun]AllGather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        initOffset, inputMemSize), ret);

    // 第二步，各个AI Server 间 AllGather
    u64 inputDataCount = inputMemSize / perDataSize;
	u64 level1Offset = commIndex * level1RankSize * inputMemSize;
	HCCL_INFO("[CollAllGatherMeshGraphExecutor][KernelRun] userRank[%u] commIndex[%u] level1Offset[%llu] level0RankSize[%u] " \
		"level1RankSize[%u] outputMemSize[%llu] count[%u]", topoAttr_.userRank, commIndex, level1Offset, level0RankSize, level1RankSize, 
		execMem.outputMem.size(), execMem.count);
	DeviceMem level1OutputMem = execMem.outputMem.range(level1Offset, inputMemSize * level1RankSize);
	CHK_SMART_PTR_NULL(level1OutputMem);

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
		level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
		level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
		level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
		level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
    } else {
		level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("AllGather mesh: algo is [%s] using halving-doubling algo inter-server.", (HCCL_ALGO_LEVEL1_MAP.at(algType_.algoLevel1)).c_str());
    }
    CHK_SMART_PTR_NULL(level1TempAlg);
    //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
    CHK_RET(level1TempAlg->Prepare(level1OutputMem, level1OutputMem, execMem.inputMem, inputDataCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), level1Offset));

    u32 rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
	HCCL_INFO("[CollAllGatherMeshGraphExecutor][KernelRun] AllGather level1 run success");

	//  第3步 机内AllGather
	std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = level0RankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize * level1RankSize, dataSegsSlice));
	// mesh算法stream数量为server内rank数减1
    CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));

	std::unique_ptr<AlgTemplateBase> level0TempAlg;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
		level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC,
                                                                       dispatcher_);
    } else {
        HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun] current device type [%u] not supported.", topoAttr_.deviceType);
		return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level0TempAlg);
	CHK_RET(level0TempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, nullptr, commIndex, level0RankSize));
    CHK_RET(level0TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem,
        execMem.count * level0RankSize * level1RankSize, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));
    rankSize = level0RankSize;
    CHK_RET(level0TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("[CollAllGatherMeshGraphExecutor][KernelRun] AllGather level0 run success");

	// 第4步 内存重排
	std::vector<bool> reorderedDone(topoAttr_.userRankSize, false);		
	for (u32 scratchSliceIndex = 0; scratchSliceIndex < topoAttr_.userRankSize; scratchSliceIndex++) {
		if (reorderedDone[scratchSliceIndex]) {
			continue;
		}

		// 当前位置应该放的数据实际所在位置
		u32 srcServerId = scratchSliceIndex / level0RankSize;
		u32 srcLocalRank = scratchSliceIndex % level0RankSize;
		u32 srcSliceRankId = srcLocalRank * level1RankSize + srcServerId;
		if (srcSliceRankId == scratchSliceIndex) {
			reorderedDone[scratchSliceIndex] = true;
			continue;
		}

		u32 dstSliceRankId = scratchSliceIndex;
		// 将当前数据拷贝到input暂存
		u64 srcOutputOffset = scratchSliceIndex * inputMemSize;
		u64 dstOutputOffset = dstSliceRankId * inputMemSize;
		DeviceMem currentDataMem = execMem.outputMem.range(dstOutputOffset, inputMemSize);
		CHK_SMART_PTR_NULL(currentDataMem);
		ret = HcclD2DMemcpyAsync(dispatcher_, execMem.inputMem, currentDataMem, const_cast<Stream&>(param.stream));
		CHK_PRT_RET(ret != HCCL_SUCCESS,
        	HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun] slice[%u] to input mem temp failed",
        	scratchSliceIndex), ret);
		// 环遍历到当前位置退出循环
		u32 loopCount = 0;
		while (srcSliceRankId != scratchSliceIndex) {
			HCCL_INFO("[CollAllGatherMeshGraphExecutor][KernelRun] slice[%u] to slice[%u]", srcSliceRankId, dstSliceRankId);
			srcOutputOffset = srcSliceRankId * inputMemSize;
			dstOutputOffset = dstSliceRankId * inputMemSize;
			DeviceMem srcOutputMem = execMem.outputMem.range(srcOutputOffset, inputMemSize);
			CHK_SMART_PTR_NULL(srcOutputMem);
			DeviceMem dstOutputMem = execMem.outputMem.range(dstOutputOffset, inputMemSize);
			CHK_SMART_PTR_NULL(dstOutputMem);
			ret = HcclD2DMemcpyAsync(dispatcher_, dstOutputMem, srcOutputMem, const_cast<Stream&>(param.stream));
			CHK_PRT_RET(ret != HCCL_SUCCESS,
        		HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun] slice[%u] to slice[%u] failed",
        		srcSliceRankId, dstSliceRankId), ret);
			reorderedDone[dstSliceRankId] = true;
			dstSliceRankId = srcSliceRankId;
			srcServerId = srcSliceRankId / level0RankSize;
			srcLocalRank = srcSliceRankId % level0RankSize;
			srcSliceRankId = srcLocalRank * level1RankSize + srcServerId;

			loopCount++;
			if (loopCount > topoAttr_.userRankSize) {
				HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun] ERROR: loop exceeds user rank size");
				return HCCL_E_INTERNAL;
			}
		}
		dstOutputOffset = dstSliceRankId * inputMemSize;
		DeviceMem ringEndMem = execMem.outputMem.range(dstOutputOffset, inputMemSize);
		CHK_SMART_PTR_NULL(ringEndMem);
		ret = HcclD2DMemcpyAsync(dispatcher_, ringEndMem, execMem.inputMem, const_cast<Stream&>(param.stream));
		CHK_PRT_RET(ret != HCCL_SUCCESS,
        	HCCL_ERROR("[CollAllGatherMeshGraphExecutor][KernelRun] temp input to slice[%u] failed",
        	dstSliceRankId), ret);
		reorderedDone[dstSliceRankId] = true;
	}
    return HCCL_SUCCESS;
}
```

### CollAllGatherMeshGraphExecutor::SelectTempAlg

[coll_all_gather_mesh_graph_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_graph_executor.cc#L237)

```cpp
HcclResult CollAllGatherMeshGraphExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
            // 1-单server-SDMA
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("AllGather mesh: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
            HCCL_INFO("AllGather mesh: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("AllGather mesh: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("AllGather mesh: algo is [%s] using halving-doubling algo inter-server.", (HCCL_ALGO_LEVEL1_MAP.at(algType_.algoLevel1)).c_str());
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}
```
