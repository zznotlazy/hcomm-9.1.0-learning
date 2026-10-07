# alltoall 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### AlltoAllOperator::UpdateAlltoAllCopyMode

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L125)

```cpp
void AlltoAllOperator::UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    std::string& copyMode)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= cclBufferManager_.GetInCCLbufferSize()) &&
                                   (maxRecvSize <= cclBufferManager_.GetInCCLbufferSize());
        if (isAlltoAllZCopyMode) {
           copyMode = "ZCopy";
        }
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], CopyMode[%s]", maxSendSize, maxRecvSize,
            cclBufferManager_.GetInCCLbufferSize(), copyMode.c_str());
    } else {
        // 图模式走ZCopy实现
        copyMode = "ZCopy";
    }
}
```

### AlltoAllOperator::SelectAlgforAiv

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L205)

```cpp
HcclResult AlltoAllOperator::SelectAlgforAiv(const OpParam& param, std::string& algName)
{
    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    if (deviceType_ == DevType::DEV_TYPE_910B &&
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALL &&
        !isSingleMeshAggregation_)
    {
        // aiv模式下910A2多server场景 alltoall算子
        bool isSingleAX = serverNum_ == 1 && moduleNum_ == 2;         // a+x单机跨module场景
        bool isSupportNpuDirect = isOpbase && param.supportRoceDirect;
        if (isSupportNpuDirect && ((isSingleAX && GetExternalInputIntraRoceSwitch() == 1) || !isSingleAX)) {
            // 单算子支持Roce直驱场景，使用DirectFullmesh
            algName = "AlltoAllDirectFullmeshAIVExecutor";
        } else {
            algName = "AlltoAllStagedAIVRdmaExecutor";
        }
    } else if (deviceType_ == DevType::DEV_TYPE_910_93 && serverNum_ > 1) {
        algName = "AlltoAllMeshAivFor91093Executor";
    } else if (deviceType_ == DevType::DEV_TYPE_910_93 &&
                !isOpbase &&
                param.opType == HcclCMDType::HCCL_CMD_ALLTOALL &&
                param.All2AllDataDes.sendCount * SIZE_TABLE[param.All2AllDataDes.sendType] <= AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE)
    {
        algName = "AlltoAllMeshAivSmallCountExecutor";
    } else {
        algName = "AlltoAllMeshAivExecutor";
    }

    HCCL_INFO("[SelectAlgforAlltoAll] AllToAll algName is [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
```

### AlltoAllOperator::SelectAlgforAlltoAll

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L238)

```cpp
HcclResult AlltoAllOperator::SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode,
    const ResourceLimit& resourceLimit)
{
    if (IsSatisfyAlltoAllAivCondition(param)) {
        CHK_RET(SelectAlgforAiv(param, algName));
        return HCCL_SUCCESS; // alltoall aiv不需要后面操作，直接返回
    }

    if (resourceLimit.ifCompileForAiv) {
        HCCL_DEBUG("[SelectAlgForAlltoAll] compile for aiv, early return.");
        return HCCL_SUCCESS;
    }

    std::vector<HcclAlgoType> algoTypeArr = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL);
    bool useOneLevelAlgorithm =
        algoTypeArr[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        algoTypeArr[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE;
        // 用户配置打平 alltoall

    CHK_PRT_RET(deviceType_ == DevType::DEV_TYPE_310P3 && userRankSize_ > MAX_310P_RANK_SIZE,
        HCCL_ERROR("[AlltoAllOperator][SelectAlgforAlltoAll]rankSize[%u] is not supported. AlltoAll/AlltoAllV does not "\
        "support the scenario where the rankSize is greater than 4.", userRankSize_), HCCL_E_NOT_SUPPORT);

    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);
    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode) {
        algName = "RunAlltoAllSingleExecutor";
        return HCCL_SUCCESS ;
    } else if (isCommon310P3DUO_) {
        algName = "RunAlltoAllVFor310PExecutor";
    } else if(IsSatisfyA2AContinuousPipelineFor91093Condition(param)) {
        algName = "RunAlltoAllVPipelineFor91093";
        HCCL_INFO("[SelectAlgforAlltoAll] AllToAll algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    } else if (IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
        isSingleMeshAggregation_, userRankSize_, cclBufferManager_.GetInCCLbufferSize()) ||
        (deviceType_ == DevType::DEV_TYPE_910_93 && param.aicpuUnfoldMode) || deviceType_ == DevType::DEV_TYPE_310P3) {
        bool isHCCS = (serverNum_ == 1 || (serverNum_ != 1 && !GetExternalInputInterHccsDisable()));
        if (param.supportSymmetricMemory && superPodNum_ == 1 && isHCCS) {
            algName = "RunAlltoAllFullMeshSymmetricMemory";
        } else {
            algName = "RunAlltoAllDirectFullmesh";
        }
        HCCL_INFO("[SelectAlgforAlltoAll] AllToAll algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    } else if (IsSatisfyAlltoallContinuousPipelineCondition(param)) {
        algName = "RunAlltoAllVContinuousPipeline"; // continuous pipeline 算法
        HCCL_INFO("[SelectAlgforAlltoAll] AllToAll algName is [%s]", algName.c_str());
        return HCCL_SUCCESS ;
    } else if (IsSatisfyAlltoallPipelineCondition()) {
        algName = "RunAlltoAllVTwoLevelPipeline";
    } else if (SatisfyIntraSuperPod(deviceType_, userRankSize_, useSuperPodMode_, superPodNum_) ||
        useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p || multiModuleDiffDeviceNumMode_ ||
        multiSuperPodDiffServerNumMode_) {
        algName = "RunAlltoAllVFullMesh";   //910B卡数不一致走这
    } else {
        algName = "RunAlltoAllVStaged";
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        // alltoallv
        CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL){
        // alltoallvc&&alltoall
        CHK_RET(GetAlltoAllvcSendRecvInfo(param.All2AllDataDes.sendCountMatrix, param.All2AllDataDes.sendType,
            param.All2AllDataDes.recvType));
    } else {
        HCCL_ERROR("[AlltoAllOperator][SelectAlgforAlltoAll] get wrong opType");
        return HCCL_E_PARA;
    }
    UpdateAlltoAllCopyMode(allMeshAggregationSendRecvInfo_, copyMode);

    HCCL_INFO("[SelectAlgforAlltoAll] AllToAll algName is [%s].", algName.c_str());
    return HCCL_SUCCESS;
}
```

### AlltoAllOperator::SelectAlg

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L314)

```cpp
HcclResult AlltoAllOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    ResourceLimit resourceLimit;
    return SelectAlg(tag, param, algName, newTag, resourceLimit);
}
```

### AlltoAllOperator::SelectAlg

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L321)

```cpp
HcclResult AlltoAllOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag, const ResourceLimit& resourceLimit)
{
    HcclResult ret;
    std::string copyMode = "BCopy";

    if (isDiffDeviceType_) {
        HCCL_ERROR("[AlltoAllOperator][SelectAlg] AlltoAll not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    }
    ret = SelectAlgforAlltoAll(param, algName, copyMode, resourceLimit);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SelectAlgforAlltoAll][SelectAlg]tag[%s], Alltoall failed, return[%d].", tag.c_str(), ret), ret);

    if (resourceLimit.ifCompileForAiv) {
        HCCL_DEBUG("[SelectAlg] AlltoAllOperator compile for aiv, early return.");
        return HCCL_SUCCESS;
    }

    bool useA2AAiv = IsSatisfyAlltoAllAivCondition(param);
    bool useDirectFullmesh = IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
            isSingleMeshAggregation_, userRankSize_, cclBufferManager_.GetInCCLbufferSize());

    bool aicpuUnfoldModeFor910B = deviceType_ == DevType::DEV_TYPE_910B && param.aicpuUnfoldMode;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (useDirectFullmesh || param.aicpuUnfoldMode) {
            newTag = tag + algName;
            // A2 alltoall AICPU展开需要区分Zcopy与bCopy
            if (aicpuUnfoldModeFor910B) {
                newTag += copyMode;
            }
        } else {
            newTag = tag + algName + copyMode;
        }
        newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    } else {
        newTag = tag;
    }
    if ((!useA2AAiv && !useDirectFullmesh && !param.aicpuUnfoldMode) || aicpuUnfoldModeFor910B) {
        CHK_RET(SetExcutorExtraInfo(algName, param));
    }
    return ret;
}
```

### AlltoAllOperator::IsSatisfyAlltoallPipelineCondition

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L518)

```cpp
bool AlltoAllOperator::IsSatisfyAlltoallPipelineCondition()
{
    bool cclBigEnough = cclBufferManager_.GetInCCLbufferSize() >= ALLTOALL_PIPELINE_MIN_CCL_SIZE;
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    bool isDefaultAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)
        || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD);
    bool isPipelineAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE);
    bool isSatisfyAlgType = (isPipelineAlgo || isDefaultAlgo) &&
        CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLTOALL) <= HCCL_FFTS_CAPACITY;
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallPipelineCondition]multiRankPerServer %u, "
        "isMultiServer %u, satisfyAlgType, %u, multiModuleDiffDeviceNumMode_ %u", multiRankPerServer,
        isMultiServer, isSatisfyAlgType, multiModuleDiffDeviceNumMode_);
    bool res = (deviceType_ == DevType::DEV_TYPE_910B && isSatisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_ && cclBigEnough);
    if (isSatisfyAlgType && !res) {
        HCCL_WARNING("AllToAll algo type is set to pipeline, but cclBigEnough is %u, multiRankPerServer is %u, "
            "isMultiServer is %u", cclBigEnough, multiRankPerServer, isMultiServer);
    }
    return res;
}
```

### AlltoAllOperator::IsSatisfyAlltoAllAivCondition

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L550)

```cpp
bool AlltoAllOperator::IsSatisfyAlltoAllAivCondition(const OpParam& param)
{
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isBufferEnough = !isOpbase || cclBufferManager_.GetInCCLbufferSize() >= AIV_ALL_TO_ALL_BIG_SIZE * MAX_RANK_SIZE;

    bool isSupportAiv = topoMatcher_->GetAivModeConfig()
                    && IsSupportAIVCopy(param.All2AllDataDes.sendType)
                    && userRankSize_ > 1
                    && isBufferEnough
                    && !retryEnable_;
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isSupportAiv) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(!IsSupportAIVCopy(param.All2AllDataDes.sendType), HCCL_ERROR("current data type[%s] not supported, support range: "\
            "[int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16]",
            GetDataTypeEnumStr(param.All2AllDataDes.sendType).c_str()), false);
        CHK_PRT_RET(userRankSize_ == 1, HCCL_ERROR("current userRankSize[%u] equal to 1.", userRankSize_), false);

        CHK_PRT_RET(!isBufferEnough, HCCL_ERROR("current buffer size[%llu] is not enough. isOpbase[%d]",
            cclBufferManager_.GetInCCLbufferSize(), isOpbase), false);

        CHK_PRT_RET(retryEnable_, HCCL_ERROR("retryEnable_[%d] is true.", retryEnable_), false);
        return false;
    }

    if (deviceType_ == DevType::DEV_TYPE_910B) {
        bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
            topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH ||
            userRankSize_ == moduleNum_;
        bool isModuleSatisfy = isSingleMeshAggregation_;

        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            // alltoall算子支持单机和多机场景
            if (!isSingleMeshAggregation_) {
                // 多机场景下当前不支持module间卡数不一致场景，集群中总的服务器数需要满足条件，cclbuffer大小需要满足
                isModuleSatisfy = isOpbase
                    && !multiModuleDiffDeviceNumMode_
                    && (moduleNum_ <= MAX_RMDA_RANK_SIZE)
                    && IsBufferSatisfyAlltoAllAivCondition(param);
            }
        }
        if (isOnlyAiv && (!isModuleSatisfy || !isMeshTopo)) {
            HCCL_ERROR("opType[%d] topoType_[%d] isSingleMeshAggregation_[%d] isOpbase[%d] multiModuleDiffDeviceNumMode_[%d] "\
                "moduleNum_[%u] IsBufferSatisfyAlltoAllAivCondition[%u]", param.opType, topoType_, isSingleMeshAggregation_,
                isOpbase, multiModuleDiffDeviceNumMode_, moduleNum_, IsBufferSatisfyAlltoAllAivCondition(param));
            return false;
        }
        return isSupportAiv && isModuleSatisfy && isMeshTopo;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        bool isSupportInterHccs = (superPodNum_ == 1 && serverNum_ > 1 && !GetExternalInputInterHccsDisable());

        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            u64 dataSize = param.All2AllDataDes.sendCount * SIZE_TABLE[param.All2AllDataDes.sendType];
            bool isAllToAllBufferEnough = serverNum_ == 1 && ((dataSize <= AIV_ALL_TO_ALL_A3_ENTRY_SIZE || isOnlyAiv) || !isOpbase);

            if (isOnlyAiv && (!isAllToAllBufferEnough && !isSupportInterHccs)) {
                HCCL_ERROR("serverNum_[%u] isOpbase[%d] dataSize[%llu] superPodNum_[%u] hccs disable[%u]",
                    serverNum_, isOpbase, dataSize, superPodNum_, GetExternalInputInterHccsDisable());
                return false;
            }
            return isSupportAiv && ( isAllToAllBufferEnough || isSupportInterHccs);
        }
        if (isOnlyAiv && serverNum_ != 1 && !isSupportInterHccs) {
            HCCL_ERROR("serverNum_[%u] isOpbase[%d] superPodNum_[%u] hccs disable[%u]",
                    serverNum_, isOpbase, superPodNum_, GetExternalInputInterHccsDisable());
            return false;
        }
        return isSupportAiv && (serverNum_ == 1 || isSupportInterHccs);
    }

    return false;
}
```

### AlltoAllOperator::IsBufferSatisfyAlltoAllAivCondition

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L626)

```cpp
bool AlltoAllOperator::IsBufferSatisfyAlltoAllAivCondition(const OpParam& param)
{
    u64 sendCount = param.All2AllDataDes.sendCount;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC && param.All2AllDataDes.sendCountMatrix != nullptr) {
        sendCount = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix));
    }

    u64 dataSize = SIZE_TABLE[param.All2AllDataDes.sendType];
    u64 scratchMemSize = sendCount * dataSize * userRankSize_;
    s64 dataSizeLimit = param.supportRoceDirect ? HCCL_SMALL_COUNT_4_MB : HCCL_SMALL_COUNT_190_KB;
    if (param.supportRoceDirect) {
        // 使用roce直驱时，total数据量应小于4M
        if(!(scratchMemSize <= static_cast<u64>(dataSizeLimit))) {
            HCCL_WARNING("[AlltoAllOperator]total dataSize[%llu] > [%lld], doesn't meet the aiv condition, select default algorithm",
                scratchMemSize, dataSizeLimit);
            return false;
        }
        // Roce直驱跨机场景，ccl需要预留flag位（与AIV保持一致，预留2M）
        scratchMemSize += AIV_FLAG_OFFSET;
    } else {
        // 每个rank的数据需要满足小于190K
        if (!(sendCount * dataSize <= static_cast<u64>(dataSizeLimit))) {
            HCCL_WARNING("[AlltoAllOperator]dataSize[%llu] > [%lld], doesn't meet the aiv condition, select default algorithm",
                sendCount * dataSize, dataSizeLimit);
            return false;
        }
    }

    // cclbuffer是否足够存储每个rank的中转数据
    if (!(scratchMemSize <= cclBufferManager_.GetInCCLbufferSize())) {
        HCCL_WARNING("[AlltoAllOperator]cclbuffer[%llu] < scratchMemSize[%llu]+32K, don't meet the aiv condition, "
            "please set HCCL_BUFFSIZE to increase cclbuffer", cclBufferManager_.GetInCCLbufferSize(), scratchMemSize);
        return false;
    }
    return true;
}
```

### AlltoAllOperator::IsSatisfyAlltoallContinuousPipelineCondition

[alltoall_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L688)

```cpp
bool AlltoAllOperator::IsSatisfyAlltoallContinuousPipelineCondition(const OpParam& param)
{
    std::vector<HcclAlgoType> algoTypeArr = topoMatcher_->GetAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALLV);
    bool useOneLevelAlgorithm =
        algoTypeArr[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        algoTypeArr[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE;

    bool cclBigEnough = cclBufferManager_.GetInCCLbufferSize() >= ALLTOALL_PIPELINE_MIN_CCL_SIZE;
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = (meshAggregationRankSize_ != 0) && ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    bool isDefaultAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING)
        || (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD);
    bool isPipelineAlgo = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE);
    bool satisfyAlgType = isPipelineAlgo || isDefaultAlgo;
    bool isAlltoAllv = param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV;
    bool res = (deviceType_ == DevType::DEV_TYPE_910B && isAlltoAllv && satisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_ && cclBigEnough && !useOneLevelAlgorithm && !param.isCapture);
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallContinuousPipelineCondition] isSatisfy[%d], isAlltoAllv %u,"
        "multiRankPerServer %u, isMultiServer %u, satisfyAlgType %u, multiModuleDiffDeviceNumMode_ %u,"
        "useOneLevelAlgorithm %u, isCapture %u.",
        res, isAlltoAllv, multiRankPerServer, isMultiServer, satisfyAlgType, multiModuleDiffDeviceNumMode_,
        useOneLevelAlgorithm, param.isCapture);
    return res;
}
```
