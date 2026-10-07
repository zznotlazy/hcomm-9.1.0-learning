# all_gather 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### AllGatherOperator::SelectAlg

[all_gather_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L38)

```cpp
HcclResult AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1) {
        algName = "AllGatherSingleExecutor";
        HCCL_INFO("[SelectAlg] AllGather SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    HcclResult ret;

    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    }  else {
        HCCL_ERROR("[AllGatherSelector][SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherSelector][SelectAlg]tag[%s], AllGather failed, return[%d]", tag.c_str(), ret), ret);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), 
                    HCCL_ERROR("[AllGatherSelector]level1: algType1[%u] is invalid.", algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    if (algName == "AllGatherARSFor91093Executor") {
        u32 ringSize = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_ALLGATHER);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_DEBUG("[AllGatherSelector][SelectAlg]newTag is [%s].", newTag.c_str());
    return ret;
}
```

### AllGatherOperator::SelectAlgfor910B

[all_gather_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L131)

```cpp
HcclResult AllGatherOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isAivMode = topoMatcher_->GetAivModeConfig()
                    && isSingleMeshAggregation_
                    && IsSupportAIVCopy(param.DataDes.dataType)
                    && (dataSize <= AIV_BIG_SIZE || isOnlyAiv);
    bool isA2APreAG = (param.tag == HCCL_ALLTOALL_PARA_ALLGATHER);
    if (isAivMode && !isA2APreAG) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_ALL_GATHER_SMALL_SIZE) {
            algName = "AllGatherMeshAivSmallCountExecutor"; 
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        } else {
            algName = "AllGatherMeshAivExecutor"; 
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        }
        return HCCL_SUCCESS;
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
         (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLGATHER));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    // 多机场景下aiv支持情况
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isSupportAivRdmaCount = !isSingleMeshAggregation_
                                && !multiModuleDiffDeviceNumMode_
                                && (((isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB)
                                && dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB
                                && dataSize <= HCCL_SMALL_COUNT_256_KB) || isOnlyAiv);

    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    // 暂只支持单算子模式
    bool isCCLBufferGE16M = isOpbase && commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB;

    bool isAivRdmaMode = topoMatcher_->GetAivModeConfig()
                        && IsSupportAIVCopy(param.DataDes.dataType)
                        && isMeshTopo
                        && isCCLBufferGE16M
                        && isSupportAivRdmaCount;
    if (isAivRdmaMode && !isA2APreAG) {
        algName = "AllGatherAivRdmaExecutor";
    } else if (isMeshTopo) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (isSingleMeshAggregation_) {
                algName = "AllGatherMeshOpbaseExecutor";
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
                algName = "AllGatherMeshOpbasePipelineExecutor";
            }
        }
        if (algName.empty()) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && moduleNum_ > 1 &&
                deviceNumPerAggregation_ > 1 &&
                (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR ||
                    algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
                algName = "AllGatherMeshGraphPipelineExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
                       dataSize > HCCL_SMALL_COUNT_1_MB) {
                algName = "AllGatherMeshExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                algName = "AllGatherMeshGraphExecutor";
            }
        }
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivRdmaMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(!IsSupportAIVCopy(param.DataDes.dataType), HCCL_ERROR("current data type[%s] not supported, support range: "\
            "[int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16]",
            GetDataTypeEnumStr(param.DataDes.dataType).c_str()), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(!isMeshTopo, HCCL_ERROR("current topo type[%d] not supported", topoType_), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(!isCCLBufferGE16M, HCCL_ERROR("current isOpbase[%d] or commInputSize[%llu] or commOutputSize[%llu] not supported",
            isOpbase, commInputSize, commOutputSize), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(!isSingleMeshAggregation_ && multiModuleDiffDeviceNumMode_,
            HCCL_ERROR("The number of cards between servers in a multi-server setup must be consistent. "\
            "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d]",
            isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_), HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor910B] AllGather SelectAlgfor910B is algName [%s], current mode is [%u].", algName.c_str(), workflowMode_);
    return HCCL_SUCCESS;
}
```
