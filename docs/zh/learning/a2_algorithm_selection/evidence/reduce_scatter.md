# reduce_scatter 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### ReduceScatterOperator::SelectAlg

[reduce_scatter_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L42)

```cpp
HcclResult ReduceScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    ResourceLimit limit;
    return SelectAlg(tag, param, algName, newTag, limit);
}
```

### ReduceScatterOperator::SelectAlg

[reduce_scatter_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L49)

```cpp
HcclResult ReduceScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag, const ResourceLimit &limit)
{
    if (userRankSize_ == 1) {
        algName = "ReduceScatterSingleExecutor";
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
        ret = SelectAlgfor91093(param, algName, limit);
    }  else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterSelector][SelectAlg]tag[%s], ReduceScatter failed, return[%d]",
            tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType_.algoLevel1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType_.algoLevel1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_NO_INLINE = "_no_inline";
        newTag = (isInlineReduce && isRdmaReduce) ? newTag : newTag + REDUCE_SCATTER_NO_INLINE;
    }
    if (algName == "ReduceScatterARSFor91093Executor") {
        u32 ringSize = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### ReduceScatterOperator::SelectAlgfor910B

[reduce_scatter_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L164)

```cpp
HcclResult ReduceScatterOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize() / userRankSize_;

    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
        cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT) {
        if (multiModuleDiffDeviceNumMode_) {
            // 保序规约场景（多batch一致），当前不支持A2非对称场景
            HCCL_ERROR("[SelectAlgfor910B] reduce order preservation only support"
                " Symmetry(multiModuleDiffDeviceNumMode_[%d]).", multiModuleDiffDeviceNumMode_);
            return HCCL_E_NOT_SUPPORT;
        }
        if (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32
            || param.DataDes.dataType == HCCL_DATA_TYPE_BFP16) {
            // 只有浮点数存在多batch不一致的可能，整数天然一致
            if (param.aicpuUnfoldMode || topoMatcher_->GetAivModeConfig()) {
                // AIV / AICPU场景，规约保序优先级更高
                HCCL_WARNING("[SelectAlgfor910B]aicpuMode[%d], AivModeConfig[%d], "
                    "the AIV/AICPU mode does not support when the reduce order preservation is enabled.",
                    param.aicpuUnfoldMode, topoMatcher_->GetAivModeConfig());
            }
            algName = "ReduceScatterOrderPreservedExecutor";
            HCCL_INFO("[SelectAlgfor910B] ReduceScatterSelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
    }

    // 暂只支持单算子模式
    bool isCCLBufferGE16M = isOpbase &&
        (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);

    bool isSupportAivRdmaCount = !isSingleMeshAggregation_
                                && !multiModuleDiffDeviceNumMode_
                                && isMeshTopo
                                && (((isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB)
                                && dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB
                                && isCCLBufferGE16M
                                && dataSize <= HCCL_SMALL_COUNT_256_KB) || isOnlyAiv);

    bool isSupportAivDeter = isSingleMeshAggregation_
                            && serverNum_ == 1
                            && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE)
                            && ((dataSize * userRankSize_ <= HCCL_SMALL_COUNT_8_MB) || isOnlyAiv);

    bool isAivMode = topoMatcher_->GetAivModeConfig()
                    && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
                    && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE || isSupportAivDeter )
                    && ((isSingleMeshAggregation_ && (dataSize <= AIV_BIG_SIZE || isOnlyAiv)) || isSupportAivRdmaCount);
    if (isAivMode) {
        if (isSupportAivDeter) {
            if (dataSize * userRankSize_ <= HCCL_SMALL_COUNT_8_MB){
                algName = "ReduceScatterAivDeterSmallExecutor";
            } else {
                algName = "ReduceScatterAivDeterExecutor"; 
            }
            HCCL_INFO("[SelectAlgfor910B] ReduceScatter SelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
        if (isSupportAivRdmaCount) {
            algName = "ReduceScatterAivRdmaExecutor";
        } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterMeshAivSmallCountExecutor";
        } else {
            algName = "ReduceScatterMeshAivExecutor";
        }
        HCCL_INFO("[SelectAlgfor910BAIV] ReduceScatterSelectAlgfor910B is algName [%s].", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, dataSize, cclBufferSize, algTypeLevel1Tag,
            isInlineReduce, isRdmaReduce));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
         (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ "\
                "graph[%u], reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isMeshTopo) {
        if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE
            && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
            && deviceNumPerAggregation_ > DEVICE_TWO) {
            algName = "ReduceScatterDeterPipelineExecutor";
        } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            bool enableSmallCountDeterministicAlgo = !isSingleMeshAggregation_ &&
                IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
            if (SingleMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    algName = "ReduceScatterMeshDmaEliminationExecutor";
                }
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE &&
                algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
                IsMultiMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                algName = "ReduceScatterMeshOpbasePipelineExecutor";
            } else if (enableSmallCountDeterministicAlgo && ((dataSize <= HCCL_SMALL_COUNT_512_KB &&
                topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE) ||
                dataSize * userRankSize_< HCCL_SMALL_COUNT_512_KB)) {
                algName = "ReduceScatterMeshOpbaseSmallCountDeterministicExecutor";
            }
        } else {
            if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE &&
                    deviceNumPerAggregation_ > DEVICE_TWO) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    if (dataSize <= HCCL_SMALL_COUNT_1_MB &&
                        workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                        algName = "ReduceScatterMeshGraphExecutor";
                    } else {
                        algName = "ReduceScatterMeshExecutor";
                    }
                }
            }
        }
        if (algName.empty()) {
            if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && moduleNum_ > 1 &&
                deviceNumPerAggregation_ > 1 && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE &&
                IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
                (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR ||
                    algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
                algName = "ReduceScatterMeshGraphPipelineExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
                       dataSize > HCCL_SMALL_COUNT_1_MB) {
                algName = "ReduceScatterMeshExecutor";
            } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
                algName = "ReduceScatterMeshGraphExecutor";
            }
        }
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(!IsSupportAIVReduce(param.DataDes.dataType, param.reduceType), HCCL_ERROR("current data type[%s] or reduceType[%s] not supported, "\
            "data type support range:[int8, int16, int32, float16, float32, bfloat16] reduce type support range:[sum, max, min]",
            GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str()), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isSupportAivDeter, HCCL_ERROR("is not support aiv deter.isSingleMeshAggregation_[%d] isOpbase[%d] "\
            "deterministic config[%u] dataSize[%llu], serverNum_[%u]",
            isSingleMeshAggregation_, isOpbase, topoMatcher_->GetDeterministicConfig(), dataSize, serverNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isSingleMeshAggregation_ && multiModuleDiffDeviceNumMode_,
            HCCL_ERROR("The number of cards between servers in a multi-server setup must be consistent. "\
            "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d]",
            isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isMeshTopo, HCCL_ERROR("current topo type[%d] not supported", topoType_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isServNumPowOfTwo, HCCL_ERROR("server num[%u] is pow of two.", serverNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isCCLBufferGE16M, HCCL_ERROR("current isOpbase[%d] or commInputSize[%llu] or commOutputSize[%llu] not supported",
            isOpbase, commInputSize, commOutputSize), HCCL_E_NOT_SUPPORT);
        HCCL_ERROR("isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d] dataSize[%llu]",
            isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_, dataSize);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor910B] ReduceScatter SelectAlgfor910B is algName [%s], current mode is [%u].", algName.c_str(), workflowMode_);
    return HCCL_SUCCESS;
}
```
