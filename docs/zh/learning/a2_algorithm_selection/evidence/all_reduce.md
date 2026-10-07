# all_reduce 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### AllReduceOperator::GetCountTypeForDeterAllReduce

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L33)

```cpp
HcclDataCountType AllReduceOperator::GetCountTypeForDeterAllReduce(const u64 count, const HcclDataType dataType)
{
    u64 dataSize = SIZE_TABLE[dataType] * count;
    if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        if (dataSize <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else if ((dataSize <= HCCL_MEDIUM_COUNT_GRAPH_4_MB) && (deviceNumPerAggregation_ == DEVICE_EIGHT)) {
            return HcclDataCountType::HCCL_COUNT_MEDIUM;
        } else {
            return HcclDataCountType::HCCL_COUNT_HUGE;
        }
    } else {
        if (dataSize <= HCCL_SMALL_COUNT_128_KB) {
            return HcclDataCountType::HCCL_COUNT_SMALL;
        } else {
            if (deviceNumPerAggregation_ == DEVICE_EIGHT) {
                return HcclDataCountType::HCCL_COUNT_MEDIUM;
            } else {
                return HcclDataCountType::HCCL_COUNT_HUGE;
            }
        }
    }
}
```

### AllReduceOperator::SelectAlg

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L112)

```cpp
HcclResult AllReduceOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1) {
        algName = "AllReduceSingleExecutor";
        HCCL_INFO("[SelectAlg] AllReduce SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        if (is310PDuoCard_) {
            ret = SelectAlgfor310P3DUO(param, algName);
        } else {
            ret = SelectAlgfor310P3(param, algName);
        }
    } else if (Is310PDevice()) {
        ret = SelectAlgfor310PHelper(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("[AllReduceOperator][SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceSelector][SelectAlg]tag[%s], AllReduce failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = algType_.algoLevel1;
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
                algType1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string ALL_REDUCE_NO_INLINE = "_no_inline";
        newTag = (!isDiffDeviceType_ || (isDiffDeviceType_ && isInlineReduce && isRdmaReduce)) ?
            newTag : newTag + ALL_REDUCE_NO_INLINE;
    } else {
        newTag = tag;
    }
    if (algName == "AllReduceARSFor91093Executor") {
        u32 ringSize = CalcOptimalIntraRingsize(param.DataDes.count, param.DataDes.dataType, HcclCMDType::HCCL_CMD_ALLREDUCE);
        newTag += std::to_string(ringSize);
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### AllReduceOperator::SelectAlgfor910B

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L294)

```cpp
HcclResult AllReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节

    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));

    // aiv场景单独判断逻辑，满足AIV模式打开+支持AIVReduce+非确定性场景+外层为mesh+（单机/跨机小数据/跨机中数据）时进入分支
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isMesh = IsAlgTypeLevel0Mesh(algType_.algoLevel0);
    u64 rankCountSize = dataSize / deviceNumPerAggregation_;
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);

    bool isSupportAivRdmaSmallCount = !isSingleMeshAggregation_
                                    && !multiModuleDiffDeviceNumMode_
                                    && isServNumPowOfTwo
                                    && ((rankCountSize <= HCCL_SMALL_COUNT_190_KB || isOnlyAiv));

    bool isSupportAivRdmaMidCount = !isSingleMeshAggregation_
                                && !multiModuleDiffDeviceNumMode_
                                && (dataSize <= HCCL_MID_COUNT_16_MB);

    bool isSupportAivDeter = isSingleMeshAggregation_
                            && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE)
                            && (dataSize <= HCCL_SMALL_COUNT_8_MB);

    bool isCCLBufferGE16M = !isOpbase ||
        (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);

    bool isBarrierOp = param.syncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE;   // Barrier算子不使能AIV
    bool isAivMode = (topoMatcher_->GetAivModeConfig() && !isBarrierOp)
                    && IsSupportAIVReduce(param.DataDes.dataType, param.reduceType)
                    && isMesh
                    && isCCLBufferGE16M
                    && (isSingleMeshAggregation_ || isSupportAivRdmaSmallCount || isSupportAivRdmaMidCount)
                    && (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE || isSupportAivDeter);
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(
            HcclCMDType::HCCL_CMD_ALLREDUCE, dataSize, commInputSize,
            algTypeLevel1Tag, isInlineReduce, isRdmaReduce, isAivMode));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // AHC 算法选择逻辑
    if (((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) ||
         (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE))) {
        CHK_RET(SelectAlgforAHC(dataSize, AHCOpType::AHC_OP_TYPE_ALLREDUCE));
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    // 图模式不会重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLREDUCE);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_WARNING("[AllReduceOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT && multiModuleDiffDeviceNumMode_) {
        // 保序规约场景（多batch一致），当前不支持非对称场景
        HCCL_ERROR("[SelectAlgfor910B] reduce order preservation only support Symmetry("
            "multiModuleDiffDeviceNumMode_[%d]).", multiModuleDiffDeviceNumMode_);
        return HCCL_E_NOT_SUPPORT;
    }

    if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT &&
        (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
        param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)) {
        if (param.aicpuUnfoldMode || (topoMatcher_->GetAivModeConfig() && !isBarrierOp)) {
            // AIV / AICPU场景，规约保序优先级更高
            HCCL_WARNING("[SelectAlgfor910B]aicpuMode[%d], AivModeConfig[%d], "
                "the Aiv/AICPU mode does not support when the reduce order preservation is enabled.",
                param.aicpuUnfoldMode, topoMatcher_->GetAivModeConfig());
        }
        // 只有浮点数存在多batch不一致的可能，整数天然一致
        algName = "AllReduceOrderPreservedExecutor";
    } else if (isAivMode) {
        if (isSupportAivDeter) {
            if (dataSize <= HCCL_SMALL_COUNT_8_MB){
                algName = "AllReduceAivDeterSmallExecutor"; 
            }else{
                algName = "AllReduceAivDeterExecutor"; 
            }
            HCCL_INFO("[SelectAlgfor910B] AllReduce SelectAlgfor910B is algName [%s].", algName.c_str());
            return HCCL_SUCCESS;
        }
        bool isOpbaseBigCount = isOpbase && (dataSize >= AIV_ALL_REDUCE_BIG_SIZE);
        HCCL_INFO("[SelectAlgfor910B] Select AivMode Alg: DataSize[%llu], RankCountSize[%llu], DeviceNumPerAgg [%u]",
            dataSize, rankCountSize, deviceNumPerAggregation_);
        if (isSupportAivRdmaSmallCount) {
            algName = "AllReduceSmallCountAivRdmaExecutor";  // 多server，满足二次幂，小数据量（单卡190K以内）
        } else if (isSupportAivRdmaMidCount) {
            algName = "AllReduceMidCountAivRdmaExecutor";  // 多server，中小数据量（总数据量16M以内）
        } else if (isOpbaseBigCount || !isOpbase) {
            algName = "AllReduceMeshAivExecutor"; // 单server，单算子AIV模式大数据 和 图模式AIV 共用一个Executor
        } else {
            algName = "AllReduceMeshAivSmallCountExecutor"; // 单server，单算子AIV模式小数据单独一个Executor
        }
    // 小于等于两卡场景单独判断逻辑
    } else if (deviceNumPerAggregation_ <= DEVICE_TWO) {
        // 动态图算子融合场景？
        if ((param.inputPtr == commInputPtr) && (param.outputPtr == commOutputPtr &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isMeshTopo) {
            algName = "AllReduceMeshExecutor";
        // 两卡不存在确定性问题 server内
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            ret = MeshTopoSelector(algName, dataSize);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] AllReduce MeshTopoSelector failed, return[%d]", ret), ret);
        // 标卡场景（只有2p）
        } else if (Is2U2PInfer()) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isInlineReduce) {
                algName = "AllReduceMeshOneshotLoopExecutor";
            } else {
                algName = "AllReduceRingExecutor";
            }
        // 多机单卡/两卡 pipeline需单独做判断(pipeline无确定性算法，并只支持单算子模式）
        } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE &&
            algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
            IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
                if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    algName = "AllReduceMeshOpbasePipelineExecutor";
                } else {
                    algName = "AllReduceMeshGraphPipelineExecutor";
                }
        // 常规910B为mesh拓扑
        } else if (isMeshTopo) {
            algName = "AllReduceMeshExecutor";
        // 多机单卡topo为ring
        } else if (isRingTopo) {
            algName = "AllReduceRingExecutor";
        // 通信域打平场景
        } else {
            algName = "AllReduceComm";
        }
    // 多卡场景
    } else {
        if (isMeshTopo) {
            if ((param.inputPtr == commInputPtr) && (param.outputPtr == commOutputPtr &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
                algName = "AllReduceMeshExecutor";
            // 非确定性算法
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE) {
                ret = NonDeterministicSelector(param, algName, dataSize);
            // 确定性算法
            } else {
                ret = DeterministicSelector(param, algName);
            }
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[SelectAlgfor910B] AllReduce SelectAlgfor910B failed, return[%d]", ret), ret);
            if (algName.empty()) {
                algName = "AllReduceMeshExecutor";
            }
        } else {
            algName = "AllReduceComm";
        }
    }
    // 如果配置了aiv only,但是实际没有选择aiv算法,需要通过DFX打印出具体原因
    if (isOnlyAiv && !isAivMode) {
        HCCL_ERROR("The current conditions do not meet the aiv only execution criteria because:");
        CHK_PRT_RET(!IsSupportAIVReduce(param.DataDes.dataType, param.reduceType), HCCL_ERROR("current data type[%s] or reduceType[%s] not supported, "\
            "data type support range:[int8, int16, int32, float16, float32, bfloat16] reduce type support range:[sum, max, min]",
            GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str()), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isMesh, HCCL_ERROR("current algoLevel0Mesh[%d] not supported", algType_.algoLevel0), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isCCLBufferGE16M, HCCL_ERROR("current isOpbase[%d] or commInputSize[%llu] or commOutputSize[%llu] not supported",
            isOpbase, commInputSize, commOutputSize), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isSingleMeshAggregation_ && multiModuleDiffDeviceNumMode_,
            HCCL_ERROR("The number of cards between servers in a multi-server setup must be consistent. "\
            "isSingleMeshAggregation_[%d] multiModuleDiffDeviceNumMode_[%d]",
            isSingleMeshAggregation_, multiModuleDiffDeviceNumMode_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isServNumPowOfTwo, HCCL_ERROR("server num[%u] is pow of two.", serverNum_), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isSupportAivRdmaMidCount, HCCL_ERROR("current data size[%llu] not support aiv rdma mid count.", dataSize), HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(!isSupportAivDeter, HCCL_ERROR("is not support aiv deter.isSingleMeshAggregation_[%d] isOpbase[%d] deterministic config[%u], dataSize[%llu]",
            isSingleMeshAggregation_, isOpbase, topoMatcher_->GetDeterministicConfig(), dataSize), HCCL_E_NOT_SUPPORT);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgfor910B] AllReduce SelectAlgfor910B is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}
```

### AllReduceOperator::MeshTopoSelector

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L503)

```cpp
HcclResult AllReduceOperator::MeshTopoSelector(std::string& algName, u64 unitSize)
{
    // 单算子选择逻辑
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (unitSize <= HCCL_SMALL_COUNT_256_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshOpbaseLoopExecutor";
        }
    // 图模式选择逻辑
    } else {
        if (unitSize  <= HCCL_SMALL_COUNT_GRAPH_64_KB) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else {
            algName = "AllReduceMeshExecutor";
        }
    }
    return HCCL_SUCCESS;
}
```

### AllReduceOperator::NonDeterministicSelector

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L523)

```cpp
HcclResult AllReduceOperator::NonDeterministicSelector(const OpParam& param, std::string& algName, u64 dataSize)
{
    const bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    if (isOpbase) {
        if (IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
            algName = "AllReduceMeshOpbasePipelineExecutor";
        } else if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
            if (dataSize <= HCCL_SMALL_COUNT_256_KB) {
                algName = "AllReduceMeshSmallCountExecutor";
            } else {
                algName = "AllReduceMeshOpbaseLoopExecutor";
            }
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        IsMultiMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
            algName = "AllReduceMeshGraphPipelineExecutor";
    }
    if (!algName.empty() || !isOpbase) {
        return HCCL_SUCCESS;
    }
    const bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    // 单算子 + 数据量小于512kB
    if (dataSize < HCCL_SMALL_COUNT_512_KB && !isSingleMeshAggregation_ && isInlineReduce) {
        algName = "AllReduceMeshOpbaseSmallCountDeterministicExecutor";
    }
    return HCCL_SUCCESS;
}
```

### AllReduceOperator::DeterministicSelector

[all_reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L554)

```cpp
HcclResult AllReduceOperator::DeterministicSelector(const OpParam& param, std::string& algName)
{
    // 确定性图和单算子归一流程
    HcclDataCountType countType = GetCountTypeForDeterAllReduce(param.DataDes.count, param.DataDes.dataType);
    const bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    const bool isInlineReduce =
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);

    if (isOpbase && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && deviceNumPerAggregation_ > DEVICE_TWO) {
        u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        if (dataSize >= deviceNumPerAggregation_ * HCCL_MIN_SLICE_ALIGN) {
            algName = "AllReduceDeterPipelineExecutor";
            return HCCL_SUCCESS;
        }
    }
    if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
        if (countType == HcclDataCountType::HCCL_COUNT_SMALL) {
            algName = "AllReduceMeshSmallCountExecutor";
        } else if (countType == HcclDataCountType::HCCL_COUNT_MEDIUM) {
            algName = "AllReduceMeshMidCountLoopExecutor";
        } else { 
            algName = "AllReduceMeshOneshotLoopExecutor";
        }
    } else {
        u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        if (isOpbase && !isSingleMeshAggregation_ && isInlineReduce) {
            if (dataSize <= HCCL_SMALL_COUNT_512_KB) {
                // 单算子 + 确定性 + 数据量小于512kB
                algName = "AllReduceMeshOpbaseSmallCountDeterministicExecutor";
            } else {
                algName = "AllReduceMeshOpbaseMidCountDeterministicExecutor";
            }
        }
    }
    return HCCL_SUCCESS;
}
```
