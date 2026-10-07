# reduce_scatter_v 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### ReduceScatterVOperator::SelectAlg

[reduce_scatter_v_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_v_operator.cc#L29)

```cpp
HcclResult ReduceScatterVOperator::SelectAlg(const std::string& tag, const OpParam& param,
    std::string& algName, std::string& newTag)
{
    HcclResult ret;

    if (isDiffDeviceType_) {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg] ReduceScatterV not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg] ReduceScatterV only support A3, A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg]tag[%s], ReduceScatterV failed, return[%d]",
            tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = algType_.algoLevel1;
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
                HCCL_ERROR("level1: algType1[%u] is invalid.", algType1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.VDataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_V_NO_INLINE = "_no_inline";
        newTag = isInlineReduce ? newTag : newTag + REDUCE_SCATTER_V_NO_INLINE;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### ReduceScatterVOperator::SelectAlgfor910B

[reduce_scatter_v_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_v_operator.cc#L121)

```cpp
HcclResult ReduceScatterVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    //图模式切入确定性
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && !isSingleMeshAggregation_) {
        if (!multiModuleDiffDeviceNumMode_) {
            algName = "ReduceScatterVDeterExecutor";//多机图模式当前默认选中确定性算法
            if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING ||
                algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB ||
                algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)) {
                //只支持server间ring,NB和NHR，默认使能NHR
                algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
                HCCL_WARNING("[ReduceScatterVOperator][SelectAlgfor910B] only support ring, NB and NHR in AlgoLevel1 yet,"
                    " default algType is NHR.");
            }
            HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
            return HCCL_SUCCESS;
        } else {
            HCCL_ERROR("[ReduceScatterVOperator][SelectAlgfor910B] ReduceScatterV not support uneven devices in multiServer.");
            return HCCL_E_NOT_SUPPORT;
        }
    }
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    // Deterministic 确定性分支
    bool isDeterministic = topoMatcher_->GetDeterministicConfig() != DETERMINISTIC_DISABLE  
        && isMeshTopo && !multiModuleDiffDeviceNumMode_
        && (param.VDataDes.dataType == HCCL_DATA_TYPE_FP16 
            || param.VDataDes.dataType == HCCL_DATA_TYPE_FP32
            || param.VDataDes.dataType == HCCL_DATA_TYPE_BFP16);
    if (isDeterministic) {
        // 只有浮点数存在不确定性
        algName = "ReduceScatterVDeterExecutor";
        HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    // pipeline算法回退: task数量多，如果超出FFTS子图限制，则重定向到NHR算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[ReduceScatterVOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ "\
                "graph[%u], reset algorithm to NHR.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    //Pipeline
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE 
        && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE 
        && IsMultiMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshOpbasePipelineExecutor";
        HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    } 

    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 maxDataSize = maxCount * unitSize; // 单位：字节
    // 910B单机AIV模式下ReduceScatterV算子当前仅支持单卡数据量不大于256M的场景，大于256M暂不支持
    bool isAivMode = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
        && topoMatcher_->GetAivModeConfig() 
        && isSingleMeshAggregation_ 
        && maxDataSize <= AIV_BIG_SIZE
        && IsSupportAIVReduce(param.VDataDes.dataType, param.reduceType)
        && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE;
    HCCL_INFO("[ReduceScatterVOperator][SelectAlgfor910B]isAivMode[%d], maxCount[%llu], maxDataSize[%llu], "
        "deterministic[%u], isSingleMeshAggregation[%d].", isAivMode, maxCount, maxDataSize,
        topoMatcher_->GetDeterministicConfig(), isSingleMeshAggregation_);

    if (isAivMode) {
        if (maxDataSize > AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterVAIVBigCountExecutor";
        } else {
            algName = "ReduceScatterVMeshAivSmallCountExecutor";
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshOpbaseExecutor";
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING ||
            algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB ||
            algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR)) {
            //只支持server间ring,NB和NHR，默认使能NHR
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[ReduceScatterVOperator][SelectAlgfor910B] only support ring, NB and NHR in AlgoLevel1 yet,"
                " default algType is NHR.");
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshExecutor";
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlgfor910B] ReduceScatterV only support inlinereduce.");
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO("[SelectAlgfor910B] ReduceScatterV SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
```
