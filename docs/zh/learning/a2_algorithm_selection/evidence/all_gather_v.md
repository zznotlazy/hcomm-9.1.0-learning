# all_gather_v 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### AllGatherVOperator::SelectAlg

[all_gather_v_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_v_operator.cc#L32)

```cpp
HcclResult AllGatherVOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    HCCL_DEBUG("[%s] SelectAlg begins", __func__);
    if (isDiffDeviceType_) {
        HCCL_ERROR("[AllGatherVOperator][SelectAlg] AllGatherV not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[AllGatherVOperator][SelectAlg] AllGatherV only support A3, A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherVOperator][SelectAlg]tag[%s], AllGatherV failed, return[%d]", tag.c_str(), ret), ret);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);

        newTag = tag + level1Iter->second + algName;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### AllGatherVOperator::SelectAlgfor910B

[all_gather_v_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_v_operator.cc#L107)

```cpp
HcclResult AllGatherVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 dataSize = maxCount * unitSize;
    bool isBigData = false;

    if (dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
        isBigData = true;
    }
    
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER_V, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到NHR算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER_V);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherVOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+ graph[%u], "
                "reset algorithm to NHR.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    bool isAivMode = topoMatcher_->GetAivModeConfig()
                    && isSingleMeshAggregation_
                    && IsSupportAIVCopy(param.VDataDes.dataType)
                    && dataSize <= AIV_BIG_SIZE;

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (isAivMode) {
            algName = isBigData ? "AllGatherVMeshAivExecutor" : "AllGatherVMeshAivSmallCountExecutor";
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && !isSingleMeshAggregation_) {
            algName = "AllGatherVMeshOpbasePipelineExecutor";
        } else {
            algName = "AllGatherVMeshExecutor";
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        if (isSingleMeshAggregation_) {
            algName = "AllGatherVMeshGraphExecutor";
        } else if (deviceNumPerAggregation_ > 1 &&
                (dataSize > HCCL_SMALL_COUNT_1_MB || moduleNum_ <= MODULE_NUM_FOUR ||
                                    algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) {
            algName = "AllGatherVMeshGraphPipelineExecutor";
        } else {
            algName = "AllGatherVMeshExecutor"; 
        } 
    }
    HCCL_INFO("[SelectAlgfor910B] AllGatherV SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
```
