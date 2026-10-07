# broadcast 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### BroadCastOperator::SelectAlg

[broadcast_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/broadcast_operator.cc#L39)

```cpp
HcclResult BroadCastOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    isAivMode_ = false;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (Is310PDevice() && topoType_ == TopoType::TOPO_TYPE_2P_MESH) {
        ret = SelectAlgfor310P(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("BroadCastOperator[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastSelector][SelectAlg]tag[%s], broadcast failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else if (isAivMode_ || (Is310P3Common(isHaveCpuRank_, deviceType_) &&
                            (algType_.algoLevel1 != AlgTypeLevel1::ALG_LEVEL1_HD))) {
        newTag = tag + algName;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
        newTag = tag + algName;
        std::string appendTag = "";
        u32 serverNumPerSuperPod = superPodNum_ == 0 ? moduleNum_ : moduleNum_ / superPodNum_;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / deviceNumPerAggregation_ % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (superPodNum_ - (1 << static_cast<u32>(log2(superPodNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_ / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_") + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[BroadCastOperator][SelectAlg]tag is [%s]", tag);
        newTag = newTag + '_' + appendTag;
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    HCCL_DEBUG("[%s] SelectAlg for newTag", __func__);
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### BroadCastOperator::SelectAlgfor910B

[broadcast_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/broadcast_operator.cc#L152)

```cpp
HcclResult BroadCastOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    
    // 暂只支持单算子模式
    const u64 commInputSize = cclBufferManager_.GetInCCLbufferSize();
    const u64 commOutputSize = cclBufferManager_.GetOutCCLbufferSize();
    u64 dataSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    bool isCCLBufferGE16M = commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB;
    isAivMode_ = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_ && isOpbase && isCCLBufferGE16M &&
                     IsSupportAIVCopy(param.DataDes.dataType) && dataSize <= AIV_MAX_DATASIZE;
    if (isAivMode_) {
        algName = "BroadcastMeshAivExecutor";
    } else if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910B] broadcast SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}
```
