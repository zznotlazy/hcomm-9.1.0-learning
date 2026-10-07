# reduce 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### ReduceOperator::SelectAlg

[reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_operator.cc#L32)

```cpp
HcclResult ReduceOperator::SelectAlg(const std::string &tag, const OpParam &param, std::string &algName,
    std::string &newTag)
{
    HcclResult ret = HCCL_SUCCESS;

    if (userRankSize_ == 1) {
        algName = "ReduceSingleExecutor";
        HCCL_INFO("[SelectAlg] Reduce SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }

    newTag = param.tag;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD
        || algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD)) {
        std::string appendTag = "";
        u32 serverNumPerSuperPod = superPodNum_ == 0 ? moduleNum_ : moduleNum_ / superPodNum_;
        HCCL_DEBUG("[ReduceOperator][SelectAlg]serverNumPerSuperPod is %u", serverNumPerSuperPod);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
            u32 part1Size = FACTOR_TWO * (serverNumPerSuperPod - (1 << static_cast<u32>(log2(serverNumPerSuperPod))));
            u32 rootId = param.root / deviceNumPerAggregation_ % serverNumPerSuperPod;
            appendTag += "L1_" + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ReduceOperator]SelectAlg for algoLevel1");
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            u32 part1Size = FACTOR_TWO * (superPodNum_ - (1 << static_cast<u32>(log2(superPodNum_))));
            u32 rootId = param.root / deviceNumPerAggregation_ / serverNumPerSuperPod;
            appendTag += (appendTag.empty() ? "L2_" : "_L2_") + std::to_string((rootId >= part1Size) || ((rootId % FACTOR_TWO) == 0));
        }
        HCCL_DEBUG("[ReduceOperator][SelectAlg]tag is [%s]", tag);
        newTag = newTag + '_' + appendTag;
        if (GetExternalInputHcclEnableEntryLog() && param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    }

    if (isDiffDeviceType_) {
        HCCL_ERROR("[ReduceOperator][SelectAlg] Reduce not support diffDeviceType");
        return HCCL_E_NOT_SUPPORT;
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("ReduceOperator[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceSelector][SelectAlg]tag[%s], reduce failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType_.algoLevel1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType_.algoLevel1), HCCL_E_INTERNAL);
        newTag = newTag + level1Iter->second + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    return ret;
}
```

### ReduceOperator::SelectAlgfor910B

[reduce_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_operator.cc#L111)

```cpp
HcclResult ReduceOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    (void) param;
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    if (isMeshTopo) {
        algName = "ReduceMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceRingPlusHd";
    } else {
        algName = "ReduceComm";
    }

    HCCL_INFO("[SelectAlgfor910B] reduce SelectAlgfor910B is algName [%s].", algName.c_str());
    return HCCL_SUCCESS;
}
```
