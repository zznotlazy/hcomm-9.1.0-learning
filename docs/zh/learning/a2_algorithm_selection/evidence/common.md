# 公共选择函数

### CollAlgOperator::SelectAlg

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L53)

```cpp
HcclResult CollAlgOperator::SelectAlg(const std::string& tag,
    const OpParam& param, std::string& algName, std::string& newTag)
{
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlg

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L59)

```cpp
HcclResult CollAlgOperator::SelectAlg(const std::string& tag,
    const OpParam& param, std::string& algName, std::string& newTag, const ResourceLimit &limit)
{
    return SelectAlg(tag, param, algName, newTag);
}
```

### CollAlgOperator::SelectAlg

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L118)

```cpp
HcclResult CollAlgOperator::SelectAlg(const std::string& tag, const OpParam &param, const ResourceLimit &limit,
    std::string &algName, AlgDesc &algDesc, std::string &newTag)
{
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
    bool supportOnlyAiv = (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER || 
                               param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
                               param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                               param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ||
                               param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                               param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE);
    CHK_PRT_RET(isOnlyAiv && !supportOnlyAiv,
            HCCL_ERROR("[CollAlgOperator][SelectAlg] opType[%s] currently do not support aivonly",
                GetCMDTypeEnumStr(param.opType).c_str()), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(isOnlyAiv && userRankSize_ == 1 && supportOnlyAiv,
            HCCL_ERROR("[CollAlgOperator][SelectAlg] aivonly not support, please ensure rankNum is greater than one"),
                HCCL_E_NOT_SUPPORT);

    // 兼容老接口
    if (limit.ifLimit) {
        CHK_RET(SelectAlg(tag, param, algName, newTag, limit));
    } else {
        CHK_RET(SelectAlg(tag, param, algName, newTag));
    }

    // 非AIV算法提前返回, 采用兜底Executor
    if (algName.empty()) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec("SendExecutor", dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][SelectAlg]Fail to find executor for algName[DefaultExecutor]"),
            HCCL_E_PARA);
    } else {
        // 校验控核
        if (limit.ifLimit && deviceType_ == DevType::DEV_TYPE_910_93 && topoMatcher_->GetAivModeConfig()) {
            CHK_RET(SelectAlgFor91093WithCoreLimit(param, limit, algName));
        }

        // 从对应executor获取算法描述
        if (executor_.get() == nullptr) {
            executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
            CHK_PRT_RET(executor_.get() == nullptr,
                HCCL_ERROR("[CollAlgOperator][SelectAlg]Fail to find executor for algName[%s]", algName.c_str()),
                HCCL_E_PARA);
            CHK_RET(SetExecutorAttr(param));
        }
    }

    bool isLastSelect = algDesc.isLastSelect;
    algDesc = executor_->GetAlgDesc();
    // 打印维测日志
    if (UNLIKELY(GetDebugConfig() & HCCL_ALG) && isLastSelect) {
        // 获取展开模式，转换成字符串
        std::string opExpansionStr;
        CHK_RET(GetOpExpansionStr(param, algDesc, opExpansionStr));
        // 尝试获取确定性属性（如果Executor有声明自己是否为确定性）
        std::string appendStr = "";
        if (algDesc.deterministic >= 0) {
            appendStr += "deterministic[" + std::to_string(algDesc.deterministic) + "]";
        }
        // 打印关键维测内容
        bool isOpBase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
        HCCL_CONFIG_INFO(HCCL_ALG,
            "[%s] newTag[%s] algName[%s] userRank[%u] topoType[%d] algType[%s] "\
            "userRankSize[%u] level0Size[%u] moduleNum_[%u] level2Size[%u] ",
            __func__, newTag.c_str(), algName.c_str(), userRank_, topoType_, AlgTypeToStr(algDesc.algType).c_str(),
            userRankSize_, deviceNumPerAggregation_, moduleNum_, superPodNum_);
        HCCL_CONFIG_INFO(HCCL_ALG,
            "[%s] newTag[%s] "\
            "opExpansionMode[%s] isZeroCopy[%u] retryEnable[%u] isOpBase[%u] isCapture[%u] aivCoreLimit[%u] %s.",
            __func__, newTag.c_str(), 
            opExpansionStr.c_str(), algDesc.isZeroCopy, retryEnable_, isOpBase, param.isCapture, limit.aivCoreLimit, appendStr.c_str());
    }
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::CalcResRequest

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L240)

```cpp
HcclResult CollAlgOperator::CalcResRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    return executor_->CalcResRequest(param, resourceRequest);
}
```

### CollAlgOperator::Orchestrate

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L253)

```cpp
HcclResult CollAlgOperator::Orchestrate(const std::string& algName, OpParam& param, AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlgOperator][Orchestrate]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    executor_->SetAivClearEnable(aivClearEnable_);
    executor_->SetAlgOpContext(algOpContext_);
    executor_->SetOpCounter(opCounter_);
    return executor_->Orchestrate(param, algResource);
}
```

### CollAlgOperator::SetTopoAttr

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L306)

```cpp
void CollAlgOperator::SetTopoAttr(AlgConfigurator* algConfigurator)
{
    const HcclTopoAttr& topoAttr = algConfigurator->GetTopoAttr();
    serverNum_= topoAttr.serverNum;
    moduleNum_ = topoAttr.moduleNum;
    superPodNum_ = topoAttr.superPodNum;
    deviceNumPerServer_ = topoAttr.deviceNumPerServer;
    deviceNumPerAggregation_ = topoAttr.deviceNumPerAggregation;
    multiModuleDiffDeviceNumMode_ = topoAttr.multiModuleDiffDeviceNumMode;
    multiSuperPodDiffServerNumMode_ = topoAttr.multiSuperPodDiffServerNumMode;
    multiSuperPodDiffDeviceNumMode_ = topoAttr.multiSuperPodDiffDeviceNumMode;
    isDiffDeviceType_ = topoAttr.isDiffDeviceType;
    gcdDeviceNumPerAggregation_ = topoAttr.gcdDeviceNumPerAggregation;

    meshAggregationRankSize_ = topoAttr.meshAggregationRankSize;
    isDiffDeviceModule_ = topoAttr.isDiffDeviceModule;
    isSingleMeshAggregation_ = topoAttr.isSingleMeshAggregation;
    isAllRankSamePlane_ = topoAttr.isAllRankSamePlane;
    is310PDuoCard_ = topoAttr.is310PDuoCard;
    isCommon310P3DUO_ = topoAttr.isCommon310P3DUO;
    hccsPortNum_ = topoAttr.hccsPortNum;

    userRank_ = topoAttr.userRank;
    realUserRank_ = topoAttr.realUserRank;
    userRankSize_ = topoAttr.userRankSize;

    devicePhyId_ = topoAttr.devicePhyId;
    deviceLogicId_ = topoAttr.deviceLogicId;
    deviceType_ = topoAttr.deviceType;

    nicList_ = topoAttr.nicList;
    pairLinkCounter_ = topoAttr.pairLinkCounter;
    isSupportRdmaLite_ = topoAttr.isSupportRdmaLite;
    isSupportHccsAndSio_ = topoAttr.isSupportHccsAndSio;
    useSuperPodMode_ = topoAttr.useSuperPodMode;
    isARSDoubleRing_  = topoAttr.isARSDoubleRing;
    return;
}
```

### CollAlgOperator::SetAlgoAttr

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L345)

```cpp
void CollAlgOperator::SetAlgoAttr(AlgConfigurator* algConfigurator)
{
    const HcclAlgoAttr& algoAttr = algConfigurator->GetAlgoAttr();
    isHaveCpuRank_ = algoAttr.isHaveCpuRank;
    inlineReduceSwitchOn_ = algoAttr.inlineReduceSwitchOn;
    identifier_ = algoAttr.identifier;
    return;
}
```

### CollAlgOperator::SetExecutorAttr

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L354)

```cpp
HcclResult CollAlgOperator::SetExecutorAttr(const OpParam& param)
{
    CHK_RET(executor_->SetAlgType(algType_));
    CHK_RET(executor_->SetCCLInBuffer(cclBufferManager_.GetInCCLbufferSize()));

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        bool isSupportSDMAReduce = false;
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            isSupportSDMAReduce = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType,
                param.reduceType);
        } else {
            isSupportSDMAReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        }
        CHK_RET(executor_->SetIsSupportSDMAReduce(isSupportSDMAReduce));
    }
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::AutoSelectAlgTypeLevel1

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L404)

```cpp
HcclResult CollAlgOperator::AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                                    std::string &algTypeLevel1Tag, bool isInlineReduce,
                                                    bool isRdmaReduce, bool isAivMode)
{
    if (isSingleMeshAggregation_) {
        HCCL_INFO("[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, no need to choose algo.",
                  serverNum_, moduleNum_);
        return HCCL_SUCCESS;
    }

    // auto algo selection process
    if (isAlgoLevel1Default_) {
        // parse algType_ and get algTypeLevel1 and algTypeLevel0
        auto originalAlgTypeLevel0 = algType_.algoLevel0;
        // set algTypeLevel1
        AlgTypeLevel1 algTypeLevel1;
        CHK_RET(
            GetDefaultAlgoLevel1V2(
                hcclCMDType, countSize, cclBufferSize, algTypeLevel1, isInlineReduce, isRdmaReduce, isAivMode));
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("[AutoSelectAlgTypeLevel1] level1: algType[%u] is invalid.", algTypeLevel1),
            HCCL_E_INTERNAL);
        HCCL_INFO("[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, using %s algo",
                  serverNum_, moduleNum_, iter->second.c_str());
        algType_.algoLevel0 = originalAlgTypeLevel0;
        algType_.algoLevel1 = algTypeLevel1;
        // tag 增加所选的算法
        AppendTag(algTypeLevel1, algTypeLevel1Tag);
    }
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoForComm

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L437)

```cpp
HcclResult CollAlgOperator::SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    // 从map中查找对应的计算函数
    auto it = selectFuncMap_.find(hcclCMDType);
    if (it == selectFuncMap_.end()) {
        HCCL_ERROR("[Get][AlgTypeLevel1] The hcclCMDType %d is not supported.", hcclCMDType);
        return HCCL_E_NOT_SUPPORT;
    }
    return (it->second)(delay, curSize, bandWidth, algType);
}
```

### CollAlgOperator::CalcContextNumForPipeline

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L450)

```cpp
u32 CollAlgOperator::CalcContextNumForPipeline(HcclCMDType hcclCMDType)
{
    bool isDeterPipeline = topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE
        && (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE || hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    const u32 stepNum = moduleNum_;  // 通信步数
    const u32 hccsContextNumPerStep = 5 * (deviceNumPerAggregation_ - 1);   // SDMA跨片每步所需context数
    const u32 roceContextNumPerStep = 7;  // RDMA每步所需context数
    const u32 copyContextNumPerStep = 1;  // SDMA片内每步所需context数
    const u32 localReduceNumPerStep = isDeterPipeline ? (deviceNumPerAggregation_ - 1) : 0;
    const u32 contextNumPerStep = hccsContextNumPerStep + roceContextNumPerStep + copyContextNumPerStep
        + localReduceNumPerStep; // 小计
    const u32 barrierContextNum = 4;  // 通信结束时barrier操作所需context数

    switch (hcclCMDType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:             // fall-through
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:        // fall-through
        case HcclCMDType::HCCL_CMD_ALLGATHER: 
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:{
            const u32 copyContextNum = 1;    // 通信首尾所需context数量
            u32 contextNum = stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
            if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
                contextNum += contextNum;
            }
            if (isDeterPipeline) {
                contextNum += stepNum - 1; // 最后的local reduce
            }
            return contextNum;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLV:             // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:            // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            const u32 copyContextNum = 1 + moduleNum_;   // 通信首尾所需context数量
            return stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
        }
        default:
            return 0;
    }
}
```

### CollAlgOperator::GetDefaultAlgoLevel1V2

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L489)

```cpp
HcclResult CollAlgOperator::GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
    AlgTypeLevel1 &algType, bool isInlineReduce, bool isRdmaReduce, bool isAivMode)
{
    // pipeline mode is deployed,where there is multi-sever multi-device(insever) now,
    // since RDMA is not reduced by normal serial orchestration of tasks.
    // So pipeline mode is more dominant than normal serial orchestration now.
    auto originalAlgTypeLevel0 = algType_.algoLevel0;
    bool disdeterniminsticWithInlineReduce = isInlineReduce && isRdmaReduce &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE;
    bool deterniminsticWithInlineReduce = isInlineReduce && isRdmaReduce &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_ENABLE;

    // 对于不支持Rdma Lite的场景，下发性能较差，RS和AG需要一个很大的数据量（AR的一半）才能掩盖下发时间
    u64 pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (((hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && disdeterniminsticWithInlineReduce) ||
        hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER || hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER_V) &&
        deviceNumPerAggregation_ != 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
        CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }
    if (hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && deterniminsticWithInlineReduce &&
        deviceNumPerAggregation_ > 1 &&
        curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
        CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY
        && moduleNum_ > 1 && curSize >= HCCL_SMALL_COUNT_256_KB) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }

    // 对于不支持Rdma Lite的场景，下发性能较差，AllReduce需要一个较大的数据量才能掩盖下发时间
    pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_ALLREDUCE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // 计算每个slice的大小
        u64 allreduceCurSize = 0;
        allreduceCurSize = curSize / (moduleNum_ * deviceNumPerAggregation_);
        if (disdeterniminsticWithInlineReduce && deviceNumPerAggregation_ != 1 &&
            allreduceCurSize >= pipelineMinSize && !isAivMode && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
            CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
        if (deterniminsticWithInlineReduce &&
            deviceNumPerAggregation_ > 1 &&
            allreduceCurSize >= HCCL_SMALL_COUNT_1_MB && !isAivMode && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
            CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
    }
    u64 dataSizePerLoop = curSize > cclBufferSize ? cclBufferSize : curSize;
    float delay = LATENCY; // 静态时延 60 us;
    float bandWidth;
    CHK_RET(GetBandWidthPerNPU(1, userRankSize_, deviceNumPerAggregation_, bandWidth)); // 单位：GB/s
    bandWidth = bandWidth * GB2B; // 单位：B/s
    CHK_RET(SelectAlgoForComm(hcclCMDType, delay, dataSizePerLoop, bandWidth, algType));
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForReduceScatter

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L548)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD,
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForAllGather

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L587)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForAllGatherV

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L625)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForAllGatherV(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    HCCL_DEBUG("[%s] CollAlgOperator for SelectAlgoTypeForAllGatherV", __func__);
    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;

    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForGather

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L646)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForAllReduce

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L672)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = NHR_FACTOR_TWO * ceil(log2(moduleNum_)) * delay +
                NHR_FACTOR_TWO * static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                curSize / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // compare costs between NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = DOUBLE_SUB_HCCLCMD * ceil(log2(moduleNum_)) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForBroadcast

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L713)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth
                 * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = scatter + allgather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::SelectAlgoTypeForReduce

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L741)

```cpp
HcclResult CollAlgOperator::SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    HCCL_DEBUG("[CollAlgOperator]SelectAlgoTypeForReduce start");
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = reducescatter + gather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::Is2U2PInfer

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L775)

```cpp
bool CollAlgOperator::Is2U2PInfer()
{
    return ((deviceNumPerAggregation_ == HCCL_DEVICE_NUM_TWO) && (serverNum_ == 1) &&
            (deviceType_ == DevType::DEV_TYPE_910B) && (meshAggregationRankSize_ == HCCL_DEVICE_NUM_TWO) &&
            (pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0));
}
```

### CollAlgOperator::Is910BSingleMesh

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L782)

```cpp
bool CollAlgOperator::Is910BSingleMesh()
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isSingleMesh =
        (deviceType_ == DevType::DEV_TYPE_910B) && (isMeshTopo || Is2U2PInfer()) && (userRankSize_ != 1);
    return isSingleMesh;
}
```

### CollAlgOperator::NeedCreateSingleMeshPlane

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L792)

```cpp
bool CollAlgOperator::NeedCreateSingleMeshPlane(const bool isInlineReduce)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool meshSinglePlane = Is910BSingleMesh() && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_DISABLE &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    return meshSinglePlane;
}
```

### CollAlgOperator::SingleMeshInlineReduce

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L801)

```cpp
bool CollAlgOperator::SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool singleMeshInlineReduce = Is910BSingleMesh() && isInlineReduce && isSingleMeshAggregation_;
    return singleMeshInlineReduce;
}
```

### CollAlgOperator::IsMultiMeshInlineReduce

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L808)

```cpp
bool CollAlgOperator::IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool multiMeshInlineReduce = (deviceType_ == DevType::DEV_TYPE_910B) &&
                                 isMeshTopo && isInlineReduce && isRdmaReduce && (!isSingleMeshAggregation_);
    return multiMeshInlineReduce;
}
```

### CollAlgOperator::SelectAlgforAHC

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L895)

```cpp
HcclResult CollAlgOperator::SelectAlgforAHC(u64 dataSize, AHCOpType ahcOpType)
{
    if (multiModuleDiffDeviceNumMode_) {
        return HCCL_SUCCESS;
    }

    bool isAHCWholeConfig = (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_RESERVED &&
        (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE));

    CommPlane ahcSubGroupLevel = COMM_LEVEL1_AHC;
    if (isAHCWholeConfig) {
        if (deviceType_ != DevType::DEV_TYPE_910_93) {
            ahcSubGroupLevel = COMM_COMBINE;
        } else {
            ahcSubGroupLevel = COMM_COMBINE_ORDER;
        }
    } else if (deviceType_ != DevType::DEV_TYPE_910_93) {
        HCCL_DEBUG("[AHCAlgSelect] hccl algorithm: 910B not support level1 ahc, return ERROR.");
        return HCCL_E_PARA;
    }

    HCCL_INFO("[SelectAlgforAHC] ahcOpType[%u] isAHCWholeConfig[%u] AHClevel[%u] algType_[%u] deviceType_[%u]",
            ahcOpType, isAHCWholeConfig, ahcSubGroupLevel, algType_.algoLevel1 , deviceType_);

    AlgTypeLevel1 algTypeLevel1;

    std::vector<std::vector<std::vector<u32>>> globalSubGroups;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    CHK_RET(topoMatcher_->GetGlobalSubGroups(ahcSubGroupLevel, globalSubGroups));
    topoMatcher_->GetAHCAlgOption(ahcAlgOption);
 
    AHCAlgSelectParam ahcAlgSelectParam;
    ahcAlgSelectParam.opType = ahcOpType;
    ahcAlgSelectParam.dataSize = dataSize;

    //AHC 封装算法选择逻辑
    CHK_RET(AHCAlgSelect(algTypeLevel1, globalSubGroups, ahcAlgOption, ahcAlgSelectParam));
 
    topoMatcher_->SetAHCAlgOption(ahcAlgOption);

    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
    CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
                HCCL_ERROR("[AHCAlgSelect] level1: algType_[%u] is invalid.", algTypeLevel1),
                HCCL_E_INTERNAL);

    // 支持 AHC 自适应调节为 BROKE 类型
    if (algType_.algoLevel1 != algTypeLevel1 && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        algType_.algoLevel1 = algTypeLevel1;
    }

    HCCL_INFO("[AHCAlgSelect] hccl algorithm: there are %u server(%u module) in level1, using %s algo",
                serverNum_, moduleNum_, iter->second.c_str());

    return HCCL_SUCCESS;
}
```

### CollAlgOperator::AHCAlgSelect

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L952)

```cpp
HcclResult CollAlgOperator::AHCAlgSelect(AlgTypeLevel1 &algType, std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
    std::map<AHCConcOpType, TemplateType> &ahcAlgOption, AHCAlgSelectParam &ahcAlgSelectParam)
{
    // globalSubGroups 参数检查
    CHK_RET(CommAHCBaseInfo::CheckGlobalGroups(globalSubGroups));

    bool isAHCType = false;
    u32 minSubGroupSize = globalSubGroups[0][0].size();
    u32 maxSubGroupSize = globalSubGroups[0][0].size();
    for (u32 i = 1; i < globalSubGroups[0].size(); ++i) {
        if (globalSubGroups[0][i].size() < minSubGroupSize) {
            minSubGroupSize = globalSubGroups[0][i].size();
        }
        if (globalSubGroups[0][i].size() > maxSubGroupSize) {
            maxSubGroupSize = globalSubGroups[0][i].size();
        }
    }
    for (u32 i = 0; i < globalSubGroups[0].size(); ++i) {
        if (globalSubGroups[0][i].size()!= minSubGroupSize) {
            isAHCType = true;
            break;
        }
    }
  
    //多平面 reduce scatter 和 all gather 算子，强制写死成BROKE类型
    if (deviceNumPerServer_ != 1 && ahcAlgSelectParam.opType != AHCOpType::AHC_OP_TYPE_ALLREDUCE) {
        isAHCType = false;
    }

    //add AHC Conc Type logic here,  modify init Type  depend on the input para
    CHK_RET(AHCAlgOptionSelect(algType, globalSubGroups, ahcAlgOption, ahcAlgSelectParam));
    
    if (ahcAlgSelectParam.enableAlgAutoSelect == false) { // 关闭算法自适应功能时，默认设置AHC算法
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
        return HCCL_SUCCESS;
    }
 
    if (isAHCType) {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC; // 设置为 AHC 类型
    } else {
        algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE; // 设置为 BROKE 类型
    }

    HCCL_DEBUG("[AHCAlgSelect] end minSubGroupSize = %u maxSubGroupSize = %u isAHCType = %u", 
        minSubGroupSize, maxSubGroupSize, isAHCType);

    return HCCL_SUCCESS;
}
```

### CollAlgOperator::AHCAlgOptionSelect

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L1001)

```cpp
HcclResult CollAlgOperator::AHCAlgOptionSelect(const AlgTypeLevel1 &algType, std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
    std::map<AHCConcOpType, TemplateType> &ahcAlgOption, const AHCAlgSelectParam &ahcAlgSelectParam)
{
    (void) algType;
    (void) ahcAlgSelectParam;
    AHCConcOpType ahcConcOpType;
    //一层组间拼接时，分组数大于设定阈值则修改默认算法为NHR
    if(globalSubGroups[0].size() <= AHC_LEVEL0_GROUP_SIZE_THRESHOLD ) {
        HCCL_DEBUG("[AHCAlgSelect]  conc inter select type RING ");
        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_REDUCESCATTER_RING;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_REDUCE_RING;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_GATHER_RING;              
    } else {
        HCCL_DEBUG("[AHCAlgSelect]  conc inter select type NHR ");
        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_REDUCESCATTER_NHR;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLREDUCE};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_REDUCE_NHR;

        ahcConcOpType = {AHCLevel::AHC_LEVEL_0, ConcType::CONC_INTER, AHCOpType::AHC_OP_TYPE_ALLGATHER};
        ahcAlgOption[ahcConcOpType] = TemplateType::TEMPLATE_ALL_GATHER_NHR;
    }
    return HCCL_SUCCESS;
}
```

### CollAlgOperator::IsNeedStrictMode

[coll_alg_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L1105)

```cpp
bool CollAlgOperator::IsNeedStrictMode(const OpParam& param)
{
    bool isStrictMode = (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_STRICT)
                        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
                            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP64)
                        && (param.reduceType == HCCL_REDUCE_SUM || param.reduceType == HCCL_REDUCE_PROD)
                        && userRankSize_ >= MIN_STRICT_RANK_NUM;

    return isStrictMode;
}
```

## alg_configurator.cc

### algoAttr_

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L25)

```cpp
: algoAttr_(algoAttr), topoAttr_(topoAttr),
        deterministic_(GetExternalInputHcclDeterministicV2())
{ }
```

### AlgConfigurator::Init

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L31)

```cpp
HcclResult AlgConfigurator::Init(bool isHeterogComm)
{
    if (!isHeterogComm) {
        // 获取算法类型
        CHK_RET(SelectAlgType(topoAttr_.moduleNum, topoAttr_.deviceType, algType_));
        // 获取拓扑类型，根据算法类型转化
        CHK_RET(GetTopoTypeByAlgType(algType_[HcclCMDType::HCCL_CMD_ALL], topoAttr_.deviceType, topoType_));
    } else {
        topoType_ = TopoType::TOPO_TYPE_HETEROG;
    }

    HCCL_INFO("alg configurator init success.");
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SelectAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L46)

```cpp
HcclResult AlgConfigurator::SelectAlgType(u32 moduleNum, const DevType deviceType,
    std::map<HcclCMDType, AlgType>& algType)
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        CHK_RET(SelectCurrOpAlgType(moduleNum, deviceType, static_cast<HcclCMDType>(opType), algType));
    }
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SelectCurrOpAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L55)

```cpp
HcclResult AlgConfigurator::SelectCurrOpAlgType(
    u32 moduleNum, const DevType deviceType, HcclCMDType opType, std::map<HcclCMDType, AlgType>& algType)
{
    AlgTypeLevel0 algType0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    AlgTypeLevel2 algType2 = AlgTypeLevel2::ALG_LEVEL2_RESERVED; // 第2层拓扑算法, 待梳理后考虑是否和第0层、第1层算法归一

    std::vector<HcclAlgoType> algoCommConfig = algoAttr_.commAlgoConfig[opType];
    HcclAlgoType algoConfigLevel0 = algoCommConfig[HCCL_ALGO_LEVEL_0];
    HcclAlgoType algoConfigLevel1 = algoCommConfig[HCCL_ALGO_LEVEL_1];
    HcclAlgoType algoConfigLevel2 = algoCommConfig[HCCL_ALGO_LEVEL_2];

    bool isConfigAHC = (algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC ||
                        algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE);

    bool isConfigNULL = algoConfigLevel0 == HcclAlgoType::HCCL_ALGO_TYPE_NULL;

    HCCL_INFO("[AlgConfigurator][SelectCurrOpAlgType] isConfigAHC[%u] isConfigNULL[%u] multiModuleDiffDeviceNumMode[%u] "\
        "multiSuperPodDiffServerNumMode[%u] multiSuperPodDiffDeviceNumMode[%u]",isConfigAHC, isConfigNULL,
        topoAttr_.multiModuleDiffDeviceNumMode, topoAttr_.multiSuperPodDiffServerNumMode, topoAttr_.multiSuperPodDiffDeviceNumMode);
 
    bool isSupportCmdARS = (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        opType == HcclCMDType::HCCL_CMD_ALLREDUCE);
 
    bool isSupportCmdAHC = (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_ALL);
    
    // server内卡数不对称，且不是ARS的情况
    bool isNoARS = (topoAttr_.multiModuleDiffDeviceNumMode && !(isSupportCmdARS && 
        deviceType == DevType::DEV_TYPE_910_93 && !topoAttr_.multiSuperPodDiffDeviceNumMode));
 
    // server内卡数对称, server内卡数不对称情况isNoARS已经讨论完毕；
    // 超节点server非对称，且不是AHC的情况
    bool isNoAHC = (!topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode && 
        !(isSupportCmdAHC && isConfigAHC));

    if (Is310P3Common(algoAttr_.isHaveCpuRank, topoAttr_.deviceType)) {
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
    } else if (!topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode &&
               (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALL)) {
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(HcclAlgoType::HCCL_ALGO_TYPE_AHC, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        if (!topoAttr_.isStandardCard && deviceType != DevType::DEV_TYPE_910B && !topoAttr_.isDiffDeviceType) {
            if (topoAttr_.nicList.size() != DEVICE_EIGHT && topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT &&
                algType0 != AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
                HCCL_ERROR("[Set][AlgType]nicSize[%zu] error, algType is not 8P ring.", topoAttr_.nicList.size());
                return HCCL_E_PARA;
            }
        }
        HCCL_INFO("[AlgConfigurator][SelectCurrOpAlgType] multiModuleDiffDeviceNumMode is true, set ahc.");
    } else if (!topoAttr_.multiModuleDiffDeviceNumMode && topoAttr_.multiSuperPodDiffServerNumMode
                && deviceType == DevType::DEV_TYPE_910_93 && (opType == HcclCMDType::HCCL_CMD_ALLTOALL
                || opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
        // A3 ALLTOALL非对称拓扑
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(algoConfigLevel1, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        algType[opType].algoLevel2 = algType2;
        if (algoConfigLevel0 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT &&
            algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
            algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
            isAlgoLevel1Default_[opType] = false;
        }
    } else if ((isNoARS || isNoAHC) && !isConfigNULL) { // 多server不同卡模式，设置为单层拓扑类型
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
        isAlgoLevel1Default_[opType] = false;
        if (algoConfigLevel0 != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT ||
            algoConfigLevel1 != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_WARNING("multiModuleDiffDeviceNumMode[%d], multiSuperPodDiffServerNumMode_[%d], algorithm type [%d] is selected by force.", \
                         topoAttr_.multiModuleDiffDeviceNumMode, topoAttr_.multiSuperPodDiffServerNumMode, algType[opType].algoLevel0);
        }
        HCCL_INFO("[AlgConfigurator][SelectCurrOpAlgType] multiModuleDiffDeviceNumMode is true, set default ring.");
    } else if (algoAttr_.isHaveCpuRank) {
        algType[opType].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
        algType[opType].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_STAR;
    } else {
        CHK_RET(SetAlgoLevel0(algoConfigLevel0, algType0));
        CHK_RET(SetAlgoLevel1(algoConfigLevel1, moduleNum, algType1, opType));
        CHK_RET(SetAlgoLevel2(algoConfigLevel2, algType2));
        algType[opType].algoLevel0 = algType0;
        algType[opType].algoLevel1 = algType1;
        algType[opType].algoLevel2 = algType2;
        if (!topoAttr_.isStandardCard && deviceType != DevType::DEV_TYPE_910B && !topoAttr_.isDiffDeviceType) {
            if (topoAttr_.nicList.size() != DEVICE_EIGHT && topoAttr_.deviceNumPerAggregation == DEVICE_EIGHT &&
                algType0 != AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
                HCCL_ERROR("[Set][AlgType]nicSize[%zu] error, algType is not 8P ring.", topoAttr_.nicList.size());
                return HCCL_E_PARA;
            }
        }
    }

    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType0);
    CHK_PRT_RET(level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(), HCCL_ERROR("level0: algType0[%u] is invalid.",
        algType0), HCCL_E_INTERNAL);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
    CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
        algType1), HCCL_E_INTERNAL);
    auto level2Iter = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType2);
    CHK_PRT_RET(level2Iter == HCCL_ALGO_LEVEL2_NAME_MAP.end(),
        HCCL_ERROR("level2: algType2[%u] is invalid.", algType2), HCCL_E_INTERNAL);
    HCCL_RUN_INFO("Device Type[%u], average device count[%u], HccsNum[%u], SIONum[%u], HCCS_SW_NUM[%u], optype[%u]",
        deviceType, topoAttr_.deviceNumPerAggregation,
        topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size(),
        topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size(),
        topoAttr_.pairLinkInfo[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size(),
        opType);
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgoLevel0

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L173)

```cpp
HcclResult AlgConfigurator::SetAlgoLevel0(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (topoAttr_.isStandardCard) {
        CHK_RET(SetAlgoLevel0StandardCard(algoConfig, algType));
    } else {
        CHK_RET(SetAlgoLevel0Module(algoConfig, algType));
    }
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgoLevel1

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L183)

```cpp
HcclResult AlgConfigurator::SetAlgoLevel1(HcclAlgoType algoConfig, u32 moduleNum, AlgTypeLevel1 &algType, HcclCMDType opType)
{
    HcclAlgoType algoConfigShadow = algoConfig;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            HCCL_INFO("server num[%u]: level1:hdr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC:
            if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                HCCL_INFO("server num[%u]: level1:ahc algo is set.", moduleNum);
                break;
            } else if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc algo is not support, set default.", moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC;
                HCCL_INFO("server num[%u]: level1:ahc algo is set.", moduleNum);
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE:
            if (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is set.", moduleNum);
                break;
            } else if (opType < HcclCMDType::HCCL_CMD_ALL) {
                algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is not support, set default.", moduleNum);
                break;
            } else {
                algType = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
                HCCL_INFO("server num[%u]: level1:ahc broke algo is set.", moduleNum);
                return HCCL_SUCCESS;
            }
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not supported. the config is ignored.");
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            break;
        default:
            algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
            HCCL_INFO("server num[%u]: level1:default algo is set.", moduleNum);
            break;
    }

    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel1] algType[%u], deviceType_[%u], workflowmode[%u]", algType,
        topoAttr_.deviceType, GetWorkflowMode());
    if (algType == AlgTypeLevel1::ALG_LEVEL1_PIPELINE && ((topoAttr_.deviceType != DevType::DEV_TYPE_910B 
            && topoAttr_.deviceType != DevType::DEV_TYPE_910_93) ||
            GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algoConfigShadow = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        HCCL_WARNING("hccl algorithm: there are %u server in level1, config pipeline algo failed.", moduleNum);
    }

    if (algoConfigShadow == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
        if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
            isAlgoLevel1Default_[opType] = true;
        }
        CHK_RET(GetDefaultAlgoLevel1V1(moduleNum, algType));
    }
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetDefaultAlgoLevel1V1

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L270)

```cpp
HcclResult AlgConfigurator::GetDefaultAlgoLevel1V1(u32 moduleNum, AlgTypeLevel1 &algType) const
{
    if (moduleNum >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ?
            AlgTypeLevel1::ALG_LEVEL1_RING :
            AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AlgConfigurator][GetDefaultAlgoLevel1V1] algType[%u], moduleNum[%u]", algType, moduleNum);
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgoLevel2

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L289)

```cpp
HcclResult AlgConfigurator::SetAlgoLevel2(HcclAlgoType algoConfig, AlgTypeLevel2 &algType)
{
    u32 superPodNum = topoAttr_.superPodNum;
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel2::ALG_LEVEL2_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel2::ALG_LEVEL2_RING;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel2::ALG_LEVEL2_NB;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel2::ALG_LEVEL2_PIPELINE;
            break;
        default: {
            // Level2默认选择NHR算法
            algType = AlgTypeLevel2::ALG_LEVEL2_NHR;
            break;
        }
    }
    HCCL_DEBUG("[AlgConfigurator][SetAlgoLevel2]algType[%u], deviceType_[%u], superPodNum_[%u]",
        algType, topoAttr_.deviceType, superPodNum);
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgoLevel0StandardCard

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L319)

```cpp
HcclResult AlgConfigurator::SetAlgoLevel0StandardCard(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0StandardCard(algType));
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetDefaultAlgoLevel0StandardCard

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L334)

```cpp
HcclResult AlgConfigurator::GetDefaultAlgoLevel0StandardCard(AlgTypeLevel0 &algType) const
{
    if (topoAttr_.deviceNumPerAggregation == DEVICE_TWO) {
        if ((topoAttr_.deviceType == DevType::DEV_TYPE_910B)) {
            algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
        } else {
            algType = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
        }
    } else if (topoAttr_.deviceNumPerAggregation > DEVICE_TWO && topoAttr_.deviceNumPerAggregation <= DEVICE_EIGHT) {
        // 随标卡支持rank数变更
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (topoAttr_.deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        HCCL_ERROR("in standard card[num %u] there is no supported algo.", topoAttr_.deviceNumPerAggregation);
        return HCCL_E_PARA;
    }
    HCCL_DEBUG("[GetDefaultAlgoLevel0StandardCard] AlgTypeLevel0 is set to [%u].", algType);
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgoLevel0Module

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L355)

```cpp
HcclResult AlgConfigurator::SetAlgoLevel0Module(HcclAlgoType algoConfig, AlgTypeLevel0 &algType)
{
    if (algoConfig == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
        algType = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        return HCCL_SUCCESS;
    }

    if (algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT && algoConfig != HcclAlgoType::HCCL_ALGO_TYPE_NA) {
        HCCL_WARNING("level0:%d algo is not supported. the config is ignored.", algoConfig);
    }

    CHK_RET(GetDefaultAlgoLevel0Module(algType));
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetDefaultAlgoLevel0Module

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L370)

```cpp
HcclResult AlgConfigurator::GetDefaultAlgoLevel0Module(AlgTypeLevel0 &algType)
{
    u32 deviceNumPerAggregation = topoAttr_.deviceNumPerAggregation;
    if (deviceNumPerAggregation == DEVICE_EIGHT) {
        algType = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    } else if (deviceNumPerAggregation == DEVICE_FOUR) {
        algType = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    } else if (deviceNumPerAggregation == DEVICE_TWO) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else if (deviceNumPerAggregation == DEVICE_ONE) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    } else {
        algType = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
    }

    if ((topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] ==
        deviceNumPerAggregation * (deviceNumPerAggregation - 1) ||
        topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] ==
        FACTOR_NUM_TWO * deviceNumPerAggregation * (deviceNumPerAggregation - 1)) &&
        topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        algType = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module] AlgTypeLevel0 is set to ALG_LEVEL0_NP_MESH (HCCS links is enabled).");
    }

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        algType = IsHCCSSWNumEqualToTwiceSIONum() ? AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING :
                                                    AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    }
    if (topoAttr_.isDiffDeviceType) {
        HCCL_DEBUG("[GetDefaultAlgoLevel0Module]gcdDeviceNumPerAggregation [%u] isDiffDeviceType [%u] algType [%u]",
            topoAttr_.gcdDeviceNumPerAggregation, topoAttr_.deviceType, algType);
    }
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L405)

```cpp
bool AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum()
{
    u32 hccsSWNum = topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)];
    u32 sioNum = topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)];
    HCCL_DEBUG(
        "In pairLinkCounter_, the hccsSWNum is [%lu], the sioNum is [%lu], the deviceNumPerAggregation is [%lu]",
        hccsSWNum, sioNum, topoAttr_.deviceNumPerAggregation);
    if (hccsSWNum == 0) {
        return false;
    }
    if (sioNum == 0) {
        return false;
    }
    // The following 2 means that the device has no HCCS_SW link with itself and its companion linked by same SIO link.
    return (hccsSWNum == ((topoAttr_.deviceNumPerAggregation - 2) * topoAttr_.deviceNumPerAggregation)) &&
           (sioNum == topoAttr_.deviceNumPerAggregation);
}
```

### AlgConfigurator::CheckAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L423)

```cpp
HcclResult AlgConfigurator::CheckAlgType(const AlgType algType)
{
    if(algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        HCCL_ERROR("[Check][AlgType]errNo[0x%016llx] algType[%s] is not supported", HCCL_ERROR_CODE(HCCL_E_PARA), AlgTypeToStr(algType).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetLevel0AlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L432)

```cpp
AlgTypeLevel0 AlgConfigurator::GetLevel0AlgType(const AlgType algType) const
{
    if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_NP_STAR &&
        algType.algoLevel1 != AlgTypeLevel1::ALG_LEVEL1_STAR) {
        return algType.algoLevel0;
    }

    return AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
}
```

### AlgConfigurator::GetTopoTypeByAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L442)

```cpp
HcclResult AlgConfigurator::GetTopoTypeByAlgType(const AlgType &algType, const DevType deviceType,
    TopoType &topoType)
{
    CHK_RET(CheckAlgType(algType));
    CHK_RET(CheckDeviceType(deviceType));

    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            topoType = TopoType::TOPO_TYPE_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            topoType = TopoType::TOPO_TYPE_4P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_2P_MESH:
            topoType = TopoType::TOPO_TYPE_2P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING:
            topoType = TopoType::TOPO_TYPE_NP_SINGLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_1P_MESH:
            topoType = TopoType::TOPO_TYPE_1P_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_RING:
            topoType = TopoType::TOPO_TYPE_4P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            topoType = TopoType::TOPO_TYPE_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING:
        case AlgTypeLevel0::ALG_LEVEL0_RESERVED:
            topoType = TopoType::TOPO_TYPE_COMMON;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_STAR:
            topoType = TopoType::TOPO_TYPE_ES_MESH;
            break;
        default:
            HCCL_ERROR("[AlgConfigurator][GetTopoTypeByAlgType]errNo[0x%016llx] case: device type[%d](0~1:V910),"
                " algorithm[%s] is not support", HCCL_ERROR_CODE(HCCL_E_PARA), deviceType,
                AlgTypeToStr(algType).c_str());
            return HCCL_E_PARA;
    }

    HCCL_INFO("[AlgConfigurator][GetTopoTypeByAlgType]algtype[%s], devicetype[%d],topotype[%d] is selected",
        AlgTypeToStr(algType).c_str(), deviceType, topoType);
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L492)

```cpp
HcclResult AlgConfigurator::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    CHK_RET(CheckAlgType(algType));
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SetAlgType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L500)

```cpp
HcclResult AlgConfigurator::SetAlgType(AlgType algType, HcclCMDType opType)
{
    CHK_RET(CheckAlgType(algType));
    algType_[opType] = algType;
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::SupportDeterministicOptim

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L507)

```cpp
bool AlgConfigurator::SupportDeterministicOptim() const
{
    bool support = topoAttr_.isSingleMeshAggregation &&
                   topoAttr_.deviceNumPerAggregation > DEVICE_TWO &&
                   topoAttr_.deviceType == DevType::DEV_TYPE_910B &&
                   deterministic_ != DETERMINISTIC_DISABLE;
    return support;
}
```

### AlgConfigurator::GetTopoType

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L516)

```cpp
void AlgConfigurator::GetTopoType(TopoType &topoType)
{
    topoType = topoType_;
    return;
}
```

### AlgConfigurator::GetAlgTypeDirect

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L522)

```cpp
void AlgConfigurator::GetAlgTypeDirect(AlgType &algType, HcclCMDType opType)
{
    opType = (algType_.find(opType) == algType_.end() ? HcclCMDType::HCCL_CMD_INVALID : opType);
    algType = algType_[opType];
    return;
}
```

### AlgConfigurator::GetAlgoLevel1DefaultSwitch

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L529)

```cpp
HcclResult AlgConfigurator::GetAlgoLevel1DefaultSwitch(bool &isAlgoLevel1Default, HcclCMDType opType)
{
    isAlgoLevel1Default = isAlgoLevel1Default_[opType];
    return HCCL_SUCCESS;
}
```

### AlgConfigurator::GetTopoAttr

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L535)

```cpp
const HcclTopoAttr& AlgConfigurator::GetTopoAttr()
{
    return topoAttr_;
}
```

### AlgConfigurator::GetAlgoAttr

[alg_configurator.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L540)

```cpp
const HcclAlgoAttr& AlgConfigurator::GetAlgoAttr()
{
    return algoAttr_;
}
```

## alg_env_config.cc

### ResetAlgEnvConfigInitState

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L21)

```cpp
HcclResult ResetAlgEnvConfigInitState()
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    g_algEnvConfig.SetDefaultParams();
    return HCCL_SUCCESS;
}
```

### GetExternalInputHcclAlgoConfig

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L28)

```cpp
const std::vector<HcclAlgoType> GetExternalInputHcclAlgoConfig(HcclCMDType opType)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    return g_algEnvConfig.hcclAlgoConfig[opType];
}
```

### SetCommonAlgType

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L34)

```cpp
HcclResult SetCommonAlgType(std::vector<std::string> &algos)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    std::vector<HcclAlgoType> algType;
    CHK_RET(ParseAlgoString("all op type", algos[0], algType));
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        g_algEnvConfig.hcclAlgoConfig[static_cast<HcclCMDType>(opType)] = algType;
    }
    return HCCL_SUCCESS;
}
```

### SetSpecificAlgType

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L45)

```cpp
HcclResult SetSpecificAlgType(std::vector<std::string> &algos)
{
    std::lock_guard<std::mutex> lock(g_algEnvConfigMutex);
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (opStringName == "others") {
            std::vector<HcclAlgoType> algType;
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            CHK_RET(ParseAlgoString("others op type", remainAlgoConfig, algType));
            for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
                g_algEnvConfig.hcclAlgoConfig[static_cast<HcclCMDType>(opType)] = algType;
            }
        }
    }
    std::map<std::string, HcclCMDType> hcclOpTypeMap = {
        {"broadcast", HcclCMDType::HCCL_CMD_BROADCAST},
        {"allreduce", HcclCMDType::HCCL_CMD_ALLREDUCE},
        {"reduce", HcclCMDType::HCCL_CMD_REDUCE},
        {"send", HcclCMDType::HCCL_CMD_SEND},
        {"receive", HcclCMDType::HCCL_CMD_RECEIVE},
        {"allgather", HcclCMDType::HCCL_CMD_ALLGATHER},
        {"reducescatter", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
        {"alltoall", HcclCMDType::HCCL_CMD_ALLTOALL},
        {"gather", HcclCMDType::HCCL_CMD_GATHER},
        {"scatter", HcclCMDType::HCCL_CMD_SCATTER},
        {"sendrecv", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV},
    };
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (hcclOpTypeMap.find(opStringName) != hcclOpTypeMap.end()) {
            HcclCMDType optype = hcclOpTypeMap[opStringName];
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            std::vector<HcclAlgoType> algType;
            CHK_RET(ParseAlgoString(opStringName, remainAlgoConfig, algType));
            if (algType[0] == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
                HCCL_ERROR("[SetSpecificAlgType] specific config level0 not support null type.");
                return HCCL_E_PARA;
            }
            g_algEnvConfig.hcclAlgoConfig[optype] = algType;
        } else {
            HCCL_ERROR("[SetSpecificAlgType] specific config optype[%s] is invalid, please check",
                opStringName.c_str());
            return HCCL_E_PARA;
        }
    }
    g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALLV] =
        g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALL];
    g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALLVC] =
        g_algEnvConfig.hcclAlgoConfig[HcclCMDType::HCCL_CMD_ALLTOALL];
    return HCCL_SUCCESS;
}
```

### ParserHcclAlgoLevel

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L99)

```cpp
HcclResult ParserHcclAlgoLevel(const std::string &algoLevel, u32 &level, HcclAlgoType &algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, u32> hcclAlgoLevelMap = {
        {"level0", HCCL_ALGO_LEVEL_0},
        {"level1", HCCL_ALGO_LEVEL_1},
        {"level2", HCCL_ALGO_LEVEL_2},
        {"level3", HCCL_ALGO_LEVEL_3}
    };

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"AHC", HcclAlgoType::HCCL_ALGO_TYPE_AHC},
        {"AHC_BROKE", HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
    };

    auto iterAlgoLevel = hcclAlgoLevelMap.find(orginalLevel);
    if (iterAlgoLevel == hcclAlgoLevelMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, level %s is not supported.", orginalLevel.c_str());
        return HCCL_E_PARA;
    }

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }

    level = iterAlgoLevel->second;
    algoType = iterAlgoType->second;

    return HCCL_SUCCESS;
}
```

### ParseAlgoString

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L150)

```cpp
HcclResult ParseAlgoString(std::string opName, std::string &algoString, std::vector<HcclAlgoType> &algType)
{
    algType = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    std::vector<std::string> algoLevels;
    HcclResult ret = SplitHcclAlgoLevel(algoString, algoLevels);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "\
        "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
        algoString.c_str()), ret);
    for (auto algoLevel : algoLevels) {
        u32 level = 0;
        HcclAlgoType algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        ret = ParserHcclAlgoLevel(algoLevel, level, algo);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "\
            "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
            algoString.c_str()), ret);
        // 检查是否存在重复配置level
        if (algType[level] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_ERROR("[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "\
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str());
            return HCCL_E_PARA;
        }
        algType[level] = algo;
    }
    auto level0Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_0]);
    auto level1Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_1]);
    auto level2Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_2]);
    auto level3Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_3]);
    HCCL_RUN_INFO("hccl algo op %s config: config level0:%s, level1:%s, level2:%s, level3:%s",
        opName.c_str(),
        level0Iter->second.c_str(), level1Iter->second.c_str(),
        level2Iter->second.c_str(), level3Iter->second.c_str());
    return HCCL_SUCCESS;
}
```

### SplitHcclOpType

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L185)

```cpp
HcclResult SplitHcclOpType(const std::string &algoConfig, std::vector<std::string> &algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find("/");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][SplitHcclOpType] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    }
    algos.push_back(algoConfig.substr(0, found));
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclOpType(remainAlgoConfig, algos));
    }
    return HCCL_SUCCESS;
}
```

### CheckAlgoConfigValid

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L203)

```cpp
HcclResult CheckAlgoConfigValid(
    std::vector<std::string> &algos,
    bool& anyCommonConfig,
    bool& anySpecificConfig)
{
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        if ((found == 0) || (found == (algConfig.length() - 1))) {
            HCCL_ERROR("[Split][CheckAlgoConfigValid] algo config is invalid.");
            return HCCL_E_PARA;
        } else if (found != std::string::npos) {
            anySpecificConfig = true;
        } else {
            anyCommonConfig = true;
        }
    }
    if (anyCommonConfig && anySpecificConfig) {
        HCCL_ERROR("[CheckAlgoConfigValid]should not set both algo config way");
        return HCCL_E_PARA;
    }
    if (anyCommonConfig && algos.size() > 1) {
        HCCL_ERROR("[CheckAlgoConfigValid]should only set one common config");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
```

### SplitHcclAlgoLevel

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L230)

```cpp
HcclResult SplitHcclAlgoLevel(const std::string &algoConfig, std::vector<std::string> &algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find(";");
    if ((found == 0) || (found == (algoConfig.length() - 1))) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainAlgoConfig = algoConfig.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
    }
    algos.push_back(algoConfig.substr(0, found));

    if (algos.size() > HCCL_ALGO_LEVEL_NUM) {
        HCCL_ERROR("[Split][HcclAlgoLevel] algo config is invalid. algo level is more than %u.", HCCL_ALGO_LEVEL_NUM);
        return HCCL_E_PARA;
    }
    if (!remainAlgoConfig.empty()) {
        CHK_RET(SplitHcclAlgoLevel(remainAlgoConfig, algos));
    }

    return HCCL_SUCCESS;
}
```

### GetInternalExecTimeOut

[alg_env_config.cc](../../../../../src/legacy/ascend910/algorithm/impl/alg_env_config.cc#L255)

```cpp
s32 GetInternalExecTimeOut()
{
    double timeout = GetExternalInputHcclExecTimeOut();
    // 向上取整获取s32秒级超时时间
    return static_cast<s32>(std::ceil(timeout));
}
```

## coll_alg_utils.cc

### IsAlgTypeLevel0Mesh

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L18)

```cpp
bool IsAlgTypeLevel0Mesh(const AlgTypeLevel0 &originalAlgTypeLevel0)
{
    return originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH;
}
```

### IsAlltoAllvcSatisfyBufferSize

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L26)

```cpp
bool IsAlltoAllvcSatisfyBufferSize(const OpParam& param, u32 userRankSize, u64 cclbufferSize) {
    for (u32 i = 0; i < userRankSize; i++) {
        u64 maxSendLength = 0;
        u64 maxRecvLength = 0;
        // 计算每个rank需使用的中转内存大小是否满足cclbuffer大小
        for (u32 j = 0; j < userRankSize; j++) {
            u64 curSendCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i * userRankSize + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];

            u64 curRecvCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i + userRankSize * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];

            maxSendLength += curSendLength;
            maxRecvLength += curRecvLength;
        }
        if ((maxSendLength <= cclbufferSize) || (maxRecvLength <= cclbufferSize)) {
            return false;
        }
    }
    return true;
}
```

### IsSupportUnifiedMarch

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L50)

```cpp
bool IsSupportUnifiedMarch(const OpParam& param, const TopoType& topoType, u32 serverNum, u32 superPodNum)
{
    bool isGraphMode = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    bool isDoubleRing = topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    bool isSingleServer = (serverNum == 1) && (superPodNum == 1);
    return (param.aicpuUnfoldMode) && isDoubleRing && isGraphMode && isSingleServer;
}
```

### IsSupportDirectFullmeshForAlltoallv

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L58)

```cpp
bool IsSupportDirectFullmeshForAlltoallv(const OpParam& param, DevType deviceType, bool useSuperPodMode, u32 serverNum,
    bool isSingleMeshAggregation, u32 userRankSize, u64 cclbufferSize)
{
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool baseInfo = (deviceType == DevType::DEV_TYPE_910_93 || (deviceType == DevType::DEV_TYPE_910B && isOpbase));
    bool isHCCS = false;
    bool isSatisfyBuffer = true;
    if (deviceType == DevType::DEV_TYPE_910_93) {
        isHCCS = (serverNum > 1) ?
            (!GetExternalInputInterHccsDisable() && useSuperPodMode) : true;
    } else if (deviceType == DevType::DEV_TYPE_910B) {
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            // alltoallv算子单机和多机(小于8机64卡)acl graph场景都走directfullmesh算法，且不支持卡数不一致场景
            aclmdlRI rtModel = nullptr;
            bool isCapture = false;
            if (isOpbase) { // acl graph模式下获取capture信息，图模式需要规避
                HcclResult retCapture = GetStreamCaptureInfo(param.stream.ptr(), rtModel, isCapture);
                CHK_PRT_CONT(retCapture != HCCL_SUCCESS, 
                    HCCL_ERROR("Get capture status error. return[%d], capture model", retCapture));
            }
            isHCCS = (userRankSize <= MAX_ALLTOALLV_DIRECT_FULLMESH_RANKSIZE &&
                      serverNum <= MAX_ALLTOALLV_DIRECT_FULLMESH_SERVER_NUM && isCapture) ||
                      isSingleMeshAggregation;
            // A+X单机双module启用下， 未使能RDMA不能走directfullmesh算法
            bool isDifModule = serverNum == 1 && !isSingleMeshAggregation && userRankSize > HCCL_ALLTOALLV_P2P_SIZE;
            if (isDifModule && (GetExternalInputIntraRoceSwitch() == 0)) {
                isHCCS = false;
            }
        } else {
            // alltoall和alltoallvc在A2上仅支持单机
            isHCCS = (isSingleMeshAggregation) ? (true) : (false);
            if (isHCCS) {
                // A2场景下alltoall和alltoallvc需满足数据量大于cclbuffer大小条件
                isSatisfyBuffer = IsAlltoAllvcSatisfyBufferSize(param, userRankSize, cclbufferSize);
            }
        }
    }
    HCCL_DEBUG("[IsSupportDirectFullmeshForAlltoallv]baseInfo[%u], isOpbase[%u], isHCCS[%u], isSatisfyBuffer[%u]",
        baseInfo, isOpbase, isHCCS, isSatisfyBuffer);
    return baseInfo && isHCCS && isSatisfyBuffer;
}
```

### SatisfyIntraSuperPod

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L100)

```cpp
bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode, u32 superPodNum)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    bool isSingleSuperPod = superPodNum == 1;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    return (isDevice91093 && rankSizeSupport && isHCCS && isSingleSuperPod && isOpbase);
}
```

### FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L110)

```cpp
bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode,
    std::vector<HcclAlgoType> algoConfig)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool twoLevelIntraUseMesh =
        (algoConfig[HCCL_ALGO_LEVEL_0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        algoConfig[HCCL_ALGO_LEVEL_1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition]isDevice91093 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91093, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91093),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] AllToAll read only algorithm only "
            "support 910_93 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] AllToAll read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91093 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
}
```

### IsConfigAHCAlgo

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L130)

```cpp
bool IsConfigAHCAlgo(std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfigMap)
{
    const std::set<HcclCMDType> hcclSupportAHCOpSet = {
        HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLGATHER
    };
 
    for (const auto& opType : hcclSupportAHCOpSet) {
        HcclAlgoType algoConfigLevel1 = algoConfigMap[opType][HCCL_ALGO_LEVEL_1];
        bool isConfigAHC =
            (algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC ||
            algoConfigLevel1 == HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE);
        if (isConfigAHC) {
            return true;
        }
    }
    return false;
}
```

### GetAlgoString

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L149)

```cpp
std::string GetAlgoString(const std::map<keyType, std::string>& levelMap, keyType key) {
    auto iter = levelMap.find(key);
    if (iter == levelMap.end()) {
        return "invalid algo type";
    } else {
        return iter->second;
    }
}
```

### AlgTypeToStr

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L158)

```cpp
std::string AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel0 algTypeLevel0 = algType.algoLevel0;
    AlgTypeLevel1 algTypeLevel1 = algType.algoLevel1;
    AlgTypeLevel2 algTypeLevel2 = algType.algoLevel2;
    std::string algStrLevel0 = GetAlgoString(HCCL_ALGO_LEVEL0_NAME_MAP, algTypeLevel0);
    std::string algStrLevel1 = GetAlgoString(HCCL_ALGO_LEVEL1_NAME_MAP, algTypeLevel1);
    std::string algStrLevel2 = GetAlgoString(HCCL_ALGO_LEVEL2_NAME_MAP, algTypeLevel2);
    std::string algStr;
    algStr.append("level0:").append(algStrLevel0).append(",level1:").append(algStrLevel1).append(",level2:").append(algStrLevel2);
    return algStr;
}
```

### CalculatePiplineSliceNum

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L176)

```cpp
u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum)
{
    u64 piplineSliceNum = 0;
    bool isInterRing = false;
    if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        isInterRing = true;
    } else {
        isInterRing = false;
    }

    do {
        if (!GetExternalInputHcclEnablePipline()) {
            break;
        }
        /* 不支持pipline流水的场景 */
        // 支持的硬件场景
        if (deviceType != DevType::DEV_TYPE_910B || deviceNumPerAggregation < HCCL_DEVICE_NUM_TWO ||
            moduleNum < HCCL_DEVICE_NUM_TWO) {
            break;
        }
        // 支持的算子和算法场景
        if (opType != HcclCMDType::HCCL_CMD_ALLREDUCE ||
           (isInterRing && moduleNum > MAX_RING_PIPLINE_SERVER_NUM)) {
            break;
        }
        u64 sliceNumTemp = std::min(dataSize / deviceNumPerAggregation / MIN_PER_LINK_DATA_SIZE, MAX_PIPLINE_SLICE_NUM);
        // 图模式切分数量 <= 1时, 不做切分
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            sliceNumTemp <= MIN_PIPLINE_SLICE_NUM) {
            break;
        }

        /* 支持pipline流水, 但数据量不足以进行切分的场景 */
        // Server间使用Ring算法, 且单Server数据量<64KB时, 不做切分
        if ((isInterRing && dataSize / moduleNum < MIN_RING_DATA_SIZE)) {
            sliceNumTemp = 1;
        }
        // 支持pipline但数据量不满足切分条件时, 返回1, 用于单算子场景预申请流资源
        piplineSliceNum = (sliceNumTemp == 0) ? 1 : sliceNumTemp;
    } while (0);
    return piplineSliceNum;
}
```

### HcclOpInplaceDefaultCase

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L220)

```cpp
bool HcclOpInplaceDefaultCase(const OpParam &param, u8 &isInplaceStatus)
{
    // unknown op
    if (param.inputPtr != param.outputPtr) {
        // 可以走重执行
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] != param.outputPtr[%p]. They do not overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] == param.outputPtr[%p]. They overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 1;
        return true;
    }
}
```

### IsInputOutputOverlap

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L237)

```cpp
bool IsInputOutputOverlap(const OpParam &param, u64 inputDataSize, u64 outputDataSize, u8 &isInplaceStatus)
{
    if (inputDataSize == 0 || outputDataSize == 0) {
        // 不存在overlap情况
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
    u64 inputStart = reinterpret_cast<u64>(param.inputPtr);
    u64 inputEnd = reinterpret_cast<u64>(param.inputPtr) + inputDataSize - 1;
    u64 outputStart = reinterpret_cast<u64>(param.outputPtr);
    u64 outputEnd = reinterpret_cast<u64>(param.outputPtr) + outputDataSize - 1;

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 2; // The status 2 is overlap with dataSize.
        return true;
    } else {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
}
```

### IsInputOutPtrNotNullPtr

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L264)

```cpp
bool IsInputOutPtrNotNullPtr(const OpParam &param, u8 &isInplaceStatus)
{
    if (param.inputPtr == nullptr || param.outputPtr == nullptr) {
        // 不存在overlap情况
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]param.tag[%s], the inputPtr[%p], the outputPtr[%p]."
            "They do not overlap.", param.tag.c_str(), param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        return true;
    }
}
```

### InplaceDataUnitSize

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L277)

```cpp
u32 InplaceDataUnitSize(const HcclCMDType &opType, const OpParam &param)
{
    u32 unitSize = 0;
    if (opType != HcclCMDType::HCCL_CMD_ALLTOALLV && opType != HcclCMDType::HCCL_CMD_ALLTOALLVC &&
        opType != HcclCMDType::HCCL_CMD_ALLTOALL && opType != HcclCMDType::HCCL_CMD_ALLGATHER_V &&
        opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        if (param.DataDes.dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_WARNING("[InplaceDataUnitSize] out of range[%d, %d]",
                HCCL_DATA_TYPE_INT8, static_cast<int>(HCCL_DATA_TYPE_RESERVED) - 1);
            return 0;
        }
        unitSize = SIZE_TABLE[param.DataDes.dataType];
    }
    return unitSize;
}
```

### IsHcclOpInplace

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L293)

```cpp
bool IsHcclOpInplace(const HcclCMDType &opType, const OpParam &param, u32 userRank, u32 userRankSize,
    u8 &isInplaceStatus)
{
    if (!IsInputOutPtrNotNullPtr(param, isInplaceStatus)) {
        return false;
    }
    u32 unitSize = InplaceDataUnitSize(opType, param);
    u64 inputDataSize = 0;
    u64 outputDataSize = 0;
    switch (opType) {
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
            isInplaceStatus = 0;
            return false;
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize * userRankSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            inputDataSize = param.DataDes.count * unitSize * userRankSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_GATHER:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_SCATTER:
            if (userRank == param.root) {
                inputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        default:
            return HcclOpInplaceDefaultCase(param, isInplaceStatus);
            break;
    }
    return IsInputOutputOverlap(param, inputDataSize, outputDataSize, isInplaceStatus);
}
```

### CheckUserInMemNotLargerThanCCLInMem

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L347)

```cpp
bool CheckUserInMemNotLargerThanCCLInMem(const HcclCMDType &opType, OpParam &param,
    u64 commInputSize, u32 userRankSize)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = 0;
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        dataSize = param.DataDes.count * unitSize * userRankSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        dataSize = param.DataDes.count * unitSize;
    }

    if (dataSize <= commInputSize) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] <= CCLInMem[%llu]", dataSize, commInputSize);
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] > CCLInMem[%llu]", dataSize, commInputSize);
    }
    return dataSize <= commInputSize;
}
```

### ExecutorOnlySupportDMAReduce

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L366)

```cpp
bool ExecutorOnlySupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceMeshSmallCountExecutor") || (algName == "ReduceScatterDeterExecutor");
}
```

### ExecutorCanSupportDMAReduce

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L371)

```cpp
bool ExecutorCanSupportDMAReduce(const std::string& algName)
{
    const std::set<std::string> executorCanSupportDMAReduceSet = {
        "AllReduceRingFor91093Executor",
        "AllReduceFastDoubleRingFor91093Executor",
        "AlignedAllReduceDoubleRingFor91093Executor",
        "ReduceScatterRingFor91093Executor",
        "ReduceScatterFastDoubleRingFor91093Executor",
        "AlignedReduceScatterDoubleRingFor91093Executor",
        "ReduceScatterPipelineFor91093Executor"
        };
    if (executorCanSupportDMAReduceSet.find(algName) != executorCanSupportDMAReduceSet.end()) {
        return true;
    }
    return false;
}
```

### ExecutorNoSupportDMAReduce

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L388)

```cpp
bool ExecutorNoSupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceComm") || (algName == "ReduceScatterComm");
}
```

### ExecutorSupportInPlace

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L393)

```cpp
bool ExecutorSupportInPlace(const OpParam &param, const std::string& algName, bool retryEnable,
    InplaceSupportRetryStatus &inPlaceSupportRetryStatus)
{
    (void) param;
    // case 2.2
    if (ExecutorOnlySupportDMAReduce(algName)) {
        if (retryEnable) {
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE1;
        return false;
    } else if (ExecutorNoSupportDMAReduce(algName)) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorNoSupportDMAReduce[%s] is allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::ALWAYS_NO_DMA_REDUCE;
        return true;
    } else if (ExecutorCanSupportDMAReduce(algName)) {
        if (retryEnable) {
            // 对应的executor会感应RetryEnable环境变量，走非DMA削减逻辑
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE2;
        return false;
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The unknown executor[%s] does not support "
            "for an inplace case yet.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::UNKONWN_EXECUTOR;
        return false;
    }
}
```

### CalGCD

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L466)

```cpp
u32 CalGCD(std::vector<u32> &nums)
{
    if (nums.size() == 0) {
        return 1;
    }
    std::sort(nums.begin(), nums.end(), [](const u32 &num1, const u32 &num2) {
        return num1 > num2;
    });

    u32 curGcd = nums[0];
    for (u32 i = 1; i < nums.size(); i++) {
        curGcd = CalGCD(curGcd, nums[i]);
    }
    HCCL_DEBUG("[CalGCD]size[%u], gcd[%u]", nums.size(), curGcd);
    return curGcd;
}
```

### CalGCD

[coll_alg_utils.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L483)

```cpp
u32 CalGCD(u32 a, u32 b)
{
    if (a == 0 || b == 0) {
        return 1;
    }

    u32 gcd = b;
    while (a % b != 0) {
        gcd = a % b;
        a = b;
        b = gcd;
    }
    HCCL_DEBUG("[CalGCD]a[%u] b[%u], gcd[%u]", a, b, gcd);
    return gcd;
}
```

## device_capacity.cc

### IsSupportAIVCopy

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L49)

```cpp
bool IsSupportAIVCopy(HcclDataType dataType)
{
    return (dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_UINT16 ||
        dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_UINT32 ||
        dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_UINT8 || dataType == HCCL_DATA_TYPE_BFP16 ||
        dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64 || dataType == HCCL_DATA_TYPE_FP64);
}
```

### IsSupportAIVReduce

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L57)

```cpp
bool IsSupportAIVReduce(HcclDataType dataType, HcclReduceOp op)
{
    bool checkDataType =
        (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT8 ||
         dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);

    return checkDataType && checkReduceType;
}
```

### IsAddressAlign

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L67)

```cpp
bool IsAddressAlign(const void *inputPtr, const void *outputPtr, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_910A) ==
                (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_910A);
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_910_93) ==
                (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_910_93);
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_310P1:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_310P) ==
                (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_310P);
        default:
            HCCL_WARNING("device type[%d] is out of range", static_cast<s32>(devType));
            return false;
    }
}
```

### IsDataTypeSupport

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L87)

```cpp
bool IsDataTypeSupport(HcclDataType dataType, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 ||
                    dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_INT16 ||
                    dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_BFP16);
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P1:
            return dataType == HCCL_DATA_TYPE_FP32;
        case DevType::DEV_TYPE_310P3:
            return (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT16 ||
                    dataType == HCCL_DATA_TYPE_FP16);
        default:
            HCCL_WARNING("device type[%d] is out of range", static_cast<s32>(devType));
            return false;
    }
}
```

### IsRedOpSupport

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L107)

```cpp
bool IsRedOpSupport(HcclReduceOp op, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_310P1:
            return op == HCCL_REDUCE_SUM;
        default:
            HCCL_WARNING("device type[%d] is out of range", static_cast<s32>(devType));
            return false;
    }
}
```

### IsSupportSDMAReduce

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L123)

```cpp
bool IsSupportSDMAReduce(const void *inputPtr, const void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    return IsAddressAlign(inputPtr, outputPtr, devType) && IsDataTypeSupport(dataType, devType) &&
        IsRedOpSupport(op, devType);
}
```

### IsSupportRDMAReduce

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L131)

```cpp
bool IsSupportRDMAReduce(HcclDataType dataType, HcclReduceOp op)
{
    bool checkDataType =
        (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT8 ||
        dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);
    bool isInfNanMode = IsOverFlowInfNanMode();
    return checkDataType && checkReduceType && isInfNanMode;
}
```

### GetBandWidthPerNPU

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L141)

```cpp
HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize, u32 deviceNumPerAggregation, float &bandWidth)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    // 处理 level=1、910B 的特殊条件
    if (level == 1 && (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910_93) &&
        userRankSize == deviceNumPerAggregation * 2) // 2: 910B 16p形态 单server场景
    {
        bandWidth = BANDWIDTH_PCIE_GEN5;
        return HCCL_SUCCESS;
    }
    // 其余情况查表
    static const std::map<std::pair<u32, DevType>, float> bwTable = {
        // level 0
        {{0, DevType::DEV_TYPE_310P3}, BANDWIDTH_PCIE_GEN3},
        {{0, DevType::DEV_TYPE_910},   BANDWIDTH_HCCS_910A},
        {{0, DevType::DEV_TYPE_910B},  BANDWIDTH_HCCS_910B},
        {{0, DevType::DEV_TYPE_910_93},BANDWIDTH_HCCS_910B},
        // level 1
        {{1, DevType::DEV_TYPE_910},   BANDWIDTH_RDMA_910A},
        {{1, DevType::DEV_TYPE_910B},  BANDWIDTH_RDMA_910B},
        {{1, DevType::DEV_TYPE_910_93},BANDWIDTH_RDMA_910B},
        // level 2
        {{2, DevType::DEV_TYPE_910_93},BANDWIDTH_HBM_910_93},
        // level 3
        {{3, DevType::DEV_TYPE_910_93},BANDWIDTH_SIO_910_93},
    };
    auto key = std::make_pair(level, devType);
    auto it  = bwTable.find(key);
    if (it != bwTable.end()) {
        bandWidth = it->second;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[Get][BandWidthPerNPU] Failed, deviceType[%d] Bandwidth Level[%u]", devType, level);
    return HCCL_E_NOT_SUPPORT;
}
```

### CheckDeviceType

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L178)

```cpp
HcclResult CheckDeviceType(const DevType deviceType)
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR("[Check][DeviceType]errNo[0x%016llx] device Type[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
```

### IsOverFlowInfNanMode

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L189)

```cpp
bool IsOverFlowInfNanMode()
{
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    HcclResult ret = hrtGetDeviceSatMode(&floatOverflowMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[impl][IsOverFlowInfNanMode] GetDeviceSatMode failed");
    }
    return (!GetExternalInputHcclDumpDebug()) && (floatOverflowMode == ACL_RT_OVERFLOW_MODE_INFNAN);
}
```

### IsUseSdidForDeviceId

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L204)

```cpp
bool IsUseSdidForDeviceId(const u32 superDeviceId)
{
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910_93 && superDeviceId != INVALID_UINT) {
        return true;
    }
    return false;
}
```

### IsSuperPodMode

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L214)

```cpp
HcclResult IsSuperPodMode(bool &useSuperPodMode)
{
    useSuperPodMode = false;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    s64 serverId = INVALID_SUPERPOD_SERVERID;
    if (devType == DevType::DEV_TYPE_910_93) {
        s32 deviceLogicId;
        HcclResult ret = hrtGetDevice(&deviceLogicId);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IsSuperPodMode]Get device id fail"), ret);
        CHK_RET(hrtGetDeviceInfo(deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SERVER_ID, serverId));
        useSuperPodMode = (serverId != INVALID_SUPERPOD_SERVERID);
        if (!useSuperPodMode) {
            HCCL_WARNING("If server Id is not configured, the network port may need to be " \
                "kept up to ensure normal service running, server id[%016llx]", serverId);
        }
    }
    HCCL_INFO("[IsSuperPodMode]ret[%d], devType[%d], serverId[%016llx]", useSuperPodMode, devType, serverId);
    return HCCL_SUCCESS;
}
```

### GetMaxDevNum

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L236)

```cpp
HcclResult GetMaxDevNum(u32& MaxDevNum)
{
    // 静态缓存最大设备数
    static u32 cachedMaxDevNum;
    static std::once_flag initFlag;  // 用于确保初始化只执行一次
    static std::mutex initMutex;     // 用于保护call_once调用

    // 使用mutex保护call_once调用
    std::lock_guard<std::mutex> lock(initMutex);

    // 使用call_once确保初始化逻辑只执行一次
    std::call_once(initFlag, [&]() {
        DevType devType;
        HcclResult result = hrtGetDeviceType(devType);
        if (result != HCCL_SUCCESS) {
            HCCL_ERROR("[GetMaxDevNum] [hrtGetDeviceType] get device type failed");
        }
        switch (devType) {
            case DevType::DEV_TYPE_310P3:
                cachedMaxDevNum = MAX_DEVICE_NUM_THIRTY_TWO;
                break;
            case DevType::DEV_TYPE_950:
                cachedMaxDevNum = MAX_DEVICE_NUM_SIXTY_FIVE;
                break;
            default:
                cachedMaxDevNum = MAX_DEVICE_NUM_SIXTEEN;
                break;
        }
    });

    // 直接读取缓存值
    MaxDevNum = cachedMaxDevNum;
    HCCL_DEBUG("[GetMaxDevNum] MaxDevNum[%u]", MaxDevNum);
    return HCCL_SUCCESS;
}
```

### IsSupportAicpuNormalQP

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L273)

```cpp
HcclResult IsSupportAicpuNormalQP(const u32& devicePhyId, bool &isSupportNormalQP)
{
    u32 aiQpCreateVersion = 0;
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, AI_QP_CREATE, &aiQpCreateVersion);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IsSupportAicpuNormalQP] ret[%d]"
        "devicePhyId[%u] not support" , ret, devicePhyId), HCCL_E_NETWORK);
    if (aiQpCreateVersion >= AI_NORMAL_QP_CREATE_VERSION) {
        isSupportNormalQP = true;
    } else {
        isSupportNormalQP = false;
    }
    HCCL_DEBUG("IsSupportAicpuNormalQP devicePhyId[%u], isSupportNormalQP[%d].", devicePhyId, isSupportNormalQP);
    return HCCL_SUCCESS;
}
```

### IsSupportAIVNormalQP

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L290)

```cpp
HcclResult IsSupportAIVNormalQP(const u32& devicePhyId, bool &isSupport)
{
    u32 version = 0;
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, AI_QP_CREATE_WITH_ATTRS, &version);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[hrtRaGetInterfaceVersion] ret[%d]"
        "devicePhyId[%u] not support" , ret, devicePhyId), HCCL_E_NETWORK);

    if (version >= AI_QP_CREATE_WITH_ATTRS_VERSION) {
        isSupport = true;
    } else {
        isSupport = false;
    }
    HCCL_INFO("IsSupportAIVNormalQP devicePhyId[%u], isSupport[%d].", devicePhyId, isSupport);
    return HCCL_SUCCESS;
}
```

### IsSupportRDMALite

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L308)

```cpp
bool IsSupportRDMALite(const s32 deviceLogicId)
{
    return NetworkManager::GetInstance(deviceLogicId).GetRdmaLiteStatus();
}
```

### IsSupportHccsAndSio

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L312)

```cpp
HcclResult IsSupportHccsAndSio(bool &flag)
{
    flag = false;
    size_t outputLen = 0;
    supportFeaturePara inputPara = { 0 };
    supportFeaturePara outputPara = { 0 };
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    inputPara.support_feature = CTRL_SUPPORT_SHMEM_MAP_EXBUS_MASK;
    inputPara.devid = static_cast<unsigned int>(deviceId);
    CHK_RET(hrtHalMemCtl(CTRL_TYPE_SUPPORT_FEATURE, &inputPara, sizeof(supportFeaturePara), &outputPara, &outputLen));

    if ((outputPara.support_feature & CTRL_SUPPORT_SHMEM_MAP_EXBUS_MASK) != 0) {
        flag = true;
    }
    HCCL_INFO("[IsSupportHccsAndSio] isSupportHccsAndSio %d", flag);
    return HCCL_SUCCESS;
}
```

### GetMemBlockNum

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L333)

```cpp
HcclResult GetMemBlockNum(const u32 devicePhyId, u32& memBlockNum)
{
#ifndef CCL_KERNEL_AICPU
    u32 info = 0;
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(hrtDrvGetPlatformInfo(&info));
    if (info == 0) { // 在device侧
        std::string chipName;
        if (hrtHalGetChipInfo(devicePhyId, chipName) == HCCL_SUCCESS) {
            if (chipName.find(SOC_NAME_910B) != std::string::npos) {
                // 共享内存池目前不支持动态扩容；910B场景需要的内存池较大，但是申请太大，会导致310P上内存不足，通过硬件区分。
                memBlockNum = MEM_BLOCK_NUM_BIGER;
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}
```

### GetNotifyMaxWaitTime

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L353)

```cpp
u32 GetNotifyMaxWaitTime()
{
    static bool init = false;
    static uint32_t notifyMaxWaitTime = NOTIFY_MAX_WAIT_TIME;
    if (UNLIKELY(!init)) {
        DevType deviceType;
        if (hrtGetDeviceType(deviceType) == HCCL_SUCCESS) {
            notifyMaxWaitTime = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?\
                NOTIFY_MAX_WAIT_TIME_910_93 : NOTIFY_MAX_WAIT_TIME;
            init = true;
        }
    }

    HCCL_INFO("[GetNotifyMaxWaitTime] notifyMaxWaitTime is %us", notifyMaxWaitTime);
    return notifyMaxWaitTime;
}
```

### IsSupportAtomicWrite

[device_capacity.cc](../../../../../src/legacy/ascend910/platform/common/device_capacity.cc#L370)

```cpp
HcclResult IsSupportAtomicWrite(DevType deviceType, u32 devicePhyId, bool& isSupportAtomicWrite)
{
    if (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) {
        u32 version = 0;
        HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, RA_RS_GET_ROCE_API, &version);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("%s call hrtRaGetInterfaceVersion ret[%d] devicePhyId[%u]",
            __func__, ret, devicePhyId), HCCL_E_NETWORK);
        isSupportAtomicWrite = (version >= RA_RS_ATOMIC_WRITE_VERSION);
        HCCL_INFO("%s deviceType[%d] devicePhyId[%u], version[%u], isSupportAtomicWrite[%d]",
            __func__, deviceType, devicePhyId, version, isSupportAtomicWrite);
    } else {
        isSupportAtomicWrite = false;
        HCCL_INFO("%s deviceType[%d] not support", __func__, deviceType);
    }
    return HCCL_SUCCESS;
}
```
