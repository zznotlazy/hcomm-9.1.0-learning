# batchsendrecv 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### BatchSendRecvOperator::SelectAlg

[batchsendrecv_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/batchsendrecv_operator.cc#L23)

```cpp
HcclResult BatchSendRecvOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    if (retryEnable_ && param.aicpuUnfoldMode) {
        algName = "BatchSendRecvRetry";
    } else if (param.isGroupMode) {
        algName = "BatchSendRecvGroup";
    }
    else {
        algName = "BatchSendRecv";
    }
    newTag = tag;
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[BatchSendRecvOperator][SelectAlg] algName %s newTag %s", algName.c_str(), newTag.c_str());
    return HCCL_SUCCESS;
}
```
