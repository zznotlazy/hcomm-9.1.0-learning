# send 选择证据

按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。

### SendOperator::SelectAlg

[send_operator.cc](../../../../../src/legacy/ascend910/algorithm/impl/operator/send_operator.cc#L26)

```cpp
HcclResult SendOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    algName = "SendExecutor";
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else {
        newTag = tag + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] Send newTag is [%s]", newTag.c_str());
    return HCCL_SUCCESS;
}
```
