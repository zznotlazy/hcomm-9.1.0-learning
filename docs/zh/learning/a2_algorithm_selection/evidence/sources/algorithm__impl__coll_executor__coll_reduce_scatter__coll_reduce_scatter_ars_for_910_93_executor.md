# coll_reduce_scatter_ars_for_910_93_executor.cc

共享执行器函数；按设备条件确认A2分支。

### CollReduceScatterARSFor91093Executor::CalcLevel1CommInfo

[coll_reduce_scatter_ars_for_910_93_executor.cc](../../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ars_for_910_93_executor.cc#L44)

```cpp
HcclResult CollReduceScatterARSFor91093Executor::CalcLevel1CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        HCCL_DEBUG("[CalcARSInterCommInfo] use ring comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    } else if(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        HCCL_DEBUG("[CalcARSInterCommInfo] use NB comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_NONUNIFORM_BRUCK);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    } else {
        HCCL_DEBUG("[CalcARSInterCommInfo] use NHR comm type");
        CommParaInfo commARSInter(COMM_LEVEL1_LOGICAL, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
        CHK_RET(CalcCommPlaneInfo(tag_, commARSInter, opTransport[COMM_LEVEL1_LOGICAL], inputType, outputType));
    }
    return HCCL_SUCCESS;
}
```
