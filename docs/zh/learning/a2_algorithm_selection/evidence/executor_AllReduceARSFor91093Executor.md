# AllReduceARSFor91093Executor

注册：[coll_all_reduce_ars_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ars_for_910_93_executor.cc#L86)；类 `CollAllReduceARSFor91093Executor`。

共享代码可能含其他芯片分支。模板清单为出现位置；按完整函数中的设备/层次/子域条件确认实际执行。

## coll_all_reduce_ars_for_910_93_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_all_reduce__coll_all_reduce_ars_for_910_93_executor.md)

## coll_all_reduce_ring_for_910_93_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_all_reduce__coll_all_reduce_ring_for_910_93_executor.md)

## coll_all_reduce_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_all_reduce__coll_all_reduce_executor.md)

## coll_comm_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_comm_executor.md)

## coll_native_executor_base.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_native_executor_base.md)

## 模板注册与出现位置

| 模板 | 首个出现 | 实现注册 |
|---|---|---|
| TEMPLATE_ALL_GATHER_NB | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L461) | [AllGatherNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_nb.cc#L214) |
| TEMPLATE_ALL_GATHER_NHR | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L465) | [AllGatherNHR](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_nhr.cc#L280) |
| TEMPLATE_ALL_GATHER_RING | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L457) | [AllGatherRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_ring.cc#L379) |
| TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L273) | [AllGatherRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_ring_concurrent_direct.cc#L377) |
| TEMPLATE_ALL_REDUCE_AHC | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L310) | [AllReduceAHC](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_ahc.cc#L69) |
| TEMPLATE_ALL_REDUCE_AHC_BROKE | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L313) | [AllReduceAHCBroke](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_ahc_broke.cc#L80) |
| TEMPLATE_ALL_REDUCE_NB | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L320) | [AllReduceNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_nb.cc#L245) |
| TEMPLATE_ALL_REDUCE_NHR | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L333) | [AllReduceNHR](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr.cc#L197) |
| TEMPLATE_ALL_REDUCE_NHR_ONESHOT | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L330) | [AllReduceNHROneshot](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr_oneshot.cc#L116) |
| TEMPLATE_ALL_REDUCE_NHR_V1 | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L299) | [AllReduceNHRV1](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_nhr_v1.cc#L316) |
| TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L434) | [AllReduceRecursiveHalvingDoubling](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_recursive_hd.cc#L445) |
| TEMPLATE_ALL_REDUCE_RING | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L293) | [AllReduceRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_ring.cc#L226) |
| TEMPLATE_GATHER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L977) | [GatherRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_gather/gather_ring.cc#L227) |
| TEMPLATE_MULTI_ROOT_SCATTER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1368) | [MultiRootScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/multi_root_scatter_ring.cc#L347) |
| TEMPLATE_REDUCESCATTER_HDSTAGE | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1471) | [ReduceScatterHDStage](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_hd_stage.cc#L256) |
| TEMPLATE_REDUCESCATTER_MESH | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1538) | [ReduceScatterMesh](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh.cc#L204) |
| TEMPLATE_REDUCESCATTER_MESH_ATOMIC | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1480) | [ReduceScatterMeshAtomic](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic.cc#L208) |
| TEMPLATE_REDUCESCATTER_MESH_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1475) | [ReduceScatterMeshDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic_opbase.cc#L161) |
| TEMPLATE_REDUCESCATTER_MESH_MIX_SS | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1534) | [ReduceScatterMeshMixSingleStream](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_mix_single_stream.cc#L197) |
| TEMPLATE_REDUCESCATTER_NB | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L379) | [ReduceScatterNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nb.cc#L359) |
| TEMPLATE_REDUCESCATTER_NHR | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L385) | [ReduceScatterNHR](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nhr.cc#L565) |
| TEMPLATE_REDUCESCATTER_RING | [coll_all_reduce_ring_for_910_93_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_for_910_93_executor.cc#L373) | [ReduceScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring.cc#L501) |
| TEMPLATE_REDUCESCATTER_RING_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L854) | [ReduceScatterRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring_concurrent_direct.cc#L588) |
| TEMPLATE_SCATTER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2048) | [ScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring.cc#L443) |
| TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2056) | [ScatterRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring_concurrent_direct.cc#L343) |
| TEMPLATE_SCATTER_RING_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2064) | [ScatterRingDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring_direct.cc#L238) |
