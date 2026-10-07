# BroadCastMeshExecutor

注册：[coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L252)；类 `CollBroadcastMeshExecutor`。

共享代码可能含其他芯片分支。模板清单为出现位置；按完整函数中的设备/层次/子域条件确认实际执行。

## coll_broadcast_mesh_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_broadcast__coll_broadcast_mesh_executor.md)

## coll_broadcast_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_broadcast__coll_broadcast_executor.md)

## coll_comm_executor.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_comm_executor.md)

## coll_native_executor_base.cc

[完整函数与内部条件](sources/algorithm__impl__coll_executor__coll_native_executor_base.md)

## 模板注册与出现位置

| 模板 | 首个出现 | 实现注册 |
|---|---|---|
| TEMPLATE_ALL_GATHER_MESH | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L130) | [AllGatherMesh](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_mesh.cc#L221) |
| TEMPLATE_ALL_GATHER_MESH_ATOMIC | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L127) | [AllGatherMeshAtomic](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_mesh_atomic.cc#L72) |
| TEMPLATE_ALL_GATHER_NB | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L590) | [AllGatherNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_nb.cc#L214) |
| TEMPLATE_ALL_GATHER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L280) | [AllGatherRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_ring.cc#L379) |
| TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L273) | [AllGatherRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather/all_gather_ring_concurrent_direct.cc#L377) |
| TEMPLATE_ALL_REDUCE_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L73) | [AllReduceRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce/all_reduce_ring.cc#L226) |
| TEMPLATE_BROADCAST_NB | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L114) | [BroadcastNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nb.cc#L167) |
| TEMPLATE_BROADCAST_NB_BINARY | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L111) | [BroadcastNBBinary](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nb_binary.cc#L195) |
| TEMPLATE_BROADCAST_NHR | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L97) | [BroadcastNHR](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nhr.cc#L165) |
| TEMPLATE_BROADCAST_NHR_ONESHOT | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L94) | [BroadcastNHROneshot](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nhr_oneshot.cc#L196) |
| TEMPLATE_BROADCAST_NHR_V1 | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L102) | [BroadcastNHRV1](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nhr_v1.cc#L267) |
| TEMPLATE_BROADCAST_RECURSIVE_HD | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L119) | [BcastRecursiveHalvingDoubling](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/bcast_recursive_halvingdoubling.cc#L420) |
| TEMPLATE_GATHER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L977) | [GatherRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_gather/gather_ring.cc#L227) |
| TEMPLATE_MULTI_ROOT_SCATTER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1368) | [MultiRootScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/multi_root_scatter_ring.cc#L347) |
| TEMPLATE_REDUCESCATTER_HDSTAGE | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1471) | [ReduceScatterHDStage](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_hd_stage.cc#L256) |
| TEMPLATE_REDUCESCATTER_MESH | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1538) | [ReduceScatterMesh](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh.cc#L204) |
| TEMPLATE_REDUCESCATTER_MESH_ATOMIC | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1480) | [ReduceScatterMeshAtomic](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic.cc#L208) |
| TEMPLATE_REDUCESCATTER_MESH_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1475) | [ReduceScatterMeshDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_atomic_opbase.cc#L161) |
| TEMPLATE_REDUCESCATTER_MESH_MIX_SS | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1534) | [ReduceScatterMeshMixSingleStream](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_mesh_mix_single_stream.cc#L197) |
| TEMPLATE_REDUCESCATTER_NB | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L1297) | [ReduceScatterNB](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_nb.cc#L359) |
| TEMPLATE_REDUCESCATTER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L861) | [ReduceScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring.cc#L501) |
| TEMPLATE_REDUCESCATTER_RING_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L854) | [ReduceScatterRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter/reduce_scatter_ring_concurrent_direct.cc#L588) |
| TEMPLATE_SCATTER_MESH | [coll_broadcast_mesh_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L73) | [ScatterMesh](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_mesh.cc#L163) |
| TEMPLATE_SCATTER_RING | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2048) | [ScatterRing](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring.cc#L443) |
| TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2056) | [ScatterRingConcurrentDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring_concurrent_direct.cc#L343) |
| TEMPLATE_SCATTER_RING_DIRECT | [coll_comm_executor.cc](../../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_comm_executor.cc#L2064) | [ScatterRingDirect](../../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_scatter/scatter_ring_direct.cc#L238) |
