# 源码证据索引

自动从当前快照生成。主文档负责解释A2可达路径；本索引保留有序条件的完整函数。执行器和模板表是源码出现清单，不自动证明全部分支在A2可达。

## 注册命令清单

| 命令 | 选择器 | 注册位置 |
|---|---|---|
| HCCL_CMD_ALLGATHER | AllGatherOperator | [all_gather_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L427) |
| HCCL_CMD_ALLGATHER_V | AllGatherVOperator | [all_gather_v_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_v_operator.cc#L178) |
| HCCL_CMD_ALLREDUCE | AllReduceOperator | [all_reduce_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L743) |
| HCCL_CMD_ALLTOALLV | AlltoAllOperator | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L761) |
| HCCL_CMD_ALLTOALL | AlltoAllOperator | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L762) |
| HCCL_CMD_ALLTOALLVC | AlltoAllOperator | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L763) |
| HCCL_CMD_BATCH_WRITE | BatchWriteOperator | [batch_write_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/batch_write_operator.cc#L15) |
| HCCL_CMD_BATCH_SEND_RECV | BatchSendRecvOperator | [batchsendrecv_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/batchsendrecv_operator.cc#L40) |
| HCCL_CMD_BROADCAST | BroadCastOperator | [broadcast_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/broadcast_operator.cc#L249) |
| HCCL_CMD_RECEIVE | ReceiveOperator | [receive_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/receive_operator.cc#L40) |
| HCCL_CMD_REDUCE | ReduceOperator | [reduce_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_operator.cc#L161) |
| HCCL_CMD_REDUCE_SCATTER | ReduceScatterOperator | [reduce_scatter_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L585) |
| HCCL_CMD_REDUCE_SCATTER_V | ReduceScatterVOperator | [reduce_scatter_v_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_v_operator.cc#L235) |
| HCCL_CMD_SCATTER | ScatterOperator | [scatter_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/scatter_operator.cc#L108) |
| HCCL_CMD_SEND | SendOperator | [send_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/send_operator.cc#L40) |

## 公共条件原文

[公共函数证据](evidence/common.md)


## all_gather

[完整选择函数](evidence/all_gather.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| AllGatherARSFor91093Executor | [CollAllGatherARSFor91093Executor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ars_for_910_93_executor.cc#L83) | CollAllGatherRingFor91093Executor → CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherARSFor91093Executor.md) |
| AllGatherAivRdmaExecutor | [CollAllGatherAivRdmaExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_aiv_rdma_executor.cc#L283) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherAivRdmaExecutor.md) |
| AllGatherComm | [CollAllGatherCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_comm_executor.cc#L151) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherComm.md) |
| AllGatherMeshAivExecutor | [CollAllGatherMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_aiv_executor.cc#L230) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshAivExecutor.md) |
| AllGatherMeshAivSmallCountExecutor | [CollAllGatherMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_aiv_smallcount_executor.cc#L220) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshAivSmallCountExecutor.md) |
| AllGatherMeshExecutor | [CollAllGatherMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L223) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshExecutor.md) |
| AllGatherMeshGraphExecutor | [CollAllGatherMeshGraphExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_graph_executor.cc#L262) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshGraphExecutor.md) |
| AllGatherMeshGraphPipelineExecutor | [CollAllGatherMeshGraphPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_graph_pipeline_executor.cc#L104) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshGraphPipelineExecutor.md) |
| AllGatherMeshOpbaseExecutor | [CollAllGatherMeshOpbaseExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_executor.cc#L128) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshOpbaseExecutor.md) |
| AllGatherMeshOpbasePipelineExecutor | [CollAllGatherMeshOpbasePipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_opbase_pipeline_executor.cc#L134) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherMeshOpbasePipelineExecutor.md) |
| AllGatherRingExecutor | [CollAllGatherRingExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_ring_executor.cc#L235) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherRingExecutor.md) |
| AllGatherSingleExecutor | [CollAllGatherSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_single_rank_executor.cc#L43) | CollAllGatherExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherSingleExecutor.md) |

## all_gather_v

[完整选择函数](evidence/all_gather_v.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| AllGatherVMeshAivExecutor | [AllGatherVMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_executor.cc#L152) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshAivExecutor.md) |
| AllGatherVMeshAivSmallCountExecutor | [CollAllGatherVMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gatherv_mesh_aiv_smallcount_executor.cc#L149) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshAivSmallCountExecutor.md) |
| AllGatherVMeshExecutor | [CollAllGatherVMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_executor.cc#L383) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshExecutor.md) |
| AllGatherVMeshGraphExecutor | [CollAllGatherVMeshGraphExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_graph_executor.cc#L117) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshGraphExecutor.md) |
| AllGatherVMeshGraphPipelineExecutor | [CollAllGatherVMeshGraphPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_graph_pipeline_executor.cc#L105) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshGraphPipelineExecutor.md) |
| AllGatherVMeshOpbasePipelineExecutor | [CollAllGatherVMeshOpbasePipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_opbase_pipeline_executor.cc#L178) | CollAllGatherVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllGatherVMeshOpbasePipelineExecutor.md) |

## all_reduce

[完整选择函数](evidence/all_reduce.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| AllReduceARSFor91093Executor | [CollAllReduceARSFor91093Executor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ars_for_910_93_executor.cc#L86) | CollAllReduceRingFor91093Executor → CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceARSFor91093Executor.md) |
| AllReduceAivDeterExecutor | [CollAllReduceAivDeterExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_aiv_deter_executor.cc#L163) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceAivDeterExecutor.md) |
| AllReduceAivDeterSmallExecutor | [CollAllReduceAivDeterSmallExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_aiv_deter_small_executor.cc#L172) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceAivDeterSmallExecutor.md) |
| AllReduceComm | [CollAllReduceCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_comm_executor.cc#L143) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceComm.md) |
| AllReduceDeterPipelineExecutor | [CollAllReduceDeterPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_deter_pipeline_executor.cc#L184) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceDeterPipelineExecutor.md) |
| AllReduceMeshAivExecutor | [CollAllReduceMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_aiv_executor.cc#L239) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshAivExecutor.md) |
| AllReduceMeshAivSmallCountExecutor | [CollAllReduceMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_aiv_smallcount_executor.cc#L217) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshAivSmallCountExecutor.md) |
| AllReduceMeshExecutor | [CollAllReduceMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_executor.cc#L274) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshExecutor.md) |
| AllReduceMeshGraphPipelineExecutor | [CollAllReduceMeshGraphPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_graph_pipeline_executor.cc#L100) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshGraphPipelineExecutor.md) |
| AllReduceMeshMidCountLoopExecutor | [CollAllReduceMeshMidCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_mid_count_executor.cc#L117) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshMidCountLoopExecutor.md) |
| AllReduceMeshOneshotLoopExecutor | [CollAllReduceMeshOneshotExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_oneshot_executor.cc#L125) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshOneshotLoopExecutor.md) |
| AllReduceMeshOpbaseLoopExecutor | [CollAllReduceMeshOpbaseExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_executor.cc#L127) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshOpbaseLoopExecutor.md) |
| AllReduceMeshOpbaseMidCountDeterministicExecutor | [CollAllReduceMeshOpbaseMidCountDeterministicExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_mid_count_deterministic_executor.cc#L287) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshOpbaseMidCountDeterministicExecutor.md) |
| AllReduceMeshOpbasePipelineExecutor | [CollAllReduceMeshOpbasePipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_pipeline_executor.cc#L143) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshOpbasePipelineExecutor.md) |
| AllReduceMeshOpbaseSmallCountDeterministicExecutor | [CollAllReduceMeshOpbaseSmallCountDeterministicExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_opbase_small_count_deterministic_executor.cc#L252) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshOpbaseSmallCountDeterministicExecutor.md) |
| AllReduceMeshSmallCountExecutor | [CollAllReduceMeshSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_small_count_executor.cc#L255) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMeshSmallCountExecutor.md) |
| AllReduceMidCountAivRdmaExecutor | [CollAllReduceMidCountAivRdmaExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mid_count_aiv_rdma_executor.cc#L256) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceMidCountAivRdmaExecutor.md) |
| AllReduceOrderPreservedExecutor | [CollAllReduceOrderPreservedExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_order_preserved_executor.cc#L372) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceOrderPreservedExecutor.md) |
| AllReduceRingExecutor | [CollAllReduceRingExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_ring_executor.cc#L279) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceRingExecutor.md) |
| AllReduceSingleExecutor | [CollAllReduceSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_single_rank_executor.cc#L47) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceSingleExecutor.md) |
| AllReduceSmallCountAivRdmaExecutor | [CollAllReduceSmallCountAivRdmaExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_small_count_aiv_rdma_executor.cc#L282) | CollAllReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AllReduceSmallCountAivRdmaExecutor.md) |

## alltoall

[完整选择函数](evidence/alltoall.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| AlltoAllDirectFullmeshAIVExecutor | [CollAlltoAllDirectFullmeshAIVExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_direct_fullmesh_aiv_executor.cc#L186) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AlltoAllDirectFullmeshAIVExecutor.md) |
| AlltoAllMeshAivExecutor | [CollAlltoAllMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_mesh_aiv_executor.cc#L296) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AlltoAllMeshAivExecutor.md) |
| AlltoAllMeshAivFor91093Executor | [CollAlltoAllMeshAivFor91093Executor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_mesh_aiv_for_910_93_executor.cc#L253) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AlltoAllMeshAivFor91093Executor.md) |
| AlltoAllMeshAivSmallCountExecutor | [CollAlltoAllMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_mesh_aiv_smallcount_executor.cc#L216) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AlltoAllMeshAivSmallCountExecutor.md) |
| AlltoAllStagedAIVRdmaExecutor | [CollRunAlltoAllStagedAivRdmaExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_staged_aiv_rdma_executor.cc#L247) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_AlltoAllStagedAIVRdmaExecutor.md) |
| RunAlltoAllDirectFullmesh | [CollRunAlltoAllDirectFullmesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_direct_fullmesh_executor.cc#L431) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllDirectFullmesh.md) |
| RunAlltoAllFullMeshSymmetricMemory | [CollRunAlltoAllFullMeshSymmetricMemory](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_symmetric_memory_executor.cc#L234) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllFullMeshSymmetricMemory.md) |
| RunAlltoAllSingleExecutor | [CollAlltoAllSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_single_rank_executor.cc#L78) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllSingleExecutor.md) |
| RunAlltoAllVContinuousPipeline | [CollAlltoAllVContinuousPipeline](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_continuous_pipeline_executor.cc#L215) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVContinuousPipeline.md) |
| RunAlltoAllVFor310PExecutor | [CollRunAlltoAllVFor310PExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/310P/coll_all_to_all_v_for_310p_executor.cc#L120) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVFor310PExecutor.md) |
| RunAlltoAllVFullMesh | [CollRunAlltoAllVFullMesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_fullmesh_executor.cc#L157) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVFullMesh.md) |
| RunAlltoAllVPipelineFor91093 | [CollAlltoAllVPipelineFor91093](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_pipeline_for_910_93_executor.cc#L253) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVPipelineFor91093.md) |
| RunAlltoAllVStaged | [CollRunAlltoAllVStaged](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_staged_executor.cc#L458) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVStaged.md) |
| RunAlltoAllVTwoLevelPipeline | [CollRunAlltoAllVTwoLevelPipeline](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_v_2level_pipeline_excecutor.cc#L182) | CollAlltoAllExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_RunAlltoAllVTwoLevelPipeline.md) |

## batch_write

[完整选择函数](evidence/batch_write.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|

## batchsendrecv

[完整选择函数](evidence/batchsendrecv.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| BatchSendRecv | [CollBatchSendRecvExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_executor.cc#L530) | CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BatchSendRecv.md) |
| BatchSendRecvGroup | [CollBatchSendRecvGroupExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_group_executor.cc#L776) | CollBatchSendRecvExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BatchSendRecvGroup.md) |
| BatchSendRecvRetry | [CollBatchSendRecvRetryExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_batch_send_recv_retry_executor.cc#L277) | CollBatchSendRecvExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BatchSendRecvRetry.md) |

## broadcast

[完整选择函数](evidence/broadcast.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| BroadCast4pRingExecutor | 未解析 | — | — |
| BroadCastComm | [CollBroadcastCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_comm_executor.cc#L142) | CollBroadcastExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BroadCastComm.md) |
| BroadCastMeshExecutor | [CollBroadcastMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L252) | CollBroadcastExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BroadCastMeshExecutor.md) |
| BroadCastRingExecutor | [CollBroadcastRingExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_ring_executor.cc#L253) | CollBroadcastExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BroadCastRingExecutor.md) |
| BroadcastMeshAivExecutor | [CollBroadcastMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_aiv_executor.cc#L163) | CollBroadcastExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_BroadcastMeshAivExecutor.md) |

## receive

[完整选择函数](evidence/receive.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| ReceiveExecutor | [CollReceiveExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_receive_executor.cc#L180) | CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReceiveExecutor.md) |

## reduce

[完整选择函数](evidence/reduce.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| ReduceComm | [CollReduceCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_comm_executor.cc#L98) | CollReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceComm.md) |
| ReduceMeshExecutor | [CollReduceMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_mesh_executor.cc#L192) | CollReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceMeshExecutor.md) |
| ReduceRingPlusHd | [CollReduceRingPlusHdExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_ring_plus_hd_executor.cc#L171) | CollReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceRingPlusHd.md) |
| ReduceSingleExecutor | [CollReduceSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_single_rank_executor.cc#L45) | CollReduceExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceSingleExecutor.md) |

## reduce_scatter

[完整选择函数](evidence/reduce_scatter.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| ReduceScatterARSFor91093Executor | [CollReduceScatterARSFor91093Executor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ars_for_910_93_executor.cc#L81) | CollReduceScatterRingFor91093Executor → CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterARSFor91093Executor.md) |
| ReduceScatterAivDeterExecutor | [CollReduceScatterAivDeterExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_aiv_deter_executor.cc#L206) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterAivDeterExecutor.md) |
| ReduceScatterAivDeterSmallExecutor | [CollReduceScatterAivDeterSmallExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_aiv_deter_small_executor.cc#L213) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterAivDeterSmallExecutor.md) |
| ReduceScatterAivRdmaExecutor | [CollReduceScatterAivRdmaExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_aiv_rdma_executor.cc#L265) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterAivRdmaExecutor.md) |
| ReduceScatterComm | [CollReduceScatterCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_comm_executor.cc#L207) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterComm.md) |
| ReduceScatterDeterExecutor | [CollReduceScatterDeterExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_executor.cc#L207) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterDeterExecutor.md) |
| ReduceScatterDeterPipelineExecutor | [CollReduceScatterDeterPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_deter_pipeline_executor.cc#L230) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterDeterPipelineExecutor.md) |
| ReduceScatterMeshAivExecutor | [CollReduceScatterMeshAivExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_executor.cc#L237) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshAivExecutor.md) |
| ReduceScatterMeshAivSmallCountExecutor | [CollReduceScatterMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_aiv_smallcount_executor.cc#L229) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshAivSmallCountExecutor.md) |
| ReduceScatterMeshDmaEliminationExecutor | [CollReduceScatterMeshDmaEliminationExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_dma_elimination.cc#L137) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshDmaEliminationExecutor.md) |
| ReduceScatterMeshExecutor | [CollReduceScatterMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_executor.cc#L297) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshExecutor.md) |
| ReduceScatterMeshGraphExecutor | [CollReduceScatterMeshGraphExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_graph_executor.cc#L246) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshGraphExecutor.md) |
| ReduceScatterMeshGraphPipelineExecutor | [CollReduceScatterMeshGraphPipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_graph_pipeline_executor.cc#L118) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshGraphPipelineExecutor.md) |
| ReduceScatterMeshOpbasePipelineExecutor | [CollReduceScatterMeshOpbasePipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_pipeline_executor.cc#L196) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshOpbasePipelineExecutor.md) |
| ReduceScatterMeshOpbaseSmallCountDeterministicExecutor | [CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_opbase_small_count_deterministic_executor.cc#L300) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterMeshOpbaseSmallCountDeterministicExecutor.md) |
| ReduceScatterOrderPreservedExecutor | [CollReduceScatterOrderPreservedExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_order_preserved_executor.cc#L275) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterOrderPreservedExecutor.md) |
| ReduceScatterRingExecutor | [CollReduceScatterRingExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_ring_executor.cc#L390) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterRingExecutor.md) |
| ReduceScatterSingleExecutor | [CollReduceScatterSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_single_rank_executor.cc#L50) | CollReduceScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterSingleExecutor.md) |

## reduce_scatter_v

[完整选择函数](evidence/reduce_scatter_v.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| ReduceScatterVAIVBigCountExecutor | [CollReduceScatterVAIVBigCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_aiv_big_count_executor.cc#L166) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVAIVBigCountExecutor.md) |
| ReduceScatterVDeterExecutor | [CollReduceScatterVDeterExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_deter_executor.cc#L359) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVDeterExecutor.md) |
| ReduceScatterVMeshAivSmallCountExecutor | [CollReduceScatterVMeshAivSmallCountExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_aiv_smallcount_executor.cc#L150) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVMeshAivSmallCountExecutor.md) |
| ReduceScatterVMeshExecutor | [CollReduceScatterVMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_executor.cc#L127) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVMeshExecutor.md) |
| ReduceScatterVMeshOpbaseExecutor | [CollReduceScatterVMeshOpbaseExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L353) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVMeshOpbaseExecutor.md) |
| ReduceScatterVMeshOpbasePipelineExecutor | [CollReduceScatterVMeshOpbasePipelineExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/Coll_reduce_scatter_v_mesh_opbase_pipeline_executor.cc#L183) | CollReduceScatterVExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ReduceScatterVMeshOpbasePipelineExecutor.md) |

## scatter

[完整选择函数](evidence/scatter.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| ScatterCommExecutor | [CollScatterCommExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_comm_executor.cc#L81) | CollScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ScatterCommExecutor.md) |
| ScatterMeshExecutor | [CollScatterMeshExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_mesh_executor.cc#L112) | CollScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ScatterMeshExecutor.md) |
| ScatterRingExecutor | [CollScatterRingExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_ring_executor.cc#L136) | CollScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ScatterRingExecutor.md) |
| ScatterRingFor91093Executor | [CollScatterRingFor91093Executor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_ring_for_910_93_executor.cc#L294) | CollScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ScatterRingFor91093Executor.md) |
| ScatterSingleExecutor | [CollScatterSingleRankExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_single_rank_executor.cc#L34) | CollScatterExecutor → CollCommExecutor → CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_ScatterSingleExecutor.md) |

## send

[完整选择函数](evidence/send.md)

| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |
|---|---|---|---|
| SendExecutor | [CollSendExecutor](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive/coll_send_executor.cc#L191) | CollNativeExecutorBase → CollExecutorBase | [函数与模板](evidence/executor_SendExecutor.md) |

## AIV 内核入口

AIV不一定使用传统模板注册；按设备宏和已选择执行器进入以下分派入口。

- [aiv_all_gather_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_gather/aiv_all_gather_op.h#L1)
- [aiv_all_gather_v_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_gather_v/aiv_all_gather_v_op.h#L1)
- [aiv_all_reduce_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_reduce/aiv_all_reduce_op.h#L1)
- [aiv_all_to_all_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/aiv_all_to_all_op.h#L1)
- [aiv_all_to_all_v_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/aiv_all_to_all_v_op.h#L1)
- [aiv_all_to_all_vc_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/all_to_all/aiv_all_to_all_vc_op.h#L1)
- [aiv_broadcast_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/broadcast/aiv_broadcast_op.h#L1)
- [aiv_reduce_scatter_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/reduce_scatter/aiv_reduce_scatter_op.h#L1)
- [aiv_reduce_scatter_v_op.h](../../../../src/legacy/ascend910/algorithm/base/alg_aiv_template/reduce_scatter_v/aiv_reduce_scatter_v_op.h#L1)

## 阈值定义

| 名称 | 定义 | 位置 |
|---|---|---|
| `ATTR_POS_AIV_COMM_BUFFER` | `0x00` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L25) |
| `ATTR_POS_AIV_COMM_INFO_BUFFER` | `0x01` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L26) |
| `AIV_COMM_BUFFER_BITMASK` | `0x01` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L27) |
| `AIV_COMM_INFO_BUFFER_BITMASK` | `0x02` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L28) |
| `AIV_ALL_REDUCE_BIG_SIZE` | `16 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L30) |
| `AIV_ALL_REDUCE_A3_ENTRY_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L31) |
| `AIV_ALL_REDUCE_A3_GRAPH_ENTRY_SIZE` | `4 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L32) |
| `AIV_REDUCE_SCATTER_DETER_SMALL_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L33) |
| `AIV_REDUCE_SCATTER_BIG_SIZE` | `190 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L34) |
| `AIV_REDUCE_SCATTER_MID_SIZE` | `2 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L35) |
| `AIV_REDUCE_SCATTER_SMALL_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L36) |
| `AIV_REDUCE_SCATTER_A3_ENTRY_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L37) |
| `AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE` | `4 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L38) |
| `AIV_ALL_GATHER_BIG_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L39) |
| `AIV_ALL_GATHER_SMALL_SIZE` | `700 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L40) |
| `AIV_ALL_GATHER_A3_ENTRY_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L41) |
| `AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE` | `4 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L42) |
| `AIV_ALL_TO_ALL_BIG_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L43) |
| `AIV_ALL_TO_ALL_A3_ENTRY_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L44) |
| `AIV_BIG_SIZE` | `256 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L45) |
| `AIV_ALL_REDUCE_DETER_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L46) |
| `AIV_REDUCE_SCATTER_RANK_SIZE_8` | `8` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L47) |
| `AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE` | `190 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L49) |
| `AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE` | `760 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L50) |
| `AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE` | `760 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L51) |
| `AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE` | `760 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L52) |
| `AIV_REDUCE_SCATTER_A3_SMALL_RANKSIZE_ENTRY_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L54) |
| `AIV_REDUCE_SCATTER_A3_MID_RANKSIZE_ENTRY_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L55) |
| `AIV_REDUCE_SCATTER_A3_LARGE_RANKSIZE_ENTRY_SIZE` | `128 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L56) |
| `AIV_ALL_GATHER_A3_SMALL_RANKSIZE_ENTRY_SIZE` | `1 * 1024 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L58) |
| `AIV_ALL_GATHER_A3_MID_RANKSIZE_ENTRY_SIZE` | `512 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L59) |
| `AIV_ALL_GATHER_A3_LARGE_RANKSIZE_ENTRY_SIZE` | `32 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L60) |
| `AIV_A3_CROSSNODE_TINY_SIZE` | `28 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L62) |
| `AIV_A3_CROSSNODE_SMALL_SIZE` | `112 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L63) |
| `AIV_A3_CROSSNODE_MID_SIZE` | `448 * 1024` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L64) |
| `MAX_RANK_SIZE` | `16` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L66) |
| `MAX_RANK_SIZE_A3` | `768` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L67) |
| `MAX_RANK_SIZE_RDMA` | `64` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L68) |
| `AIV_A2_ALL_REDUCE_RDMA_KERNEL_NUM` | `2` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L86) |
| `AIV_TIMEOUT_DEFAULT` | `1091` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L89) |
| `AIV_TIMEOUT_DEFAULT_US` | `1091 * TIME_S_TO_US` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L90) |
| `AIV_TIMEOUT_MAX` | `1091` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L91) |
| `AIV_TIMEOUT_MAX_US` | `1091 * TIME_S_TO_US` | [hccl_aiv.h](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L92) |
| `HCCL_SMALL_COUNT_128_KB` | `128 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L177) |
| `HCCL_SMALL_COUNT_GRAPH_64_KB` | `64 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L178) |
| `HCCL_MEDIUM_COUNT_GRAPH_4_MB` | `4 * 1024 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L179) |
| `HCCL_SMALL_COUNT_256_KB` | `256 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L180) |
| `HCCL_ALIGN_COUNT_32_B` | `32` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L182) |
| `HCCL_SMALL_COUNT_190_KB` | `190 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L183) |
| `HCCL_SMALL_COUNT_512_KB` | `512 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L184) |
| `HCCL_SMALL_COUNT_1_MB` | `1024 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L185) |
| `HCCL_SMALL_COUNT_2_MB` | `2048 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L186) |
| `HCCL_SMALL_COUNT_4_MB` | `4096 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L187) |
| `HCCL_SMALL_COUNT_8_MB` | `8192 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L188) |
| `HCCL_SMALL_COUNT_16_KB` | `16 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L189) |
| `HCCL_MID_COUNT_16_MB` | `16 * 1024 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L190) |
| `HCCL_MID_COUNT_32_MB` | `32 * 1024 * 1024` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L191) |
| `HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM` | `8` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L193) |
| `HCCL_ALLTOALLV_P2P_SIZE` | `2` | [common.h](../../../../src/legacy/ascend910/algorithm/pub_inc/common.h#L195) |
| `NHR_ALLREDUCE_SMALL_SIZE` | `256 * 1024` | [op_context.h](../../../../src/legacy/ascend910/algorithm/pub_inc/op_context.h#L18) |
| `NHR_BCAST_SMALL_SIZE` | `2 * 1024 * 1024` | [op_context.h](../../../../src/legacy/ascend910/algorithm/pub_inc/op_context.h#L19) |
| `PIPELINE_DEPTH` | `3` | [op_context.h](../../../../src/legacy/ascend910/algorithm/pub_inc/op_context.h#L24) |
| `HCCL_FFTS_CAPACITY` | `65535` | [hccl_common.h](../../../../src/legacy/ascend910/pub_inc/hccl_common.h#L88) |
| `MAX_RING_PIPLINE_SERVER_NUM` | `128` | [coll_alg_utils.h](../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.h#L24) |
| `MAX_ALLTOALLV_DIRECT_FULLMESH_RANKSIZE` | `64` | [coll_alg_utils.h](../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.h#L30) |
| `MAX_ALLTOALLV_DIRECT_FULLMESH_SERVER_NUM` | `8` | [coll_alg_utils.h](../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.h#L31) |
| `TWO_RANK_SIZE` | `2` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L35) |
| `HCCL_MIN_PIPLINE_SLICE_ALIGN` | `512` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L39) |
| `HCCL_MIN_SLICE_ALIGN_910B` | `16384` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L40) |
| `HCCL_MIN_SLICE_ALIGN_910_93` | `16384` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L41) |
| `HCCL_MIN_SLICE_ALIGN_ONCHIP` | `512` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L42) |
| `HCCL_MIN_SLICE_ALIGN` | `128` | [alg_template_base_pub.h](../../../../src/legacy/ascend910/algorithm/base/alg_template/alg_template_base_pub.h#L43) |
| `NHR_FACTOR_TWO` | `2.0` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L26) |
| `NHR_FACTOR_THREE` | `3.0` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L27) |
| `NHR_FACTOR_FOUR` | `4.0` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L28) |
| `NHR_SUB_TWO` | `2.0` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L29) |
| `PIPELINE_MIN_SIZE` | `32 * 1024` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L31) |
| `PIPELINE_ALLREDUCE_MIN_SIZE` | `1024 * 1024` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L32) |
| `PIPELINE_MIN_SIZE_NO_LITE` | `2 * 1024 * 1024` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L33) |
| `HCCL_FFTS_CAPACITY` | `65535` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L34) |
| `SERVER_COUNT_THRESHOLD_FOR_MULTI_DETER_PIPELINE` | `2` | [coll_alg_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L37) |
| `ALLTOALL_PIPELINE_MIN_CCL_SIZE` | `80 * 1024 * 1024` | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L24) |
| `MAX_RMDA_RANK_SIZE` | `8` | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L25) |
| `MAX_310P_RANK_SIZE` | `4` | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L26) |
| `AIV_FLAG_OFFSET` | `2 * 1024 * 1024` | [alltoall_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L27) |
| `AIV_MAX_DATASIZE` | `4ULL * 1024 * 1024 * 1024` | [broadcast_operator.cc](../../../../src/legacy/ascend910/algorithm/impl/operator/broadcast_operator.cc#L20) |

## 已有测试入口（未执行）

| 测试名 | 源码 |
|---|---|
| CollAlgOperatorTest.is_2u2p_infer | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L93) |
| CollAlgOperatorTest.need_create_single_mesh_plane | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L121) |
| CollAlgOperatorTest.ut_SupportRetryWithInplaceCheck_ptr_check | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L148) |
| CollAlgOperatorTest.ut_SupportRetryWithInplaceCheck_opType_check | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L200) |
| CollAlgOperatorTest.ut_SupportRetryWithInplaceCheck_retry_condition_check | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L348) |
| CollAlgOperatorTest.ut_HcclCommunicator_IsHcclOpInplace | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L498) |
| CollAlgOperatorTest.ut_HcclCommunicator_IsHcclOpInplace_alltoall | [ut_coll_alg_operator.cc](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L555) |
| AllGatherCommonTest.ut_hccl_impl_610_8rank_1server_all_gather_char | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L246) |
| AllGatherCommonTest.ut_hccl_impl_610_8rank_1server_all_gather_float | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L403) |
| AllGatherCommonTest.ut_hccl_impl_610_5rank_1server_all_gather_char | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L567) |
| AllGatherCommonTest.ut_hccl_impl_610_4rank_1server_all_gather_char | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L729) |
| AllGatherCommonTest.ut_hccl_impl_610_2rank_1server_all_gather_char | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L891) |
| AllGatherCommonTest.ut_hccl_impl_610_3rank_1server_all_gather_float | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L1054) |
| AllGatherCommonTest.ut_hccl_impl_610_2rank_1server_all_gather_float | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L1217) |
| AllGatherCommonTest.ut_hccl_impl_610_8rank_1server_all_gather_memsampler_char | [ut_allgather_common.cc](../../../../test/ut/inter/comm/ut_allgather_common.cc#L1380) |
| HcclImplTest.ut_SelectAlg_when_broadcast_910C_Expect_ReturnIs_BroadcastMeshAivExecutor | [ut_hccl_impl_new.cc](../../../../test/ut/impl/ut_hccl_impl_new.cc#L309) |

## 索引限制

候选来自人工限定的选择函数；主入口保留非A2分派，Mix分支另见主文档。继承链索引用于跟踪公共模板，模板出现表含共享函数内其他芯片分支，不能代替可达性判断。没有在这里运行C++预处理、生产选择器、CANN测试或性能测量。


## 未解析注册项

BroadCast4pRingExecutor：源码选择器存在这个字符串，但当前传统注册表未找到相同名称。不能宣称可成功执行；主文档记录正常A2拓扑来源与此异常候选。

