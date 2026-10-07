# Broadcast、Reduce、Scatter、Barrier 与点对点

符号见 [公共机制](01_common.md)。表按优先级阅读；默认配置首先经过 AlgConfigurator。这些算子没有因为公共基类包含成本函数，就自动按每次数据量重算 level1。

选定执行器后的 Broadcast NB binary、NHR oneshot、Reduce/Scatter 模板条件见 [内部模板切换](07_internal_templates.md)。

## Broadcast

[910B 选择](../../../../src/legacy/ascend910/algorithm/impl/operator/broadcast_operator.cc#L152)

| 顺序 | 条件 | 执行器 |
|---|---|---|
| 1 | A && S && O && Bi>=16 MiB && Bo>=16 MiB && Copy类型支持 && D<=4 GiB | BroadcastMeshAivExecutor |
| 2 | Mesh | BroadCastMeshExecutor |
| 3 | topoType=4P_RING | 名称为 BroadCast4pRingExecutor；本快照未找到同名注册，不能保证成功执行 |
| 4 | topoType=NP_SINGLE_RING 或 8P_RING | BroadCastRingExecutor |
| 5 | 其他 | BroadCastComm |

混合型号入口先于 910B：gcd 聚合组卡数>1 使用 BroadCastMixExecutor+NHR，否则 BroadCastComm+Ring。该选择器没有像 AG 那样在入口统一把 R=1 选为 Single。

注册核对发现 `BroadCast4pRingExecutor` 只有选择字符串，没有同名 REGISTER_EXEC。所核对的 A2 常规配置初始化没有产生 4P_RING：标卡默认 NP_SINGLE_RING，模块4卡默认4P_MESH，满足HCCS计数时改NP_MESH；null则RESERVED。因此该字符串不是已证明可达的正常A2候选，不据此断言正常Broadcast必失败。若其他入口注入4P_RING，还需追踪；GetAlgExec返回空时公共层会报错。[拓扑初始化](../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L334) · [拓扑映射](../../../../src/legacy/ascend910/algorithm/impl/alg_configurator.cc#L468)

切换发生在 AIV 开关、运行模式、S、CCL 容量、Copy 能力、4 GiB 阈值或拓扑变化时。这里没有按 D 的自动 level1 V2 调用；root 在 HD 情况影响 tag 的 root 分组标志，不能误称 root 改变了这张表的执行器。

代码事实：AIV 在 O、单组和缓冲区限制下可用。机制推断：Broadcast 是单 root 分发，拓扑决定机内组织与跨组转发；具体 root 分组影响递归 HD 的执行路径与资源复用。

## Reduce

[入口与 root 标签](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_operator.cc#L32) · [910B](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_operator.cc#L111)

| 顺序 | 条件 | 执行器 |
|---|---|---|
| 1 | R=1 | ReduceSingleExecutor，提前返回 |
| 2 | 混合型号 | 不支持 |
| 3 | A2，Mesh | ReduceMeshExecutor |
| 4 | A2，NP_SINGLE_RING | ReduceRingPlusHd |
| 5 | A2 其他拓扑 | ReduceComm |

910B 函数没有数据量阈值和 AIV 分支，也没有自动 level1 V2。`ReduceRingPlusHd` 内部还需看层次/模板，不能仅靠名字判定所有层均 HD。数据量变化可能改变切片和任务量，不一定改变执行器。

为什么：root 收集归约结果，机内组织和机间模板分开；硬件归约支持在执行器/模板进一步检查。OnlyAIV 对 Reduce 在公共包装层即拒绝。

## Scatter

[选择入口](../../../../src/legacy/ascend910/algorithm/impl/operator/scatter_operator.cc#L38)

先拒绝混合型号，再 R=1 → ScatterSingleExecutor。对于多 rank，**level1 不在 Ring/NB/NHR 时强制改 Ring**。即使配置了 HD，也不能据此预测最终用 HD。root 的 HD tag 标志在该修正之前生成，需要按原代码顺序理解。

| 顺序 | 条件 | 执行器 |
|---|---|---|
| 1 | multiModuleDiffDeviceNumMode 或 multiSuperPodDiffServerNumMode | ScatterCommExecutor |
| 2 | Mesh | ScatterMeshExecutor |
| 3 | NP_SINGLE_RING / 8P_RING / NP_DOUBLE_RING，A2 | ScatterRingExecutor |
| 4 | 其他 | ScatterCommExecutor |

NP_DOUBLE_RING 在共享判断中存在，不代表 A2 正常拓扑初始化必能生成它；初始化路径与手工字段构造必须分开。

为什么：源码明确限制支持的上层算法；非对称场景使用通用通信组织。没有硬件结果，不推断其“最快”。

## Send / Recv / BatchSendRecv

| 操作 | 有序条件 | 结果 |
|---|---|---|
| Send | 选择器无候选比较 | SendExecutor |
| Recv（类名 ReceiveOperator） | 选择器无候选比较 | ReceiveExecutor |
| BatchSendRecv | retryEnable && aicpuUnfoldMode | BatchSendRecvRetry |
| BatchSendRecv | 前项不满足且 isGroupMode | BatchSendRecvGroup |
| BatchSendRecv | 其余 | BatchSendRecv |

[Send](../../../../src/legacy/ascend910/algorithm/impl/operator/send_operator.cc#L26) · [Recv](../../../../src/legacy/ascend910/algorithm/impl/operator/receive_operator.cc#L26) · [Batch](../../../../src/legacy/ascend910/algorithm/impl/operator/batchsendrecv_operator.cc#L23)

Send/Recv 是点对点，没有必要套用 Ring/Mesh 集合通信选型。固定执行器不代表传输协议、任务分片与资源永远固定。Batch 的重执行能力优先于 Group；host 执行函数中还存在残留项目转 host 再选择路径，但注释明确关联 A3 BSR，不能计为已证明的 A2 切换路径，见验证页。

## Barrier

[实现](../../../../src/legacy/ascend910/framework/op_base/src/op_base_host.cc#L188)

传统 Barrier 调用 AllReduceOutPlace，使用 `UNLIMITED_TIMEWAITSYNCMODE`。AR 的 A2 选择器据此设置 isBarrierOp，排除 AIV；再按其他传统条件选择。不能因为普通小数据 AR 可用 AIV 就判断 Barrier 也可以。

## BatchWrite 和 Gather：避免遗漏与误归类

[BatchWrite 类](../../../../src/legacy/ascend910/algorithm/impl/operator/batch_write_operator.h#L19) 只继承基础类，没有覆写 SelectAlg；注册命令 BATCH_WRITE。基础 SelectAlg 不给出候选名；包装层 algName 为空时创建 SendExecutor 作为描述兜底。实际 MC2/特殊 BatchWrite 业务可能由专门接口与引擎路径执行，不能据此称“所有 BatchWrite 都等于 Send 算法”。本手册将其记录为特殊入口，不把它加入普通集合通信性能比较。

本版本传统 `REGISTER_OP` 清单没有 GATHER 专有选择器。配置解析器识别 `gather`、公共基类有 Gather 成本函数，都不能证明存在独立 Gather 算法选择链；复合 GatherAlltoAllV 接口也不等同普通 Gather。此处不虚构支持或候选。

执行器注册、基类与模板入口统一见 [源码索引](source_index.md)。
