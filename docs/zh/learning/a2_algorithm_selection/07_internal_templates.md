# 执行器内部还会怎样切换算法

执行器名字、`algType_.algoLevel1` 和实际模板是三个不同对象。前几页回答选哪个执行器；本页继续追到 `KernelRun → GetAlgTemplate → RunTemplate`。这里只把核对过的 A2 常用执行器分支展开，其他候选的完整函数与模板注册入口见 [证据索引](source_index.md)，不能把本页映射推广到同名算法的所有执行器。

符号沿用 [公共机制](01_common.md)。`C = execMem.count * unitSize` 是当前执行轮的数据大小；它不必等于用户总数据量 D。`x=C/K` 在以下代码中是整数除法。机间实际通信域大小记为 `r`，与模块数 M 的概念应分别核对。

## 常见分层模板映射

下表的短名省略 `TEMPLATE_` 前缀；每行只对列出的执行器成立。没有对应层通信或其规模为 1 时，模板可能不执行，具体看函数的门禁。

| 执行器 / 层 | 有序条件 | 模板结果 |
|---|---|---|
| AllGatherMesh / level1 | Ring，或 isDiffDeviceModule && serverNum==1 | ALL_GATHER_RING |
| 同上 | NHR / NHRv1 / NB | ALL_GATHER_NHR / NHR_V1 / NB |
| 同上 | 其他 | ALL_GATHER_RECURSIVE_HALVING_DOUBLING |
| AllGatherComm / 合并域 | NHR / NHRv1 / NB / HD | 对应模板 |
| 同上 | 其他 | ALL_GATHER_RING |
| AllReduceMesh / level1 | Ring | ALL_REDUCE_RING |
| 同上 | NHR && x<=256 KiB | ALL_REDUCE_NHR_ONESHOT |
| 同上 | NHR && x>256 KiB | ALL_REDUCE_NHR |
| 同上 | NHRv1 / NB | ALL_REDUCE_NHR_V1 / NB |
| 同上 | 其他 | ALL_REDUCE_RECURSIVE_HALVING_DOUBLING |
| AllReduceComm / 合并域 | NHR | 比较 C<=256 KiB，选择 NHR_ONESHOT 或 NHR；这里不除 K |
| ReduceScatterMesh / level1 | Ring / NHR / NHRv1 / NB | 对应 REDUCESCATTER 模板 |
| 同上 | 其他 | REDUCESCATTER_RECURSIVE_HD |
| BroadCastMesh / level1 | NHR && x<=2 MiB | BROADCAST_NHR_ONESHOT |
| 同上 | NHR && x>2 MiB | BROADCAST_NHR |
| 同上 | NHRv1 | BROADCAST_NHR_V1 |
| 同上 | NB | 按下一节判断 BROADCAST_NB_BINARY 或 BROADCAST_NB |
| 同上 | 其他，包含 Ring | BROADCAST_RECURSIVE_HD |
| BroadCastComm / 合并域 | NHR / NB | 使用 C，不除 K，套相应小数据判断 |
| 同上 | NHRv1 / 其他 | BROADCAST_NHR_V1 / BROADCAST_RING |

源码：[AG Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L72) · [AG Comm](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_comm_executor.cc#L69) · [AR Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_executor.cc#L89) · [AR Comm](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_comm_executor.cc#L83) · [RS Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/coll_reduce_scatter_mesh_executor.cc#L130) · [Broadcast Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_mesh_executor.cc#L59) · [Broadcast Comm](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_broadcast/coll_broadcast_comm_executor.cc#L77)

一个容易误判的例子：设置 level1=Ring，不代表 `BroadCastMesh` 会创建 BROADCAST_RING。该函数没有 Ring 分支，会进入递归 HD 的 else。相反，`AllGatherComm` 的兜底是 Ring。必须读最终执行器的模板分支。

256 KiB 和 2 MiB 的注释分别说明“小数据不切片”。这是明确代码事实。减少切片调度成本是机制解释；阈值是否对特定机器最优，需要测量。[阈值定义](../../../../src/legacy/ascend910/algorithm/pub_inc/op_context.h#L18)

## Broadcast：NB binary 的完整条件

[ShouldUseBinaryBroadcastOfNB](../../../../src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast/broadcast_nb_binary.cc#L156)

1. `r==2` 时直接返回 true，与数据量无关。
2. 否则 `s=ceil(log2(r))`，按源码的移位循环计算。
3. `B=GetBandWidthPerNPU(1,R,K)` 的 GB/s 数值，`base=B*60*1000`；此函数直接使用这个数值公式，别替换为公共成本函数中的 `1024^3` 转换。
4. `r<=4` 时阈值 `T=r*base`；`r>4` 时 `T=((2*s-2)/(s-2))*base`。**括号内按 C++ unsigned 整数除法截断**，再乘 float。例如 s=5，系数为 2，不是 8/3。
5. `dataSize<T` 选 binary；等于阈值选普通 NB。Mesh 传 dataSize=C/K，Comm 传 C。

这是静态 60 us 和带宽估计模型，没有读取当次通信延迟或实时链路吞吐。r=1 时不能脱离调用层门禁单独使用这个公式。带宽获取失败的错误路径也不是算法调优结果。

## Reduce 与 Scatter 的模板并不服从通用映射

| 路径 | 条件与结果 |
|---|---|
| ReduceMesh 的 level0 归约 | detDisable && dtype!=INT64 && A2 && reduceOp!=PROD：REDUCESCATTER_MESH_ATOMIC；否则多流 Mesh ReduceScatter |
| ReduceMesh 的 level1 | 域大小>1：Ring→REDUCE_RING，其他→REDUCE_RECURSIVE_HALVING_DOUBLING；域大小=1走本地拷贝 |
| ReduceMesh 的最终收集 | root 所在 level0 组执行 GATHER_MESH |
| ReduceComm，A2 | 合并通信域使用 REDUCE_RING，不能根据配置推成 NHR/NB |
| Scatter 的 level1 | 域大小<=1或subRoot!=userRank先返回；否则 NB→SCATTER_NB，NHR→SCATTER_NHR，其他→SCATTER_RING |
| ScatterMesh 的 level0 | SCATTER_MESH，之后拷贝到用户输出 |

源码：[Reduce Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_mesh_executor.cc#L75) · [Reduce Comm](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce/coll_reduce_comm_executor.cc#L61) · [Scatter level1](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_executor.cc#L197) · [Scatter Mesh](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter/coll_scatter_mesh_executor.cc#L39)

以上 INT64/PROD/确定性的条件是该层的原始判断，不是完整的设备支持清单。归约类型、地址和硬件能力还有前置与模板内检查。执行器名字不变也可能从 atomic 改为非 atomic，或改变通信轮次。

## V 算子的单组/跨组路径

| 执行器 | 条件 | 执行顺序 / 模板 |
|---|---|---|
| AllGatherVMesh | M>1 | 先level0 MESH_ATOMIC，再level1：Ring或单server异构模块→RING，NB→NB，其他→NHR |
| 同上 | M<=1 | RunSingleMesh，MESH_DIRECT |
| ReduceScatterVMeshOpbase | M>1 | 先level1：Ring→RING，NB→NB，其他→NHR；再level0 MESH_ATOMIC |
| 同上 | M<=1 | level0 MESH_DIRECT |

源码：[AGV level1](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_executor.cc#L174) · [AGV KernelRun](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/coll_all_gather_v_mesh_executor.cc#L278) · [RSV level1](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L289) · [RSV KernelRun](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/coll_reduce_scatter_v_mesh_opbase_executor.cc#L338)

AGV/RSV 这里的 else 是 NHR，不是固定长度 AG/RS 常见的 HD。V 算子的 counts/displs 还改变每轮分配：例如 RSV 单组分别给每个 rank 取 min(remaining,maxTotalCount)；跨组在剩余总预算内，反复均分给仍有数据的 rank。因此 `max(counts)*unitSize` 用于外层选型，不能直接替代模板里当前切片大小。

## 怎样追到“为什么切换”的最后一层

对其他执行器，在证据索引打开“函数与模板”，沿 `Orchestrate/RunLoop → KernelRun → GetAlgTemplate/ExecuteKernelLaunch` 阅读。核对以下信息：

- 本次循环的 count/counts 是怎样从用户输入与 CCL 容量生成的。
- 当前层是否真的有通信；root 是否属于需要执行的子域。
- 该执行器支持哪些 level1 值，else 到底是哪一个模板。
- AIV 是否进入独立内核分派，模板注册表不能覆盖它。
- 模板内部是否根据 rank 数、链路能力、reduceAttr、切片大小再走不同实现。

“选定执行器以后不再切换”是不准确的；“内部切换都是运行时性能自适应”同样没有源码证据。本页的切换是对参数、能力和预设模型的确定性分支。
