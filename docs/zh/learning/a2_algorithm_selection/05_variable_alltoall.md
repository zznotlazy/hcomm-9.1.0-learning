# V 算子与 AlltoAll 系列

## AllGatherV

[910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_v_operator.cc#L107)

DV=`max(counts)*unitSize`；初始化 counts 数组和每 rank 总输出量不可混淆。混合型号入口报不支持。O、!S 先调用默认 level1 V2，使用 DV 和 floor(Bo/R)；若 Pipeline context>65535，改 NHR，**不是普通 AG 的 HD**。

AivV=`A && S && Copy支持 && DV<=256 MiB`。这里没有 Q 绕过 DV 上限，也没有 R=1 的 Single 提前分支。

| 顺序 | 条件 | AllGatherV 执行器后缀 |
|---|---|---|
| 1 | O && AivV && DV<=700 KiB | MeshAivSmallCountExecutor |
| 2 | O && AivV && DV>700 KiB | MeshAivExecutor |
| 3 | O && !AivV && level1=Pipeline && !S | MeshOpbasePipelineExecutor |
| 4 | O 其余 | MeshExecutor |
| 5 | G && S | MeshGraphExecutor |
| 6 | G && !S && K>1 && (DV>1 MiB 或 M<=4 或 level1=Pipeline) | MeshGraphPipelineExecutor |
| 7 | G 其余 | MeshExecutor |

主选择器没有先检查 Mesh topo 才进入这张表；是否能成功建立所需平面由后续资源/执行器决定。因此名字含 Mesh 不等于入口已经验证所有 Mesh 条件。

## ReduceScatterV

[910B](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_v_operator.cc#L121)

该函数没有普通 RS 那样的自动 level1 V2 调用；不要直接把 RS 的 32 KiB 自动 Pipeline 条件搬过来。

| 顺序 | 条件 | 结果 |
|---|---|---|
| 1 | G && !S && 组内卡数非对称 | 不支持 |
| 2 | G && !S && 组内卡数对称 | ReduceScatterVDeterExecutor；level1 不在 Ring/NB/NHR 时改 NHR，提前返回 |
| 3 | 非上项，det!=DISABLE && Mesh && 组内卡数对称 && type=FP16/FP32/BFP16 | ReduceScatterVDeterExecutor，提前返回 |
| 4 | 前面未返回，level1=Pipeline && context(RS)>65535 | level1改 NHR，继续 |
| 5 | O && level1=Pipeline && det=DISABLE && MultiMeshInlineReduce(CCL) | ReduceScatterVMeshOpbasePipelineExecutor，提前返回 |
| 6 | O && A && S && DV<=256 MiB && AIVReduce支持 && det=DISABLE | DV>2 MiB：ReduceScatterVAIVBigCountExecutor；否则 ReduceScatterVMeshAivSmallCountExecutor |
| 7 | O && SDMA支持(CCL) | ReduceScatterVMeshOpbaseExecutor；level1不在 Ring/NB/NHR 时改 NHR |
| 8 | G && SDMA支持(用户指针) | ReduceScatterVMeshExecutor |
| 9 | 其余 | 不支持：只支持 inline reduce |

严格保序模式在这个函数表现为 `det!=DISABLE` 浮点确定性分支，不能照抄普通 RS 的 OrderPreserved 名称。混合型号在上层入口拒绝。

## AlltoAll / AlltoAllV / AlltoAllVC 共享选择器

[主选择](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L238)

输入区别：AlltoAll 使用固定 sendCount；V 使用每 rank counts/displacements 和收集的地址信息；VC 使用 count 矩阵。三者共享类，构造基础算法属性时使用 `HCCL_CMD_ALLTOALL`，配置解析也把 AlltoAll 配置复制给 V/VC。

同型号 A2 的顺序如下；后项隐含前面未选中，且资源/参数处理没有报错。

| 顺序 | 条件 | 执行器 |
|---|---|---|
| 1 | 满足 A2 AIV 条件（下文展开） | 按 AIV 子选择，提前返回 |
| 2 | ifCompileForAiv && 前项未中 | 提前返回；没有传统候选，不能当作正常执行成功 |
| 3 | R=1 && O && !aicpuUnfoldMode | RunAlltoAllSingleExecutor |
| 4 | A2 DirectFullmesh 条件 | 对称内存条件满足选 RunAlltoAllFullMeshSymmetricMemory，否则 RunAlltoAllDirectFullmesh，提前返回 |
| 5 | 满足 ContinuousPipeline | RunAlltoAllVContinuousPipeline，提前返回 |
| 6 | 满足 TwoLevelPipeline | RunAlltoAllVTwoLevelPipeline |
| 7 | NA+pairwise 显式打平，或 isAllRankSamePlane，或 R<=2，或组内卡数非对称，或超节点 server 数非对称 | RunAlltoAllVFullMesh |
| 8 | 其余 | RunAlltoAllVStaged |

共享表中的 SatisfyIntraSuperPod 是 A3 条件，A2 不满足，不将其算作 A2 额外入口。

### AIV 全条件

[AIV 条件](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L550)

基础：A、Copy发送类型支持、R>1、!retryEnable，以及 `!O || Bi>=512 KiB*16`（即 8 MiB）。拓扑：Mesh 或 R==M。

模块：V/VC 要求 S；固定 AlltoAll 则支持 S 或跨组条件 `O && !组内卡数非对称 && M<=8 && 满足数据/中转容量`。

跨组固定 AlltoAll：令 `b=sendCount*sendTypeSize`、`t=b*R`；RoCE Direct 时 `t<=4 MiB` 且 `t+2 MiB<=Bi`；否则 `b<=190 KiB` 且 `t<=Bi`。这是候选条件，不是为 V/VC 自动求完整矩阵容量。[中转检查](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L626)

AIV 子选择：[SelectAlgforAiv](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L205)

- A2 固定 AlltoAll 且 !S：`O && supportRoceDirect && (!(serverNum==1 && M==2) || IntraRoceSwitch==1)` → AlltoAllDirectFullmeshAIVExecutor；否则 AlltoAllStagedAIVRdmaExecutor。
- 其余 A2 AIV → AlltoAllMeshAivExecutor。

OnlyAIV 失败时辅助函数可能返回 false 并走传统候选，但公共/最终检查会拒绝非 AIV；不能把 false 当成用户强制模式被静默接受。

### DirectFullmesh 全条件

[辅助判断](../../../../src/legacy/ascend910/algorithm/impl/coll_alg_utils.cc#L58)

A2 需要 O。

- V：`S || (R<=64 && serverNum<=8 && GetStreamCaptureInfo 查询为 capture)`；若单 server 跨 module、!S、R>2 且 IntraRoceSwitch==0，再禁用此路径。这里使用流查询结果，不是直接使用 param.isCapture。
- 固定 AlltoAll 和 VC：S，且对矩阵中**每一个 rank**，发送行总字节和接收列总字节都 **>Bi**。辅助函数任一 rank 的任一方向 `<=Bi` 就返回 false。函数名容易使人误以为“小数据放得下才直接传”，实际是此处的大数据条件。
- Direct 选中后，对称内存 + superPodNum==1 + (`serverNum==1 || !InterHccsDisable`) 才改为 SymmetricMemory 执行器。

### Pipeline 全条件

[Continuous](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L688) · [TwoLevel](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L518)

共同要求：A2、O、meshAggregationRankSize>1、R>meshAggregationRankSize 且整除、组内卡数对称、Bi>=80 MiB、level1 在 Ring/HD/Pipeline 中。

- Continuous：还要求 op=AlltoAllV、不是 NA+pairwise、!param.isCapture；**此函数没有 context<=65535 的检查**。
- TwoLevel：还有 context(AlltoAll)<=65535；不要求 op=V，不排除 NA+pairwise，也没有本函数级的 !capture 检查。不能把 Continuous 的限制复制给 TwoLevel。

NA+pairwise 在 TwoLevel 之后才被用于 FullMesh 判断，故不是无条件强制 FullMesh。需要先判断 AIV、Direct 和 Pipeline。

### BCopy/ZCopy 是另外一条选择

[CopyMode](../../../../src/legacy/ascend910/algorithm/impl/operator/alltoall_operator.cc#L125)

对需要收集 sendRecvInfo 的非提前返回路径：O 中所有 rank 的最大 `sendLength+sendOffset`、最大 `recvLength+recvOffset` 均<=Bi 时设 ZCopy，否则保持 BCopy；非 O 设 ZCopy。源码比较两方向都用输入容量 Bi，不能擅自改为 Bo。

CopyMode 影响 tag 和额外执行器信息。并不是所有名字包含 Direct/AIV 的提前返回路径都执行这段函数，也不能把局部的 ZCopy 标签等同于所有路径的零拷贝能力描述。

## 为什么 V 与 AlltoAll 需要独立分析

代码事实：V 的最大 count 决定局部分类，counts/displacements 决定总量和布局；AlltoAll 候选还依赖 count 矩阵、收集阶段、capture、RoCE Direct 和缓冲区容量。不同算法的前置检查不完全一致。

机制推断：两级/连续流水试图重叠机内与机间阶段，Staged 借中转组织流量，Direct 避免部分中转流程；不规则消息布局和 capture 的资源约束会影响能否采用这些实现。没有设备实验时不能量化收益。

AGV/RSV 的单组与跨组模板映射见 [内部模板切换](07_internal_templates.md)。实际模板与 AIV 内核入口见 [源码索引](source_index.md)，需带着上表已命中的条件进入，检查特定阶段是否执行。
