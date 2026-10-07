# AllReduce 与 ReduceScatter

使用 [公共页](01_common.md) 的符号、SDMA/RDMA/AIV 能力定义。每张表按顺序读取，后项隐含先前条件未命中；各链最后都要通过 OnlyAIV 和执行器注册检查。

选定执行器后的 NHR oneshot 与归约模板条件见 [内部模板切换](07_internal_templates.md)。

## AllReduce 的 A2 主链

[入口](../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L112) · [910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L294)

R=1 提前选 Single。混合型号在 A2 分派前处理：严格浮点保序不支持；gcd 聚合组卡数>1 用 Mix+NHR，否则 Comm+Ring。

定义 `d=floor(D/K)`；P2 为服务器数是正的 2 的幂。

```text
SmallRDMA = !S && !组内卡数非对称 && P2 && (d<=190 KiB || Q)
MidRDMA   = !S && !组内卡数非对称 && D<=16 MiB
AivDeter  = S && det==ENABLE && D<=8 MiB
Buf       = !O || (Bi>=16 MiB && Bo>=16 MiB)
AivAR = A && 非Barrier && AIVReduce支持 && L0Mesh && Buf
        && (S || SmallRDMA || MidRDMA) && (det==DISABLE || AivDeter)
```

先在 O 路径调用 level1 自动选择（传 D、Bi、用户指针 SDMA 能力、RDMA 能力、AivAR），再处理 AHC，再对 O 的 Pipeline 进行 context>65535 → HD 回退。STRICT + 非对称组内卡数报不支持。

| 优先级 | 条件 | 执行器（AllReduce 前缀） |
|---|---|---|
| 1 | det=STRICT 且 type 为 FP16/FP32/BFP16 | OrderPreservedExecutor；优先于 AIV |
| 2 | AivAR && AivDeter | AivDeterSmallExecutor，直接返回 |
| 3 | AivAR && SmallRDMA | SmallCountAivRdmaExecutor |
| 4 | AivAR && MidRDMA | MidCountAivRdmaExecutor |
| 5 | AivAR && (!O 或 D>=16 MiB) | MeshAivExecutor |
| 6 | AivAR 剩余 | MeshAivSmallCountExecutor |
| 7 | 非 AIV，K<=2 | 按下面的双卡/单卡聚合组子表 |
| 8 | 非 AIV，K>2，Mesh | CCL 指针特例，再非确定性/确定性子表；为空回退 MeshExecutor |
| 9 | 非 AIV，K>2，非 Mesh | Comm |

代码内 AivDeter 的大数据 `else` 写了 AivDeterExecutor，但前置已限制 D<=8 MiB，故该大数据分支在本版本 A2 主链不可达。RS 的类似分支因 Q 例外则可以到达。不能仅搜字符串就称二者均可选。

K<=2 子表：

| 顺序 | 条件 | 后缀 |
|---|---|---|
| 1 | 用户输入输出分别等于 CCL 输入输出，O，Mesh | MeshExecutor |
| 2 | SingleMeshInlineReduce | O：D<=256 KiB SmallCount，否则 OpbaseLoop；非 O：D<=64 KiB SmallCount，否则 MeshExecutor |
| 3 | Is2U2PInfer | O 且 SDMA 支持：MeshOneshotLoopExecutor，否则 RingExecutor |
| 4 | det=DISABLE、level1=Pipeline、MultiMeshInlineReduce | O：MeshOpbasePipeline，否则 MeshGraphPipeline |
| 5 | Mesh / NP_SINGLE_RING / 其他 | MeshExecutor / RingExecutor / Comm |

**可达性提醒：** Is2U2PInfer 描述 PCIe 双卡，S 通常为 false，所以不会自动变成 SingleMeshInlineReduce；应按实际拓扑解析结果判定。

K>2、Mesh 先检查 CCL 指针特例（同上），再进入下列子选择：

| 模式 | 条件（按顺序） | 后缀 |
|---|---|---|
| det=DISABLE | O、MultiMeshInlineReduce、level1=Pipeline | MeshOpbasePipeline |
| det=DISABLE | O、SingleMeshInlineReduce | D<=256 KiB：MeshSmallCount，否则 MeshOpbaseLoop |
| det=DISABLE | G、MultiMeshInlineReduce、level1=Pipeline | MeshGraphPipeline |
| det=DISABLE | 尚未选择、O、!S、SDMA支持、D<512 KiB | MeshOpbaseSmallCountDeterministic |
| det!=DISABLE | O、level1=Pipeline、K>2、D>=K*HCCL_MIN_SLICE_ALIGN | DeterPipeline |
| det!=DISABLE | SingleMeshInlineReduce | 按 GetCountTypeForDeterAllReduce 分 Small/Mid/OneshotLoop |
| det!=DISABLE | 非上项、O、!S、SDMA支持 | D<=512 KiB：OpbaseSmallCountDeterministic，否则 OpbaseMidCountDeterministic |
| 任意 | 子选择结果为空 | MeshExecutor |

[子选择](../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L523) · [数据分类](../../../../src/legacy/ascend910/algorithm/impl/operator/all_reduce_operator.cc#L33)

确定性单组分类：G 下 <=64 KiB 为 Small；其后 <=4 MiB 且 K=8 为 Medium；其余 Huge。非 G 下 <=128 KiB 为 Small；其余 K=8 为 Medium，K!=8 为 Huge。Medium 不代表 O 中有固定上限。

## ReduceScatter 的 A2 主链

[入口](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L42) · [910B](../../../../src/legacy/ascend910/algorithm/impl/operator/reduce_scatter_operator.cc#L164)

D 表示每 rank 接收字节，输入总体规模通常 D*R。R=1 → Single；混合入口同样有 Mix/Comm 和严格保序限制。

STRICT 先检查卡数非对称（报错）；浮点 FP16/FP32/BFP16 选 OrderPreserved 并立即返回。整数不会因为 STRICT 直接选这个执行器。

```text
CrossRS = !S && !组内卡数非对称 && Mesh
          && (Q || ((P2 || D<=128 KiB) && D*R<=16 MiB
                    && O && Bi>=16 MiB && Bo>=16 MiB && D<=256 KiB))
AivDeter = S && serverNum==1 && det==ENABLE && (D*R<=8 MiB || Q)
AivRS = A && AIVReduce支持 && (det==DISABLE || AivDeter)
        && ((S && (D<=256 MiB || Q)) || CrossRS)
```

注意 RS 的 Q 放在跨组大括号内部，能够跳过此处部分 O/缓冲区条件；AG 的缓冲区检查在外部。这是代码差异，不扩展为硬件保证，后续资源/执行支持仍需验证。

| 顺序 | 条件 | ReduceScatter 执行器后缀 |
|---|---|---|
| 1 | AivRS && AivDeter | D*R<=8 MiB：AivDeterSmall，否则 AivDeter（需要 Q 放宽） |
| 2 | AivRS && CrossRS | AivRdma |
| 3 | AivRS && O && D<=2 MiB | MeshAivSmallCount |
| 4 | AivRS 其余 | MeshAiv |

AIV 选中后提前返回，不再走后面的自动 level1。否则 O 自动选择，输入 D、floor(Bi/R)、CCL 指针 SDMA 能力和 RDMA 能力；再 AHC，再 Pipeline 容量超限改 HD。

Mesh 传统链：

| 顺序 | 条件 | 后缀 |
|---|---|---|
| 1 | det=ENABLE、O、level1=Pipeline、K>2 | DeterPipeline |
| 2 | O、SingleMeshInlineReduce(CCL) | det!=DISABLE：Deter；否则 MeshDmaElimination |
| 3 | O、det=DISABLE、level1=Pipeline、MultiMeshInlineReduce(CCL) | MeshOpbasePipeline |
| 4 | O、!S、SDMA支持(CCL)、((D<=512 KiB 且 det=ENABLE) 或 D*R<512 KiB) | MeshOpbaseSmallCountDeterministic |
| 5 | 非 O、SingleMeshInlineReduce(用户指针)、det!=DISABLE、K>2 | Deter |
| 6 | 非 O、SingleMeshInlineReduce(用户指针) 其余 | G且D<=1 MiB：MeshGraph，否则 Mesh |
| 7 | 前面为空；G、M>1、K>1、det=DISABLE、MultiMeshInlineReduce(用户指针)、(D>1 MiB 或 M<=4 或 level1=Pipeline) | MeshGraphPipeline |
| 8 | 仍为空，O 或 D>1 MiB | Mesh |
| 9 | 仍为空，G | MeshGraph |

非 Mesh：NP_SINGLE_RING → Ring；其他 → Comm。上述后缀实际都带 Executor，源代码索引保留完整注册名。

## 为什么这些归约算子条件更多

代码事实：归约硬件类型/操作支持、Inf/NaN 模式和确定性要求直接限制算法。STRICT 保序优先，不能用吞吐优化覆盖。Pipeline context 还要计入确定性本地归约任务。

机制推断：不同并行归约顺序会影响浮点舍入结果；保序和确定性实现因此需要约束任务顺序。小数据更敏感于调度/同步开销，大数据更依赖带宽和分片；但 SmallCountDeterministic 的名字不能推出“只在开启确定性时选中”，AR/RS 的非确定性路径也能用它。

传统执行器内部的模板分派、基类循环和 AIV 内核入口见 [源码索引](source_index.md)。这些模板候选需带着算子的已命中条件阅读，不能把执行器文件内 A3 专属分支计入 A2 候选。
