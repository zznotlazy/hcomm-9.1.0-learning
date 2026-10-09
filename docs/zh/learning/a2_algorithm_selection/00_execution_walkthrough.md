# 按执行顺序读懂算法选择：从一次 AllGather 开始

你已经能大致看懂 C++，接下来要练的是：**调用进入哪个函数，在哪里修改变量，返回后接着执行哪一行。** 先沿着本文走通一次调用，再查学习手册中的条件表。暂时不要求记住所有算法名称。

本文依据仓库中的 hcomm 9.1.0 源码，讲同型号 A2/910B 的传统 host 路径。主例假定：单算子模式（`OP_BASE`）、多 rank、非 capture、非 AI CPU 展开、没有命中提前缓存路径、调用成功。它是静态走查，没有运行 NPU 通信。图模式、AIV、零拷贝和缓存会改变路线，后面分别指出。

## 1. 先知道这次调用想做什么

AllGather 的含义是：每个参与者提供一份数据，最后每个参与者都得到大家的数据。例如两个 rank：

```text
rank 0 输入 [10, 11]    rank 1 输入 [20, 21]
rank 0 输出 [10, 11, 20, 21]
rank 1 输出 [10, 11, 20, 21]
```

`rank` 是通信参与者的编号，不直接等同于服务器编号。`count` 是**每个 rank 输入的元素个数**；在上例是 2，不是最终输出的 4。算法选择中的 `dataSize = count × 每个元素字节数` 也是每 rank 的输入大小。

所有参与者要完成同一个操作，库需要决定“谁来组织通信”和“内部如何通信”。这就是选择代码的任务。

## 2. 三个名字，分三次决定

| 源码中的东西 | 现在只需这样理解 | 例子 |
| --- | --- | --- |
| `algName` | 负责组织本次操作的执行器名称，是字符串 | `AllGatherMeshExecutor` |
| `algType_.algoLevel1` | 分层通信中 level1 使用的策略字段 | `ALG_LEVEL1_RING` |
| `level1TempAlg` | 执行时按策略创建的具体模板对象 | Ring 模板对象 |

“选中 MeshExecutor”和“level1 使用 Ring”可以同时成立。前者组织整个过程，后者决定其中一个阶段怎样通信。**一个名字不能代替这三层结果。** Mesh、Ring 的内部通信原理可以稍后读；现在先关注谁决定谁。

## 3. 总路线：调用向下走，返回向上走

```text
host 的 ExecOp
  → 创建 AllGatherOperator（读取已有配置和拓扑）
  → 公共 SelectAlg（6 个参数）
      → 转接 SelectAlg（5 个参数）
          → AllGatherOperator::SelectAlg（4 个参数）
              → SelectAlgfor910B：修改 algName，必要时修改 level1
              ← 回到 4 参数函数：生成 newTag
          ← 回到公共 SelectAlg：按 algName 创建执行器，取得描述
  ← 回到 ExecOp：准备资源
  → operator 的 Orchestrate
      → executor 的 Orchestrate
          → RunLoop（本例）
              → 派生执行器的 KernelRun
                  → 创建模板 → Prepare → RunTemplate
```

箭头向下表示调用，向上表示被调用函数结束后回到调用者。`return` 只退出**当前函数**。例如 910B 函数提前返回成功，外面的 4 参数函数仍会继续生成标签；它并不直接让整次 AllGather 结束。

这张路线图省略了日志、检查、资源复用等细节；它不是可以编译的 C++。每层真实入口见后面的源码链接。

## 4. 第一步：host 把任务交给对应 operator

从 [HcclCommunicator::ExecOp 的创建与选择调用](../../../../src/legacy/ascend910/framework/communicator/impl/hccl_communicator_host.cc#L4461) 开始读。这里准备 `algName`、`newTag`，然后调用 6 参数的 `SelectAlg`。原文在 `hccl_communicator_host.cc` 第 4461、4481 行附近。

`algOperator->SelectAlg(...)` 中的 `->` 表示通过指针访问对象的成员函数。这里还没有执行通信，正在找合适的组织者。

[HcclAlg::GetAlgOperator](../../../../src/legacy/ascend910/algorithm/impl/hccl_alg.cc#L102) 使用 operator 注册表，根据命令类型取得对应对象。AllGather 的对应关系见 [REGISTER_OP 注册项](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L427)。注册项是建立对应关系的代码，不是每次调用时临时执行一次的分支。

对象建立时会执行构造函数。其基类 [CollAlgOperator::CollAlgOperator](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L40) 从配置器读取拓扑属性、算法属性、初始 `algType_`、是否允许默认 level1 自动选择、拓扑类型。**不是进入选择函数时所有成员都为空；它带着初始配置进来。** 配置器和拓扑的生成是更早的过程，暂时把它们当作已准备好的输入。

## 5. 第二步：同名 SelectAlg 为什么要经过几层

先看 [公共 6 参数 SelectAlg](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L118)。它先做 OnlyAIV 的公共支持检查。失败时通过宏返回错误，后面不会继续正常选择。

host 本段把 `limit.ifLimit` 设为 `true`，所以进入 5 参数的转接函数；[基类 5 参数 SelectAlg](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L60) 又调用 4 参数版本。AllGather 覆盖了 4 参数版本，因此实际进入 AllGather 的实现。这同时涉及“重载”（参数不同）和“虚函数派发”（对象的实际类型决定实现）；语法疑问查 [C++ 讲义](00_cpp_for_python_readers.md)。

读取 [AllGatherOperator::SelectAlg](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L38)：

1. 只有一个 rank？设置 `AllGatherSingleExecutor`，直接返回；不进入 910B 的多 rank 选择，也不执行该函数后面的标签拼接。
2. 混合设备类型？优先走混合设备函数。
3. 否则按设备类型分派；同型号 910B 进入 `SelectAlgfor910B`。
4. 被调用函数成功返回后，生成 `newTag`，再返回外层。

`algName` 是 `std::string&` 引用输出参数。内层给它赋值，就改变了外层那一个字符串。函数返回值 `HcclResult` 表示成功或错误；**算法名通过参数带回来，不通过返回值带回来。**

## 6. 第三步：910B 里面按什么顺序判断

打开 [AllGatherOperator::SelectAlgfor910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L131)，从第 131 行向下读。可以在纸上写两格：`algName = ?`、`level1 = ?`，每次赋值就更新。

| 顺序 | 做什么 | 对后面有什么影响 |
| --- | --- | --- |
| 1 | 计算每 rank 字节数；判断 Mesh/Ring 拓扑、AIV 配置等 | 准备后续条件 |
| 2 | 先判断单 Mesh 聚合组 AIV 路径 | 命中后填写名称，**立即返回**；后面的自动 level1 与普通 Mesh 都不读 |
| 3 | 单算子且跨聚合组时调用自动 level1 选择 | 在默认选择允许时，可能改 `algType_.algoLevel1`；这一步还不是最终执行器名称 |
| 4 | level1 为 AHC/AHC_BROKE 时处理 AHC | 本例配置 Ring，跳过 |
| 5 | level1 为 Pipeline 时检查 context 容量 | 超过容量则把 level1 改成 HD，后面用改后的值 |
| 6 | 读取 CCL 输入输出缓冲区，判断跨组 AIV 条件 | 读取失败立即报错；条件合格可选择 AIV RDMA 执行器 |
| 7 | 否则按 Mesh → Ring → 其他拓扑选名称 | Mesh 内还按模式、单聚合组、Pipeline 等继续分支 |
| 8 | 检查 OnlyAIV 的限制，返回结果 | 不是填了名称就必然成功；后面的检查仍可能报错 |

“单 Mesh 聚合组”是源码字段 `isSingleMeshAggregation_`，不要只凭“一个服务器”替代它。`OP_BASE` 是单算子运行模式，不是设备型号。

这里的顺序就是优先级。例如单组 AIV 已经成功返回，就不能再拿后面的 Mesh 分支覆盖它。只有执行到 `algName.empty()` 且它为真，才会进入对应的普通 Mesh 补选逻辑。

**自动选择只看一层，不要立即钻进所有辅助函数。** 先把 [AutoSelectAlgTypeLevel1](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L404) 当作“可能更新 level1 的子任务”。注意区分“外面调用了它”和“里面确实进入默认自动选择”：手动配置会影响后者。需要研究成本比较、Pipeline 阈值时再查 [公共机制](01_common.md) 和 [AllGather 条件表](02_allgather.md)。

## 7. 用一个场景逐行走完选择

下面是教学假设，不代表你当前有这样的机器：

- 同型号 910B、多 rank、Mesh 拓扑、跨 Mesh 聚合组。
- 单算子模式；AIV 与 OnlyAIV 均关闭；普通 AllGather 标签。
- 已显式配置 level1 为 Ring，默认自动选择开关关闭。
- 每 rank 输入 4096 个 float32，即 `4096 × 4 = 16384` 字节（16 KiB）。缓冲区及资源读取成功。
- 后续走 host 普通非零拷贝执行路径；本例不讨论 capture、重执行或提前缓存。

| 读到的位置 | 条件怎么判断 | 此时记下的结果 |
| --- | --- | --- |
| 4 参数函数第 41 行 | 多 rank，不走单 rank | `algName` 尚空 |
| 第 54 行 | 同型号 910B | 调用 910B 函数 |
| 910B 第 139–152 行 | AIV 关闭，不能命中单组 AIV | 继续往下 |
| 第 155–162 行 | 单算子、跨组，调用自动选择辅助函数；但默认自动开关关闭 | level1 保持 Ring |
| 第 165–178 行 | 不是 AHC，也不是 Pipeline | level1 仍为 Ring |
| 第 199–205 行 | AIV 关闭，跨组 AIV 不成立 | 进入普通 Mesh 分支 |
| 第 207–213 行 | 非单组，且 level1 不是 Pipeline | 还没有填 `algName` |
| 第 214–225 行 | 名称空；不是图模式；是单算子模式 | `algName = AllGatherMeshExecutor` |
| 第 233–248 行 | OnlyAIV 关闭 | 返回成功 |
| 回到 4 参数函数第 62–81 行 | 检查成功，读取 Ring 对应的标签字符串 | 拼接 `newTag`，末尾为 `_host` |

先不背条件，自己照表在源码上移动光标。走到这一刻，只能说“选中了 MeshExecutor，level1 为 Ring”；尚未完成通信。

## 8. 第四步：从字符串变成真实执行器

回到公共 6 参数 `SelectAlg`，不是立即回到 host。它的第 155–165 行按名称调用 `CollAlgExecRegistry::GetAlgExec`，取得执行器对象，设置属性，再取得 `algDesc`（描述它的执行能力等信息）。这里如果找不到注册名称，会返回错误。

因此执行器通常已在**公共选择包装层**创建，不要误以为一定到资源申请阶段才创建。

随后才回到 `ExecOp`。它会准备零拷贝信息、处理资源标签及资源复用；需要时调用 [CollAlgOperator::CalcResRequest](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L240)，让执行器描述所需资源，再分配通信资源。`newTag` 参与资源管理，不是另一种通信算法。已有资源与 capture 会改变这一段的具体分支。

最后 host 进入 [CollAlgOperator::Orchestrate](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L253)，它把工作交给 `executor_->Orchestrate(...)`。`Orchestrate` 在这里可理解为“组织并下发执行”。

## 9. 第五步：执行器内部还要选模板

本例的执行器进入 [CollAllGatherExecutor::Orchestrate](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L21)。这是 AllGather 执行器共享的调度代码。普通单算子、多 rank、非零拷贝、非 AllGatherV 的分支调用 `RunLoop`。图模式会直接走 `KernelRun` 等不同分支，不能照搬这一段。

读 [CollAllGatherExecutor::RunLoop](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_executor.cc#L219)：它按缓冲区容量计算一轮能处理多少元素，取 `curCount`，填入 `execMem.count`，调用 `KernelRun`；普通搬运分支还会在前后处理用户缓冲区与 CCL 缓冲区间的复制。剩余元素没有处理完，就下一轮。异步任务的组织/下发返回成功，不等于 CPU 返回时 NPU 已同步完成所有工作。

这里又发生一次派生类派发：实际对象是 MeshExecutor，所以进入 [CollAllGatherMeshExecutor::KernelRun](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L72)，不是任意一个同名 `KernelRun`。

本例内部顺序是：

1. 获取 level0/level1 子通信信息，把输入放到输出中的对应位置。
2. 为 level0 选择模板：910B 分支取得 `TEMPLATE_ALL_GATHER_MESH_ATOMIC`，准备参数，再 `RunTemplate`。
3. 为 level1 选择模板：当前字段是 Ring，取得 `TEMPLATE_ALL_GATHER_RING`，准备参数，再 `RunTemplate`。
4. 本轮完成调度后返回 `RunLoop`，必要时继续下一轮，最后逐层返回。

level0/level1 是此执行器划分的通信阶段；本例是组内 Mesh、跨组 Ring。准确的成员与链接由子通信信息决定，不能把所有路径的 level1 都简单叫作跨服务器。

第 138 行还允许“不同 device module 且单服务器”优先用 Ring；其他场景不能只看 level1 字段就断言模板。NHR、NB、HD 的后续分支见 [内部模板手册](07_internal_templates.md)。

**切片不是重选整个执行器。** 在这个 `RunLoop` 中没有重新调用 operator 的 `SelectAlg`；它反复调用同一执行器的内核。其他模板可能按本轮大小决定内部实现，不能与本次操作开始时的执行器选择混为一谈。

## 10. 换条件以后，具体在哪一站改道

每一行都是一个独立的教学场景，未列出的前置条件仍须成立。

| 改成什么条件 | 首先在哪儿改道 | 后续路线 |
| --- | --- | --- |
| 单组、AIV 开启、类型支持、普通标签、单算子、每 rank 700 KiB | 910B 第 144–152 行 | 小数据 AIV 执行器，直接返回；不会再走本例 MeshExecutor 的内核 |
| 上一行输入变为 700 KiB + 4 字节，仍满足单组 AIV 大小范围 | 同一个分支内部第 145 行 | 普通 Mesh AIV 执行器；分界条件是 `<=` |
| 跨组 Mesh、AIV 关闭、level1 为 Pipeline，context 未超过容量 | 910B 第 210 行 | MeshOpbasePipeline 执行器，不能继续套用普通 MeshExecutor 内核 |
| 上一行 context 超过 65535 | 更早的第 173–174 行 | level1 先改 HD；后面不再满足 Pipeline 分支，单算子跨组场景补选 MeshExecutor |
| 单组 Mesh、单算子、AIV 关闭 | 第 208–209 行 | MeshOpbase 执行器；它与 MeshExecutor 是不同对象 |

700 KiB 对应的定义见 [AIV_ALL_GATHER_SMALL_SIZE](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L40)；context 容量定义见 [HCCL_FFTS_CAPACITY](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L34)。这里只推导分支，没有证明这些阈值是你设备上的最快分界。

“切换”通常是下一次选择收到不同输入，走到不同分支；Pipeline 容量回退则是**同一次选择中先修改策略字段，再按修改后的字段继续判断**。不要想成正在执行到一半，把 MeshExecutor 对象突然换成另一个对象。

## 11. 现在怎样使用原学习手册

先看本文的总路线和主例，然后打开 [AllGather 手册](02_allgather.md)。看到一行条件，问三个问题：

1. 它位于上面哪一站？之前有哪些分支必须没命中？
2. 它修改的是执行器名称、level1 字段，还是内部模板？
3. 执行完以后是继续往下，还是 `return` 回上一层？

遇到函数调用，先在旁边写“输入什么、改了什么、成功后回哪里”，再决定是否跳进辅助函数。读宏时把 `CHK_RET` 想成“调用失败就退出当前函数”；不要跳过它当普通日志。遇到 `&`、`->`、`override`，回 [C++ 讲义](00_cpp_for_python_readers.md) 查语法即可。

其他算子也按“host → operator → 设备分支 → executor → 模板”追踪，但不能复制 AllGather 的分支与阈值。下一步读 [公共机制](01_common.md)，再读 [归约算子](03_reductions.md)：给归约算子额外记下归约类型和确定性等输入。

## 12. 三个自测：能回答就已经抓住执行顺序

**问题一：910B 的单组 AIV 分支 `return HCCL_SUCCESS` 后，`newTag` 还会生成吗？**

答案：正常多 rank 路径会。它回到 AllGather 的 4 参数 `SelectAlg`，成功检查后继续标签代码。若是 4 参数函数更早的单 rank 分支，则直接返回，跳过该函数标签代码。要先看 `return` 属于哪个函数。

**问题二：MeshExecutor 的 level1 是 Ring，为什么不能说最终执行器就是 RingExecutor？**

答案：`algName` 仍为 MeshExecutor；Ring 是其中 level1 阶段使用的模板。MeshExecutor 还组织 level0、缓冲区、切片等工作。

**问题三：把 Pipeline 的 context 改到超容量，哪个变量先变？**

答案：先把 `algType_.algoLevel1` 从 Pipeline 改 HD；`algName` 尚未在这一段确定。后面基于 HD 继续判断，主例的跨组单算子 Mesh 条件最终填 MeshExecutor。

下次请教时，可以直接指定：“沿本文第 7 节，停在第 155 行，只解释调用前后 level1 为什么保持 Ring。”一次弄清一个返回点，比同时展开全部算法更容易建立完整调用链。
