# A2/910B 算法选择学习手册

本手册分析当前 hcomm 源码快照中 `DEV_TYPE_910B` 的传统集合通信路径。目标是回答：什么输入选择哪个算法，输入变化时怎样改选，配置为什么可能失效，以及原因能证明到什么程度。

## 阅读路线

只会基础 Python、尚未学习 C++ 时，先读 [C++ 源码阅读讲义](00_cpp_for_python_readers.md)。它用 Python 对照解释必要语法，包含真实 AllGather 片段、10 道练习与答案。

1. [公共选择机制](01_common.md)：先弄清拓扑、配置、分层算法与执行器的关系。
2. [AllGather 完整示范](02_allgather.md)：按条件推导一次完整选择。
3. [AllReduce 与 ReduceScatter](03_reductions.md)：理解确定性、归约能力和小数据优化。
4. [Broadcast、Reduce、Scatter 与点对点](04_other_ops.md)。
5. [AllGatherV、ReduceScatterV 与 AlltoAll 系列](05_variable_alltoall.md)。
6. [切换与场景验证](06_switch_validation.md)：边界、覆盖优先级和证据限制。
7. [执行器内部模板切换](07_internal_templates.md)：NHR oneshot、NB binary、分层模板和按轮切片。
8. [源码证据索引](source_index.md)：选择函数原文、注册关系、模板入口及现有测试索引。

每页的决策表按源码执行顺序阅读：**后一行隐含前面选择分支未命中**。报错和前置处理必须先检查。符号在公共机制页定义；所有 KiB/MiB/GiB 都按 1024 换算。

## 如何回答“为什么”

- **代码事实**：实际条件、赋值、返回、调用和注释明确规定的限制。
- **机制推断**：根据调度次数、通信轮次、搬运量、并行度解释选择动机，不能当成性能测量。
- **未证实**：经验阈值的原始调优过程、真实设备最快算法、完整硬件支持矩阵。

例如“Pipeline 的 context 数超过 65535 就回退”可以由源码证明；“700 KiB 是当前设备的最优分界点”没有测量就不能证明。

## 分析边界与基线

分析基于上游源码快照，分支 `9.1.0`，提交 `9aa5deb8a2a14de0c4cd6b077ed9af1555c1402f`，来源为 `https://gitcode.com/cann/hcomm.git`。分析基线见 [baseline.json](baseline.json)，同时以 SHA-256 标识实际读到的源码。切换了版本后应重新分析，不把这些阈值视为跨版本承诺。云端教学背景见 [学习交接](CLOUD_LEARNING.md)，完整快照来源见 [source_provenance.json](source_provenance.json)。

核心范围为同型号 A2，单 rank、单 Mesh 聚合组、跨聚合组、图模式和单算子模式；混合设备路径作为入口例外记录，不把 A3/A5 分支结论移植到 A2。MC2、自定义算子、V2 分派以及对称内存等不同入口需要先确认是否进入本手册的传统选择链。

手册覆盖传统注册表的所有 15 个命令项（AlltoAll 三种变体共享选择器）。Barrier 借用 AllReduce，另外解释。BatchWrite 是基础类占位选择路径，不冒充已完整分析的特殊业务。`GATHER` 枚举/配置项不等于存在独立 Gather 选择器。

公共 API 和通信库实现均未修改。没有 NPU/CANN 运行环境，本次验证是静态走查、源码索引完整性检查和数学条件核对，没有运行 HCCL 单元测试或性能实验。

## 查询一个场景前要给出的信息

算子与数据类型；归约操作（如适用）；count/counts 矩阵；rank 数；server 数；Mesh 聚合组及组内卡数；链接类型；单算子/图模式/capture；CCL 输入输出容量；分层配置；AIV/OnlyAIV/AI CPU；确定性；重执行状态。只给“2 卡 + 1 MiB”通常不能唯一确定选择结果。

生成和检查索引：在仓库根目录运行 `python docs/zh/learning/a2_algorithm_selection/build_evidence.py`。这会更新本目录的证据附件与基线；不改通信库源码。人工说明中的链接与源码锚点也会检查。
