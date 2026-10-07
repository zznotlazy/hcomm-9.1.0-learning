# 算法切换、静态验证与证据限制

## 怎样判定真的“切换”

| 类型 | 示例 | 要检查的证据 |
|---|---|---|
| 调用之间输入变化 | AG 从 700 KiB 到更大，AIV Small→AIV | 是否重新选型、是否命中缓存、完整输入是否相同 |
| 选择过程中的修正 | Pipeline context 超容量→HD/NHR | 修正前的类型、检查顺序、最终执行器 |
| 内部模板分派 | Mesh 执行器内部 level1 Ring→NHR | 哪层、哪个子域、模板是否实际运行 |
| 执行形态变化 | 分块次数、BCopy/ZCopy、host/AI CPU | 不自动等同于算法家族切换 |
| 重试/业务特殊分派 | BatchSendRecvRetry | 重执行并不自动意味着 Ring/Mesh 动态寻优 |

[创建算子对象](../../../../src/legacy/ascend910/algorithm/impl/hccl_alg.cc#L102) · [公共执行器缓存](../../../../src/legacy/ascend910/algorithm/impl/operator/coll_alg_operator.cc#L240)

常规 host ExecOp 创建新算子对象，选择后按 newTag 复用通信资源；算法名和上层类型常被编码进 tag。但 AIV O 路径在选择之前可能命中 `hcclCacheMap_` 直接 ExecOpCache，必须查看 OpParam 的比较/哈希和 cache 创建条件后才能判断改参数是否重新选型。记录了这个例外，不把“每次必重选”当结论。

host 函数还出现 `hostSendRecvInfo` 非空时创建另一个对象、把 aicpuUnfoldMode 改 false 再 SelectAlg 的代码；注释明确为 A3 BSR NPU Direct 残留工作，**不是本次证明的 A2 运行中重选机制**。

没有找到证据表明 A2 这些传统选择器会根据运行时测到的带宽/耗时自动在算法之间寻优；V2 使用固定延迟和静态带宽表。这个结论限定已分析选择链，不扩大到项目中所有未来扩展或自定义业务。

## 切换总表

| 操作/阶段 | 原候选→结果 | 完整关键条件（另需已到达该位置） | 配置能否绕过 |
|---|---|---|---|
| 通用 level1 | 显式 Pipeline→默认 V1 | A2 且非 O | 不能；算子内部仍可能有 GraphPipeline 名称 |
| AG | Pipeline→HD | context(AG)>65535；单组 AIV 早返回时不走这步 | 显式指定也检查 |
| AR | Pipeline→HD | O 且 context(AR)>65535 | 不能 |
| RS | Pipeline→HD | 未提前进入 STRICT 浮点/AIV 返回且 context(RS)>65535 | 不能 |
| AGV | Pipeline→NHR | context(AGV)>65535 | 不能 |
| RSV | Pipeline→NHR | 未提前返回确定性分支且 context(RS)>65535 | 不能 |
| RSV | HD等→NHR | 多组 G 对称确定性分支，或 O 的普通 MeshOpbase；不在 Ring/NB/NHR | 不能 |
| Scatter | HD/Pipeline等→Ring | R>1，level1不在 Ring/NB/NHR | 不能 |
| AHC | AHC→AHC_BROKE | AHC 前置合法；autoSelect开启；子组等长或 AG/RS 每 server 多卡强制 | 已有 BROKE 不在此处反向升级 |
| AHC 内拼接 | Ring→NHR | 子组数量从<=3变为>3 | 代码写死，不声称存在数据阈值 |
| AG/AGV AIV | Small→普通 AIV | O，其他 AIV条件不变；D或DV由<=700 KiB变为>700 KiB | Q不能改变这条分类阈值 |
| RS/RSV AIV | Small→大数据 AIV | O，其他条件不变；D或DV由<=2 MiB变为>2 MiB | Q有资格例外，不等同改变分界 |
| AR AIV | Small→MeshAiv | 单组、O、非确定性、条件满足；D达到16 MiB | >=边界属于大数据 |
| AR传统 | Small→OpbaseLoop | SingleMeshInlineReduce、O；D>256 KiB | 先检查前面的 AIV/STRICT/CCL指针优先级 |
| AlltoAll V | Continuous→TwoLevel/其他 | capture或NA+pairwise改变，Direct等优先路径未中 | 重新按整条有序链判断 |
| AlltoAll复制 | ZCopy→BCopy | O，任一最大长度+偏移>Bi，且到达 CopyMode 函数 | 不等同模板家族一定改选 |
| Batch | 普通/Group→Retry | retryEnable && aicpuUnfoldMode | Retry优先Group |

## 场景走查：怎样验证表没有漏条件

这里的“验证”是按真实函数表达式静态走查，不是新写一个同逻辑模拟器再拿模拟器证明原函数。没有执行生产代码；人工构造字段可命中分支，不保证真实设备能产生该拓扑。

统一设 type=FP32，增量4字节，所有参数/资源合法，除特别指定外非混合、Q=false、普通 tag。

| 场景 | 指定条件/边界 | 静态预期 |
|---|---|---|
| 单 rank | R=1，AG/AR/RS/Reduce/Scatter | Single；不套用多 rank 阈值 |
| 单 rank强制AIV | Q=true，公共支持算子 | 公共包装层拒绝 |
| AG单组AIV | A、S、O；700 KiB-4/700 KiB/700 KiB+4 | Small/Small/MeshAiv |
| AG资格上限 | A、S、O、Mesh；256 MiB/256 MiB+4 | AIV/传统MeshOpbase；Q=true可放宽但资源未验证 |
| AG跨组默认 | A=false、O、!S、M=2、K=8、R=16、L0Mesh、Lite | 32 KiB-4：level1 Ring；32 KiB：Pipeline |
| AG跨组AIV优先 | 前项 A=true、Mesh、Bi/Bo>=16 MiB | 小数据条件满足最终AivRdma，不能仅看已选level1 |
| 非Lite | AG跨组、A=false，其余同上 | 2 MiB-4不因尺寸命中Pipeline；2 MiB命中 |
| AG图模式 | G、Mesh、!S、M=8、K>1、A=false | 1 MiB：MeshGraph；1 MiB+4：GraphPipeline |
| AG图模式M例外 | 上项 M=4 | 等于1 MiB也可GraphPipeline |
| Pipeline容量 | 已选Pipeline、未提前返回；context=65535/65536 | 不因容量回退/按算子改HD或NHR |
| 静态模型 | !S、非Pipeline、M=2/3/4/8，正数据 | V2 Ring/NHR/NHR/NHR；初始化V1不同 |
| AR确定性AIV | A、S、det=ENABLE、L0Mesh、Buf；8 MiB/8 MiB+4 | Small确定性AIV/脱离此AIV资格；大数据AIV分支不可达 |
| AR严格保序 | det=STRICT，FP32，组内卡数对称 | OrderPreserved；即使A=true也优先保序 |
| AR小数据边界 | A=false、SingleMeshInlineReduce、O、非CCL指针；256 KiB/256 KiB+4 | SmallCount/OpbaseLoop |
| AR确定性分类 | A=false、S、K=8、O、ENABLE；128 KiB/128 KiB+4 | SmallCount/MidCountLoop（排除CCL指针特例） |
| RS AIV边界 | A、S、O、det=DISABLE；D=2 MiB/2 MiB+4 | SmallCount/MeshAiv |
| RS确定性总量 | A、S、serverNum=1、ENABLE；D*R=8 MiB/超过8 MiB | 确定性SmallAIV/常规路径；Q可使大数据确定性AIV可达 |
| RS小数据不等式 | !S、O、SDMA支持，A=false且未选Pipeline | det=ENABLE且D<=512 KiB可Small；另一条件D*R严格<512 KiB |
| Broadcast资格 | A、S、O、Bi/Bo>=16 MiB；4 GiB/4 GiB+4 | AIV/传统拓扑候选 |
| Reduce拓扑 | A2 R>1，A=false；Mesh/Ring/其他 | Mesh/RingPlusHd/Comm |
| Scatter覆盖 | R>1，显式HD，Mesh | level1改Ring，执行器仍为ScatterMesh |
| AGV统计 | counts改变但max不变/ max越过700 KiB | AIV大小分类不变/改变；实际布局仍可改变 |
| RSV图模式 | G、!S、卡数对称/非对称 | Deter/不支持；与det是否ENABLE无关 |
| RSV能力 | O、非确定性、A=false、未选Pipeline、SDMA不支持 | 不支持；不是自动普通Ring兜底 |
| AlltoAll AIV跨组 | 固定AlltoAll、A、O、对称、M<=8、满足基础Buf | RoceDirect数据总量<=4 MiB且+2 MiB<=Bi才可DirectAIV |
| AlltoAllVC Direct | S、O、A=false；每rank两方向都>Bi /任一方向==Bi | Direct/不因此条件Direct |
| AlltoAllV Pipeline | A=false、!capture、非Direct、对称、O、每组>1 rank、可整除、Ring/HD/Pipeline；Bi=80 MiB-4/80 MiB | 不因Pipeline条件选中/Continuous |
| AlltoAll复制 | 到达CopyMode；最大length+offset=Bi/Bi+4 | ZCopy/BCopy |
| Batch优先级 | Group=true；retry&&aicpu=true/false | Retry/Group |
| AR内部NHR | AllReduceMesh、level1=NHR、该层实际执行；floor(C/K)=256 KiB/256 KiB+1 | NHR_ONESHOT/NHR；看当轮C，不是完整D |
| Broadcast内部NHR | BroadCastMesh、level1=NHR、该层实际执行；floor(C/K)=2 MiB/2 MiB+1 | NHR_ONESHOT/NHR |
| Broadcast内部NB | NB、r=2；合法任意dataSize | helper直接选择NB_BINARY |
| Broadcast内部NB边界 | NB、r>2且带宽获取成功；dataSize=T-1/T | NB_BINARY/NB；T按原式整数截断与float求值 |
| Broadcast Ring配置 | BroadCastMesh / BroadCastComm，level1=Ring | RECURSIVE_HD / RING；不能通用映射 |

额外可达性检查：A2 单 server 不保证 S；构造 NP_DOUBLE_RING/4P_RING 必须与初始化来源核对；G 的显式 Pipeline 会先被配置层转默认，不把不可到达状态标成真实支持。抽象 context=65536 是容量判断边界，不承诺现实可用拓扑恰好产生它。内部模板条件见 [模板切换页](07_internal_templates.md)。

## 现有测试如何使用

[公共算法测试](../../../../test/ut/inter/comm/ut_coll_alg_operator.cc#L93) 构造A2参数，覆盖辅助函数入口、single mesh plane 与 retry/inplace检查；部分辅助函数调用没有对选择结果做断言，不能当作完整选型验证。

[AllGather共用测试](../../../../test/ut/inter/comm/ut_allgather_common.cc#L246) 包含不同rank数和数据类型的功能测试；测试名/文件名不证明覆盖了本手册的每条A2选择分支。

[明确算法名断言示例](../../../../test/ut/impl/ut_hccl_impl_new.cc#L309) 是910_93，不能充当A2的Broadcast AIV验证。

本次没有CANN开发依赖和NPU环境，不执行LLT。后续若获得环境，优先增加/运行对 `algName`、`algType_.algoLevel1` 和异常返回的真实选择器测试，再做日志与性能验证。

## 文档检查与局限

证据生成器检查：源码链接存在且行号有效；手工锚点能找到；所有15个注册命令已列入索引；候选执行器能解析到注册位置；模板出现位置能关联注册或显式标记未解析；记录源码哈希和Git提交。它验证追溯完整性，**不是证明硬件性能，也不是对分支可达性的自动形式证明**。

源码索引的执行器/模板列表是出现位置清单，包含共享实现的其他芯片分支。手工决策表已按A2选择入口限制范围；逐模板的设备判断和子域大小仍必须阅读原函数。公开API参数支持、构建宏及资源合法性可能在更外层拒绝场景，本手册不提供完整硬件认证矩阵。
