# 从基础 Python 到 hcomm C++ 源码阅读

你的目标是看懂算法选择代码，能够回答“输入是什么、条件怎样判断、哪个变量被修改、什么时候返回、下一步调用谁”。先掌握这些，比先学会搭建整个 C++ 项目更有帮助。

这份讲义按零 C++ 基础编写。示例分成两种：**教学示例/改写**用于解释语法，不是生产实现；**源码片段**来自本仓库，附路径、函数和行号。删节会明确标注。短例子通常省略头文件和程序入口，单独的语句应放入函数体，不是完整可直接编译的程序。教学示例未经过编译或运行，也没有执行 CANN/NPU 测试。

建议分三次读：第一次第 1～3 节，第二次第 4～7 节，第三次第 8 节综合阅读。练习可以先自己回答，再看文末答案。所有源码链接以本仓库的 9.1.0 快照为准。

## 1. 先看清代码形状：类型、语句与代码块

### Python 对照与 C++ 示例

教学示例：

```python
count = 8
enabled = True
name = "Mesh"
if enabled:
    count = count + 1
```

```cpp
int count = 8;
bool enabled = true;
std::string name = "Mesh";
if (enabled) {
    count = count + 1;
}
```

逐句读：`int count = 8;` 是“声明一个整数变量 count，初始化为 8”。变量类型写在名字之前；这里的 `=` 是初始化。后面的 `count = count + 1;` 则是赋值。

- `bool` 是布尔类型，字面值写 `true/false`，不是 `True/False`。
- `std::string` 是标准库字符串类型；`std::` 暂时读作“标准库里的”。
- 分号 `;` 结束一条语句；换行本身一般不结束语句。
- `{ ... }` 划分代码块，缩进主要服务阅读。`if` 的条件要放在圆括号里。
- `//` 注释到行尾，`/* ... */` 可以跨行。

花括号还划定局部变量的作用域：在内层块声明的变量，通常不能在块外使用。函数、类、命名空间也使用花括号，含义要结合前面的关键字判断；类和结构体定义末尾通常还有分号 `};`。

### 阅读这里会遇到的类型

| 写法 | 先怎样理解 | 与 Python 的关键区别 |
|---|---|---|
| `int` | 普通有符号整数 | 位宽由实现决定；不是任意精度整数 |
| `u32/u64` | 项目常用的无符号整数类型名，按 32/64 位用途阅读 | 有固定范围，不表示负数；具体定义从头文件链确认 |
| `float/double` | 浮点数 | 精度有限，比较和整数除法转换有细节 |
| `std::string` | 字符串对象 | 可以通过引用在函数内改写对象 |
| `HcclResult` | 项目定义的返回类型 | 先去找定义和成功/错误值，不由名字推断为 bool |

教学上，可以用 `using u64 = std::uint64_t;` 说明“给现有类型起别名”，需要包含 `<cstdint>`；这是教学写法，不声称就是本项目的定义。`typedef` 也是类型别名机制。见到陌生类型先找 `using`、`typedef`、`enum`、`struct` 或 `class`。

Python 整数通常会自动扩展，C++ 的固定宽度整数不会。无符号整数运算超出范围会按其位宽回绕；有符号整数溢出不能当成可靠的回绕规则。尤其不要假设无符号的 `0 - 1` 会保存成负数。

`u64 size;` 只声明局部基础类型变量，没有保证它自动为 0。`u64 size = 0;` 才明确初始化；`std::string name;` 则会构造一个空字符串。别把所有声明都理解为 Python 中某个统一的默认值。

### hcomm 源码实例

[all_gather_operator.cc，SelectAlgfor910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L131)，连续片段：

```cpp
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
```

`SIZE_TABLE[...]` 按数据类型查每个元素的字节数；第二行是元素个数乘每个元素字节数。不能把 `count` 直接当作字节数。

[hccl_aiv.h，常量定义](../../../../src/legacy/ascend910/algorithm/pub_inc/hccl_aiv.h#L40)，原文：

```cpp
constexpr u64 AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
```

`constexpr` 在这里声明编译期常量，不允许之后重新赋值；数值是 716800 字节，即 700 KiB。仅有这个定义，还不能知道哪个算子、哪种模式真的使用它；必须继续查引用位置。

**练习 1：** 假设 count=8，unitSize=4，上面两行最终 dataSize 是多少？如果只有 `u64 dataSize;`，能认为它等于 0 吗？

## 2. 条件与计算：重点看括号和边界

### Python 对照与 C++ 示例

教学示例：

```python
eligible = enabled and single_group and (size <= limit or only_aiv)
suffix = "_device" if use_device else "_host"
```

```cpp
bool eligible = enabled && singleGroup && (size <= limit || onlyAiv);
std::string suffix = useDevice ? "_device" : "_host";
```

| C++ | Python 对照 | 含义 |
|---|---|---|
| `&&` | `and` | 两项都满足 |
| `||` | `or` | 至少一项满足 |
| `!x` | `not x` | 否定 |
| `==`、`!=` | 同形写法 | 相等、不等 |
| `<`、`<=`、`>`、`>=` | 同形写法 | 注意边界是否包含等于 |
| `condition ? a : b` | `a if condition else b` | 根据条件选择一个表达式 |

C++ 内建 `&&/||` 返回 bool；Python 的 `and/or` 可以返回操作数本身，因此对照表只比较判断意义。内建逻辑运算从左到右短路：`false && 后项` 不求值后项；`true || 后项` 也不求值后项。

例如教学表达式 `ptr != nullptr && ptr->ready` 先确认非空，再访问成员。它只防止空指针这一种情况，不能证明指针还活着或指向合法对象。

`&&` 的优先级高于 `||`，`!` 也不是把整条表达式自动取反。初学时不要靠记忆猜，先按源码括号拆分：`a && (b || c)` 与 `(a && b) || c` 不一样。

### 整数除法与位运算

教学示例：

```cpp
int a = 5 / 2;       // 2
double b = 5 / 2;    // 2.0：先做整数除法，再转换
double c = 5.0 / 2;  // 2.5
```

C++ 整数除法向零截断。对于本手册常见的非负大小，结果可按 Python `//` 理解；对于负数则不同：C++ `-5 / 2` 为 -2，Python `-5 // 2` 为 -3。Python `/` 通常产生浮点结果，不能直接移植来模拟 C++ 整数表达式。

`static_cast<T>(x)` 表示显式类型转换；转换为浮点发生在除法之前还是之后，会影响结果。转换到较窄整数类型可能丢失信息，`static_cast` 不是范围校验工具。赋给 `u64` 也不自动保证之前所有中间运算都用了 64 位。

另外，`&` 和 `|` 可以是**按位**与/或，不是 `&&/||`。按位运算通常不做这种短路。`<<`、`>>` 可以用于位移；循环里 `++i` 表示自增。`&` 的另一种用途会在第 3、4 节解释。

### hcomm 源码实例

[SelectAlgfor910B，AIV 资格判断](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L139)，原文连续片段：

```cpp
    bool isAivMode = topoMatcher_->GetAivModeConfig()
                    && isSingleMeshAggregation_
                    && IsSupportAIVCopy(param.DataDes.dataType)
                    && (dataSize <= AIV_BIG_SIZE || isOnlyAiv);
```

把它读成四道门：AIV 配置开启、单 Mesh 聚合组、类型支持 Copy、最后的数据条件。`isOnlyAiv` 只放宽这段最后一道门；不绕过前面三道门，也不绕过调用链其他地方的检查。赋值右边计算完，结果才保存到 isAivMode。

[同一函数，服务器数判断](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L188)，原文：

```cpp
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
```

这里中间是单个 `&`，用于位运算。正整数 n 若只有一个二进制位为 1，则 `n & (n-1)` 为 0；因此判断正数是否为 2 的幂。前面的 `serverNum_ > 0` 不可省略：0 不是这个判定要接受的服务器数。

[SelectAlg，标签后缀](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L79)，原文：

```cpp
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
```

`+=` 在字符串上是追加；布尔值为 true 追加 `_device`，否则追加 `_host`。

**练习 2：** 在上述 isAivMode 判断中，AIV 配置为 false、isOnlyAiv 为 true，结果是否为 true？为什么？

**练习 3：** `double x = (8 / 3) * 2.0;` 的 x 是多少？改成 `(8.0 / 3) * 2.0` 后是否相同？

## 3. 函数与参数：返回值和输出参数可以同时存在

### Python 对照与 C++ 示例

教学示例：先看复制、别名和函数输出。

```cpp
void setByValue(std::string name) {
    name = "Mesh";
}

void setByReference(std::string& name) {
    name = "Mesh";
}

void demonstrate() {
    std::string selected = "Ring";
    setByValue(selected);      // selected 仍为 "Ring"
    setByReference(selected);  // selected 变为 "Mesh"
}
```

`void` 表示不返回一个值，不代表函数不会修改其他对象。按值接收的 `std::string name` 是一个独立字符串对象；`std::string& name` 是调用者原对象的引用，对它赋值就修改调用者的字符串。

不要直接套用 Python 的字符串参数：

```python
def set_name(name):
    name = "Mesh"

selected = "Ring"
set_name(selected)  # selected 仍为 "Ring"
```

Python 中这次赋值重新绑定函数内部的名字，外面的名字没有改变。用 `list`/`dict` 修改内容可以示意“外部可见的修改”，但 Python 的名字绑定仍不等于 C++ 引用。**读函数签名才能判定参数是否是引用，不能仅看调用形式。**

`const std::string& name` 仍引用原字符串，但不能通过这个引用直接改写字符串。`const OpParam& param` 不表示对象做了深度冻结；其中指针指向的对象仍可能被修改。按值传入的对象也可能含共享指针，所以“按值”不等于整个对象图都被深复制。

### 声明、定义和重载

教学示例：

```cpp
int add(int a, int b);                 // 声明：告诉编译器接口
int add(int a, int b) { return a+b; }  // 定义：提供实现
```

`.h` 常放声明和类型，`.cc` 常放实现，但不是语言强制规则。相同函数名可以因参数类型或数量不同而重载；不能只靠返回类型不同形成重载。`return` 立即结束当前函数，后续语句不会继续执行。

### hcomm 源码实例

[all_gather_operator.cc，SelectAlg 定义](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L38)，签名原文：

```cpp
HcclResult AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
```

| 部分 | 阅读意义 |
|---|---|
| `HcclResult` | 函数直接返回一个状态结果 |
| `AllGatherOperator::SelectAlg` | AllGatherOperator 类的成员函数 |
| `const std::string& tag` | 通过只读引用接收标签 |
| `const OpParam& param` | 通过只读引用接收参数对象 |
| `std::string& algName` | 可修改调用者的算法名称字符串 |
| `std::string& newTag` | 可修改调用者的新标签字符串 |

因此，它可以“返回 HCCL_SUCCESS，同时把算法名写入 algName”。引用不自动表示输出参数；这里要结合函数体中的赋值，才能确认用途。

[SelectAlg，设备分派中的片段](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L54)，原文连续片段：

```cpp
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
```

调用处没有额外写 `&algName`；被调用函数的签名使用引用就够了。`ret` 接收返回状态，`algName` 的内容则可能在子函数中被改写。这是两条不同的结果通道。

[coll_alg_operator.h，SelectAlg 重载声明](../../../../src/legacy/ascend910/algorithm/pub_inc/coll_alg_operator.h#L44)，查看四参数与带 ResourceLimit 的版本。追调用时应按实参数量、顺序和类型确认对应声明，不只搜第一个同名函数。

**练习 4：** 教学示例中先调用 setByValue，再调用 setByReference，selected 分别是什么？为什么调用时看不到引用符号？

**练习 5：** SelectAlg 返回成功时，是否能断定 newTag 一定被这次调用写入？提示：先看 R=1 分支的位置。

## 4. 对象、成员与指针：别把所有 `&`、`*` 读成一种东西

### Python 对照与 C++ 示例

Python 中对象常见 `obj.field`、`obj.method()`。C++ 要区分对象本身和指向对象的指针。

教学示例：

```cpp
struct State {
    int count = 0;
};

State state;
State& alias = state;
State* ptr = &state;
state.count = 1;
alias.count = 2;
ptr->count = 3;
```

此时三种写法操作同一个 State 对象，最终 count 为 3。

| 出现位置 | 写法 | 读法 |
|---|---|---|
| 声明 | `State& alias` | alias 是引用 |
| 表达式 | `&state` | 取得 state 的地址 |
| 表达式 | `a & b` | 两个整数的按位与 |
| 声明 | `State* ptr` | ptr 是指针变量 |
| 表达式 | `*ptr` | 解引用，访问所指对象 |
| 表达式 | `a * b` | 相乘 |
| 成员访问 | `state.count` | 对象或对象引用的成员 |
| 成员访问 | `ptr->count` | 指针所指对象的成员，相当于 `(*ptr).count` |

引用初始化后不会因赋值而“改绑”；例如 `alias = other` 是把 other 的内容赋给被引用对象。指针则可以重新指向别处，也可以为空。`nullptr` 是空指针字面值，不能对它解引用。

`struct` 与 `class` 都能包含数据和函数；主要基础区别是 struct 默认成员访问权限为 public，class 默认为 private。不是“struct 只能放数据、class 才能放函数”。Python 的属性访问机制与这里的权限/内存布局不同，只借用“对象有成员”这个概念。

### hcomm 源码实例

[coll_alg_param.h，OpParam 定义](../../../../src/legacy/ascend910/algorithm/pub_inc/coll_alg_param.h#L147)，在类型定义中找 DataDes、tag、指针成员。读 `param.DataDes.count` 时，先访问 param 的 DataDes 成员，再访问这个成员对象的 count，不是调用函数。

[SelectAlgfor910B，配置读取](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L138)，原文：

```cpp
    bool isOnlyAiv = topoMatcher_->GetIsOnlyAivConfig();
```

`topoMatcher_` 是成员变量；尾部下划线是项目命名习惯，不是 C++ 特殊语法。这里的 `->` 对智能指针同样可用，第 6 节解释。`()` 表示调用成员函数，把返回结果保存到局部 bool。

[同一函数，通信缓冲区输出](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L181)，原文连续片段：

```cpp
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;
```

`void*` 只保存地址，没有可直接解引用的元素类型；不能照写 `*commInputPtr` 来读取一个元素。后续 GetInCCLbuffer 调用会通过输出参数返回缓冲区地址与大小；要进入其声明确认引用类型。指针类型转换也不会自动分配内存或证明地址有效。

**练习 6：** 上面 State 示例最终 count 为什么为 3？把 ptr 设为 nullptr 后，能直接使用 `ptr->count` 吗？

## 5. 命名与类关系：`::`、继承和 virtual

### Python 对照与 C++ 示例

Python 类可以继承并重写方法。C++ 的运行时多态需要明确看虚函数与调用对象。

教学示例：

```cpp
namespace demo {
enum class Device { A2, Other };

class Base {
public:
    virtual int select() { return 0; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    int select() override { return 1; }
};
}

demo::Derived object;
demo::Base& base = object;
int result = base.select();  // 1
```

`namespace demo` 把名字放进作用域；`demo::Derived` 是作用域限定，不是访问某个对象的成员。`enum class` 声明带作用域的枚举；`Device::A2` 是其中一个值。

`class Derived : public Base` 声明继承。`public/private/protected` 控制成员访问；private 是访问限制，不是数据加密。通过基类引用/指针普通调用虚函数时，执行实现取决于实际对象类型。按值复制成基类对象会有不同语义，不能把上面的结论扩展到所有传递方式。

`override` 要求编译器确认它确实重写了基类虚函数；不是单靠这个词才让函数变成虚函数。`virtual ~Base()` 是虚析构函数；析构函数名以 `~` 开头，用于对象销毁时的清理。`= default` 要求使用编译器提供的默认实现，不是“返回默认值”。

### 构造函数与初始化列表

构造函数与类同名，没有返回类型。它用来初始化对象，和 Python 的 `__init__` 只在用途上有相似之处，生命周期规则并不相同。

[all_gather_operator.cc，构造函数](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L28)，完整原文：

```cpp
AllGatherOperator::AllGatherOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER)
{
}
```

参数列表后的 `:` 开始初始化列表，这里初始化基类 CollAlgOperator，并传入 ALLGATHER 命令类型。不是 Python 的“代码块冒号”，也不是三元表达式里的冒号。函数体为空不表示什么都没做；基类已在进入函数体之前初始化。

### hcomm 源码实例

[all_gather_operator.h，类声明](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.h#L17)，原文中的两个独立片段，中间省略构造/析构声明：

```cpp
class AllGatherOperator : public CollAlgOperator {
```

```cpp
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag) override;
```

结合 [CollAlgOperator 的 virtual 声明](../../../../src/legacy/ascend910/algorithm/pub_inc/coll_alg_operator.h#L44)，可以理解：基类接口上的普通虚调用，实际对象若是 AllGatherOperator，就可能进入它的 SelectAlg 实现。`SelectAlgfor910B` 是另一个私有辅助函数；看见同一个类里两个类似函数名，不代表它们都属于虚函数覆盖。

函数体中的 `userRankSize_` 等成员，也可能来自基类。要找其来源，应检查类声明、继承链及初始化/SetAttr 等赋值位置，不能只搜当前 .cc 的局部变量。

**练习 7：** 教学示例的 result 为什么为 1？hcomm 构造函数花括号为空，是否表示没有调用基类构造函数？

## 6. 常见标准库写法：只学追代码需要的部分

### Python 对照与 C++ 示例

教学示例：

```python
counts = [2, 4, 8]
total = 0
for count in counts:
    total += count
```

```cpp
std::vector<int> counts = {2, 4, 8};
int total = 0;
for (int count : counts) {
    total += count;
}
```

`std::vector<int>` 是保存 int 的动态数组；尖括号给出模板类型参数，不是小于/大于比较。它在用途上类似列表，但元素类型、对象复制和内存规则不同。

`for (类型 名字 : 容器)` 是范围循环。`auto item` 通常接收值，`auto& item` 接收引用，`const auto& item` 以只读引用访问；vector 的 `operator[]` 不自动检查下标越界，不能期待一定像 Python 一样抛 IndexError。

`auto` 让编译器根据初始化表达式推导类型，不是 Python 式运行时动态类型。教学例子 `auto value = object;` 不因为用了 auto 就变成引用；需要引用时写 `auto&`。推导出的类型之后仍固定。

### 字符串、map 与迭代器

| 写法 | 阅读意义 |
|---|---|
| `name.empty()` | 字符串是否为空 |
| `name.c_str()` | 取得用于 C 风格接口的字符指针；不是打印命令 |
| `std::to_string(n)` | 数值转字符串 |
| `map.find(key)` | 寻找 key，返回迭代器 |
| `map.end()` | 尾后位置，用于表示 find 未找到；不能解引用 |
| `iter->first/second` | 对 map 元素，分别读取 key/value |

map 和 unordered_map 用途都类似字典；前者通常按键组织有序遍历，后者以哈希组织。不要仅凭变量名中的 MAP 断定具体容器类型。迭代器可先理解为“定位到容器元素的对象”，不保证一定就是裸指针。

### hcomm 源码实例

[SelectAlg，查找 level1 名称](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L70)，原文连续片段：

```cpp
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), 
                    HCCL_ERROR("[AllGatherSelector]level1: algType1[%u] is invalid.", algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
```

先寻找当前 level1 类型的文字名称。未找到就通过宏返回错误；因此成功继续到最后一行时，才可读取 second。这里的字符串拼接用于 newTag，不是再次选择执行器。这个检查宏在第 7 节展开。

### `unique_ptr` 的最低必要含义

[AllGather Mesh 执行器，KernelRun](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L72)，原文中的声明：

```cpp
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
```

它是独占拥有一个 AlgTemplateBase 对象的智能指针，默认构造时为空；不自动创建模板对象。获得对象后可以 `level1TempAlg->Prepare(...)` 调用成员函数。普通销毁会清理其拥有的对象，但不能单凭这个声明证明所有派生类型销毁规则都正确，仍需看类定义。

unique_ptr 不能像普通可复制对象那样复制所有权；通常可通过移动转移。`std::move` 是允许移动操作的转换，本身不是“立即复制/搬走对象”的函数。`.get()` 可以取出非拥有的裸指针，原拥有者销毁后不要继续使用它。此阶段不必深入实现所有权转移，只需认出“一个带所有权的对象指针”，不要把它当成 vector 一类的容器。

**练习 8：** find 返回 end 时能读取 second 吗？为什么上面的宏必须在读取 second 之前？默认构造的 level1TempAlg 已经包含一个可调用的模板对象吗？

## 7. 项目宏：看起来像调用，实际能返回当前函数

### Python 对照与 C++ 示例

Python 的 `import` 与 C++ 的 `#include` 都能帮助使用其他文件里的定义，但机制不同：`#include` 是预处理阶段包含头文件内容，不是运行时执行一个 Python 模块。

`#define` 定义宏，调用形状可能像函数，但预处理展开后就是当前代码的一部分。

教学例子：

```cpp
#define RETURN_IF_BAD(condition) do { if (condition) { return -1; } } while (0)

int choose(bool bad) {
    RETURN_IF_BAD(bad);
    return 0;
}
```

bad 为 true 时，展开的 return 直接退出 choose。Python 中普通 helper 函数内部的 return，只退出 helper，不会直接退出调用者；所以不能把项目宏简单理解成 Python 包装函数。

`do { ... } while (0)` 在这里把宏包装成一条语句，代码体正常执行一次，不是无限重试。宏定义末尾的反斜线 `\` 让定义延续到下一行。

### hcomm 的两个检查宏

[log.h，CHK_PRT_RET 定义](../../../../src/legacy/ascend910/pub_inc/log.h#L171)，原文：

```cpp
#define CHK_PRT_RET(result, exeLog, retCode) \
    do {                                      \
        if (UNLIKELY(result)) {                         \
            exeLog;                           \
            return retCode;                   \
        }                                     \
    } while (0)
```

当条件为真，执行日志表达式并返回 retCode。阅读选择链时，先把 UNLIKELY 看作包着该真假条件，不能当成“只在不常见场景才检查”。

[CHK_RET 的实际定义](../../../../src/legacy/ascend910/pub_inc/log.h#L188) 包含状态检查与日志。下面是**省略日志、重新排版后的教学改写**，不是宏原文：

```cpp
// 仅示意 CHK_RET(call) 的核心控制流
do {
    HcclResult hcclRet = call;
    if (hcclRet != HCCL_SUCCESS) {
        return hcclRet;
    }
} while (0);
```

call 在这段核心逻辑中先被求值一次。失败就退出当前函数，成功才继续后续选择。实际定义还对 HCCL_E_AGAIN 的日志等级做区分，不能认为错误被捕获后仍继续运行。

例如 [SelectAlgfor910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L158)，原文调用：

```cpp
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
```

下一行并不保证会执行。先进入 AutoSelectAlgTypeLevel1 看返回值与它修改的字段，再带着成功/失败状态回到调用处。

日志中的 `%s`、`%u`、`%llu` 等是 printf 风格格式占位符；`HCCL_INFO(...)` 不是 Python f-string。先关注赋值、检查宏和 return，再用日志理解路径，不把日志文字当作最终算法证明。

### 注册宏如何帮助你找执行入口

[all_gather_operator.cc，注册位置](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L427)，原文：

```cpp
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER, AllGather, AllGatherOperator);
```

结合 [coll_alg_op_registry.h，REGISTER_OP 定义](../../../../src/legacy/ascend910/algorithm/impl/operator/registry/coll_alg_op_registry.h#L55)，这会生成一个静态注册变量，用对应类的创建函数注册 ALLGATHER。第三项才是这里的类名；不是在这行直接执行一次 AllGather 通信。

`REGISTER_EXEC`、`REGISTER_TEMPLATE` 同样要分别找各自定义与注册表，不假设所有注册宏都有完全一样的实现。注册帮助定位“名字对应哪个类”，并不能证明当前输入一定选到它。

**练习 9：** 如果 AutoSelectAlgTypeLevel1 返回非成功状态，包住它的 CHK_RET 会只退出辅助函数，还是退出 SelectAlgfor910B？后面的 Pipeline 检查一定执行吗？

## 8. 综合阅读：从 SelectAlg 追到模板

本节把前面的语法连成一次源码追踪。不复制整个函数；每步引用原文片段或明确的教学改写。暂时不用记住所有算法名。

### Python 对照：理解输出的两条通道

下列是**接口教学示意，不是 hcomm 的等价实现**：

```python
def select_alg(tag, param):
    # Python 可以把多项结果组织成元组返回
    status = 0
    alg_name = "example"
    new_tag = tag + alg_name
    return status, alg_name, new_tag
```

C++ 的真实 SelectAlg 接口则直接返回状态，另外通过两个字符串引用传出内容。不要照上面的示意假定每条真实路径都填好了三个结果。

### 第一步：读签名，列出输入与修改目标

打开 [AllGatherOperator::SelectAlg](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L38)。先写：tag/param 为只读引用输入；algName/newTag 为可修改字符串；返回值是状态。还需关注对象成员，因为函数并非只依赖四个形参。

源码连续片段：

```cpp
    if (userRankSize_ == 1) {
        algName = "AllGatherSingleExecutor";
        HCCL_INFO("[SelectAlg] AllGather SelectAlg is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
```

这是第一个提前返回。R=1 时写入 algName 后直接退出，不继续到设备分派与 newTag 拼接。必须从函数体判断哪些输出在哪些路径被写入。

### 第二步：找到 A2 的实际分支

入口先判断混合型号，再依次检查设备类型。对于非混合的 910B，才进入 SelectAlgfor910B。`else if` 链只执行第一个成立的分支，不是把所有条件独立运行一遍。

当辅助函数结束，入口用 CHK_PRT_RET 检查返回值；失败会退出入口函数。成功继续时，按模式生成 newTag，非图/非310P路径查 level1 名称，再追加 host/device 后缀。标签构造不能替代对辅助函数选型过程的分析。

### 第三步：在 A2 函数中先认数据和提前返回

打开 [SelectAlgfor910B](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L131)。开头计算 dataSize，读取配置，形成 isAivMode，检查是否为 AlltoAll 预处理调用。

下面是**删除日志后的教学改写**，保留这段选择的主要分支；完整前置判断必须回原函数读取：

```cpp
if (isAivMode && !isA2APreAG) {
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && dataSize <= AIV_ALL_GATHER_SMALL_SIZE) {
        algName = "AllGatherMeshAivSmallCountExecutor";
    } else {
        algName = "AllGatherMeshAivExecutor";
    }
    return HCCL_SUCCESS;
}
```

内层 else 不只有“数据太大”这一种原因：运行模式不是 OP_BASE 也会进入 else。选到这里就返回，后面的自动 level1、AHC、Pipeline 容量检查不会在这次辅助函数调用中继续执行。

### 第四步：没提前返回才检查后续修正

后续 OP_BASE 跨组路径可能自动选择 level1；配置为 AHC/AHC_BROKE 时再走专门辅助函数。Pipeline 的容量分支中，源码赋值：

```cpp
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
```

位置：[SelectAlgfor910B，Pipeline 容量修正](../../../../src/legacy/ascend910/algorithm/impl/operator/all_gather_operator.cc#L170)。读作“修改对象成员 algType_ 内的 algoLevel1 字段”，不是声明新变量，也不是立即修改 algName。之后的执行器分派会读取这个修改后的字段。

这里必须区分三个东西：输入配置、当前 level1 状态、最终 algName。条件判断与字段赋值的先后关系，会让最终结果不同于最初配置。

### 第五步：按名字找到类，不凭名字猜实现

传统 Mesh 的选择还有 OP_BASE/图模式、单组/跨组、数据大小等条件。不要把每个 Mesh 名称都当成同一实现，也不要把传统模板路径套到 AIV 执行器。

从最终 algName 查 [源码证据索引](source_index.md) 的 REGISTER_EXEC 对应类。以 **AllGatherMeshExecutor 已实际被选中**为前提，继续看 [CollAllGatherMeshExecutor::KernelRun](../../../../src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/coll_all_gather_mesh_executor.cc#L72)。源码连续片段：

```cpp
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("AllGather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("AllGather mesh: using nhr algo inter-server.");
```

这个片段在后续分支前结束，完整函数还判断 NHR_V1、NB 和兜底。拆开看：

- `unique_ptr<AlgTemplateBase>`：准备保存一个模板对象。
- `algType_.algoLevel1`：读取当前状态；`topoAttr_.serverNum`：读取另一个成员对象中的字段。
- `条件1 || (条件2 && 条件3)`：第一分支有拓扑例外，不能只看 level1 值。
- `AlgTemplateRegistry::Instance()`：调用作用域中的函数取得注册表实例；其结果再用 `.` 调用 GetAlgTemplate。
- `TemplateType::TEMPLATE_ALL_GATHER_RING`：带作用域的模板类型标识。
- `level1TempAlg = ...`：将取得的模板对象所有权交给智能指针。

因此，level1=NHR 也不是无条件命中第二分支；前面的单 server 异模块条件成立时，先选 Ring 模板。还应继续检查通信域规模、空指针检查、Prepare/RunTemplate 以及实际模板实现，确认这一层真的执行。

### 把一次具体输入带着走

**静态假设场景**：输入与资源合法；R=8、设备910B、非混合型号；AIV配置开启、单Mesh聚合组、FP32支持Copy、不是AlltoAll预处理；OP_BASE；count=179200，unitSize=4，isOnlyAiv=false。

1. 入口 R!=1，不提前选 Single；进入 A2 辅助函数。
2. dataSize=179200*4=716800 字节，正好是 700 KiB。
3. 满足该单组 AIV 资格，OP_BASE 且 `<=` 边界成立，algName 写入 AllGatherMeshAivSmallCountExecutor。
4. 辅助函数返回成功；不会继续到传统 Pipeline 检查。
5. 入口继续做自身的状态检查与标签处理。实际模板/AIV内核要跟着这个最终执行器追，不进入普通 MeshExecutor 的 KernelRun。

这是对条件的静态推导，不是硬件运行证明，也不是完整公开 API 参数合法性验证。

**练习 10：** 在上述场景只把 count 改成179201，dataSize是多少，算法名如何变化？再考虑 AllGatherMeshExecutor 内部：level1=NHR，但 isDiffDeviceModule=true、serverNum=1，片段里哪个分支先命中？

## 9. 练习答案：先说结果，再说控制流

1. dataSize=32 字节。没有初始化的局部基础类型变量不能当作0；读取这种未初始化值不是合法的默认赋值行为。
2. false。第一项已经失败，&&短路；OnlyAIV只处于最后括号内，不能让前面的false变true。
3. x=4.0，因为8/3先得到整数2。使用8.0会先做浮点除法，最终约5.333333，存在浮点表示误差。
4. 第一次仍为Ring，第二次为Mesh。引用属性来自函数签名；传实参时不要求写地址运算符。按值字符串的修改作用在独立对象上，引用版本改原字符串。
5. 不能。R=1分支写algName后提前返回，没有在这条路径写newTag；调用者原有的newTag内容不会凭空更新。成功状态不等于所有输出参数每条路径都被赋值。
6. state、alias和ptr访问同一个对象，最后一次赋值为3。空指针不能用于这个成员访问；也要避免非空但已失效的悬空指针。
7. select是虚函数，base引用实际的Derived对象，所以普通调用进入Derived实现。构造函数的基类初始化列表先执行；空函数体不表示没有初始化。
8. 不能读取end的second，它不是一个可读取的元素。错误宏确保未找到时先返回。默认unique_ptr为空，不包含已经创建的模板对象。
9. 辅助函数已返回它的错误值；CHK_RET展开后的return还会退出当前SelectAlgfor910B。后续Pipeline检查不会执行；上层SelectAlg又会检查收到的错误状态。
10. dataSize=716804，即700 KiB+4；其他假设不变，选AllGatherMeshAivExecutor。独立的传统Mesh模板问题中，拓扑例外让第一个if先成立，选Ring模板而非NHR。这两题属于不同执行路径，不要串成同一次运行。

## 10. 一页速查：读到符号先问它在哪里

| 写法 | 先翻译成一句话 |
|---|---|
| `T value = expression;` | 声明T类型变量，并用表达式初始化 |
| `T value;` | 声明；默认值取决于类型与初始化规则 |
| `const T& value` | 通过只读引用访问对象，不是深度冻结 |
| `T& value` | 对原对象的引用，赋值可能修改调用者 |
| `T* ptr` / `&value` / `*ptr` | 指针声明 / 取地址 / 解引用 |
| `a & b` / `a && b` | 按位与 / 短路逻辑与 |
| `obj.field` / `ptr->field` | 对象成员 / 所指对象成员 |
| `Type::name` | 作用域中的名字，可能是枚举项或函数 |
| `condition ? a : b` | 条件为真选a，否则选b |
| `constexpr T name = ...` | 此处声明编译期常量 |
| `static_cast<T>(x)` | 显式转换，不保证范围/地址合法 |
| `auto x = ...` / `auto& x = ...` | 推导值类型 / 推导引用类型 |
| `std::vector<T>` | 保存T元素的动态数组 |
| `std::unique_ptr<T>` | 独占管理对象的智能指针，可能为空 |
| `for (const auto& x : items)` | 依次以只读引用访问元素 |
| `virtual` / `override` | 虚函数接口 / 编译器核对覆盖关系 |
| `Class::Class(...) : Base(...)` | 构造函数及基类初始化 |
| `return result;` | 立即结束当前函数 |
| `CHK_RET(...)` | 去看宏定义；失败可能退出当前函数 |
| `REGISTER_*` | 去看注册宏与注册表，定位名字到实现 |
| `(void)param;` | 这里常用于表明有意未使用该参数，不是修改参数 |
| `reinterpret_cast<T*>(p)` | 改变地址的类型解释，不创建对象、不证明内存合法 |

## 11. 读一个函数的固定步骤

1. **找接口**：返回类型、所属类、形参类型、引用/const/指针；同名函数是否重载。
2. **列修改目标**：返回值、输出引用、对象成员、指针所指数据分别记清。
3. **先标退出点**：显式return、CHK_RET/CHK_PRT_RET等宏、前置错误检查。
4. **拆条件**：按括号和短路顺序，保留<=与<的区别；不要把else只归因于某一个条件。
5. **核对大小**：count还是bytes、总量还是每rank/每轮、整数除法还是浮点运算。
6. **追辅助函数**：读取真实声明和实现，把它修改的字段与返回状态带回调用处。
7. **找最终实现**：名字→注册类→Orchestrate/RunLoop/KernelRun→模板或AIV内核；必要时沿基类读公共逻辑。
8. **带一个输入重走**：写下假设和中间变量，指出哪一行提前返回；区分静态推导与实际运行。

可用编辑器“转到定义/查找引用”，也可在仓库根目录运行：

```sh
rg -n 'SelectAlgfor910B' src/legacy/ascend910/algorithm
rg -n 'AllGatherMeshExecutor' src/legacy/ascend910/algorithm
rg -n '#define CHK_RET' src/legacy/ascend910/pub_inc
```

搜索结果只是导航。定义、调用、日志文字、注册项是不同证据，要逐个区分。

现在可以回到 [AllGather 学习页](02_allgather.md)，读到陌生语法时查本讲义；遇到算法本身的疑问，再查 [公共机制](01_common.md) 和 [内部模板切换](07_internal_templates.md)。

## 12. 目前可以暂缓的内容

复杂模板元编程、完美转发、智能指针实现细节、手动new/delete、异常安全的完整体系、锁与内存模型、编译链接全流程、算子内核编程，都可以等对应问题出现再深入。遇到lambda、函数指针或复杂转换，先认出它在当前位置负责什么，再按需求学习；不要为了读一条选择条件先通读整本C++教材。

暂缓不等于忽略：若控制流跨到这些机制，或开始编写/修改生产C++代码，需要补充相应知识。本讲义提供的是阅读起点。
