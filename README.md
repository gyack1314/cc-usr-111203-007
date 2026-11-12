# 缺测海温场的面积加权模态与外期重建

这是独立的气候资料Python项目。`data/`保存合成资料，`climate_data.io`负责严格读取、公历日期和来源字节摘要，`climate_data.domain`提供现有描述统计，`climate_data.linalg`是纯标准库线性代数（Householder QR、加权最小二乘、Jacobi特征分解、面积加权正交化），`climate_data.lowrank`实现把缺测视为未观测的面积加权低秩模态分析，`climate_data.cli`将摘要与模态结果输出为JSON。所有观测值、站点、人群、情景与金额都是虚构。字段和研究约定见[资料约定](docs/data-contract.md)。

当前Python 3.11及以上可直接使用，没有第三方Python依赖。项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q climate_data tests
python -m climate_data.cli inspect data/normal.json
python -m climate_data.cli modes data/normal.json
```

`inspect`显示当前原始摘要，不覆盖资料；`modes`输出面积加权低秩模态、选秩过程与独立外期评估。`invalid-`前缀文件为不合法资料，其他变体使用相同字段保留业务边界。来源字节SHA-256随输出标明，后续结果应同时标明输入身份、量纲和适用范围。

## 低秩模态的口径

- 矩阵按 time×cell 排列，每格公开正 `area`，目标在观察位置最小化 `sum area_j*(X_tj-mean_j-U_tV_j)^2`，固定秩 r，约束 `V diag(area) V^T=I`。缺测位不进入目标、不填零后当真值。
- 训练窗内每格均值只用该窗已观察值；全空列或有效数少于 `min_train_observations` 的列排除并保留映射，未观察格绝不借未来填均值。
- 交替最小二乘（行分数/列载荷轮流加权最小二乘）配合面积加权正交化，多起点择优。结果只声称**局部拟合**，不伪称全域最优；每次拟合列出观察目标、停止原因与目标是否下降。
- 可识别性显式打标，不用ridge制造看似确定的值：整行缺测不估时间分数；观察二部图不连通、行内已观察载荷列不足秩、列观察行不足秩或秩超过维数时，标记不可识别并跳过该秩。
- 模态按时间分数能量排序，按公开符号规则（面积加权绝对载荷最大的格取正，同值取最小cell_id）定向；近重根展示等价子空间与投影矩阵，旋转后逐列不同不判为不同结果。
- 选秩用时间尾段验证：均值、模态与一切填补只在更早训练窗求得；验证行仅在其固定 context 格上求分数，在完全不重叠的 target 格上计面积加权误差，target 值不参与该行投影。target 为空或 context 秩不足的行单列；没有可评分行就不选秩；同分取较小 r。选完在全部开发窗重拟合，再对独立外期用同一 context/target 规则输出。
- 输出保存训练/验证/开发/外期日期与观察格掩码。改动外期 target 不影响载荷或分数（有测试核对）。

容器使用固定基础镜像，可运行`docker build -t climate-material .`，随后`docker run --rm climate-material`检查资料。`docker run --rm climate-material python -m unittest discover -s tests -v`执行已有检查。运行不依赖远端账号或实时服务。项目不产出海图图片或学习模型文件。
