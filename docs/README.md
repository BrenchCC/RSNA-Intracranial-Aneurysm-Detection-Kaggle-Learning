# Docs Index

`docs/` 目录按“认知建立 -> 方案设计 -> 实验结果 -> 深入研究”组织。下面不是简单列文件，而是给出每一份文档的基本指引，帮助你判断这份文档什么时候看、重点看什么、读完后能获得什么。

## 总体阅读建议

如果你是第一次进入本项目，建议按下面顺序：

1. 先读任务背景与整体方法。
2. 再读分阶段实施方案与推理逻辑。
3. 然后查看单模型与集成结果。
4. 最后进入综合总结和深入研究文档。

## `overview/`

### [overview/introduction.md](overview/introduction.md)

基本指引：

- 适合在项目最开始阅读。
- 重点是建立对竞赛背景、医学问题、数据形式和任务定义的基础认知。
- 如果你以后需要做汇报、面试介绍、项目开场说明，这份文档最有用。
- 读完后，你应该能回答“这个比赛到底在做什么，为什么有意义，输入输出是什么”。

### [overview/recording.md](overview/recording.md)

基本指引：

- 适合在已经知道任务背景后阅读。
- 重点是从论文式视角理解整体 pipeline，包括血管分割、ROI 提取、ROI 分类和最终聚合。
- 这份文档更偏“方案结构化表达”，适合用来建立方法框架。
- 读完后，你应该能清楚描述整个多阶段检测流程及每一阶段的作用。

### [overview/training.md](overview/training.md)

基本指引：

- 适合在读完 `recording.md` 之后继续看。
- 重点是区分“哪些阶段真的训练了模型，哪些阶段只是规则计算”。
- 如果你想快速理解项目训练成本、模型数量和训练边界，这份文档很关键。
- 读完后，你应该能准确说明项目里真正训练的是哪些模型，以及 Stage 3 为什么不属于训练阶段。

### [overview/FirstSummary_CN.md](overview/FirstSummary_CN.md)

基本指引：

- 适合在前面几份 overview 文档之后阅读。
- 重点是把项目整理成一套完整的中文战略蓝图，覆盖数据理解、预处理、分阶段建模和最终推理。
- 这份文档适合作为“中文版总方案说明书”。
- 读完后，你应该能从全局上理解为什么要做分割、定位、补丁分类三段式设计。

### [overview/FirstSummary.md](overview/FirstSummary.md)

基本指引：

- 内容和 `FirstSummary_CN.md` 基本对应，是英文版本。
- 适合需要英文表达、对外沟通、写英文说明或做英文汇报时参考。
- 如果中文版本已经读过，可以把它当作术语对照材料使用。

### [overview/deep-insight.md](overview/deep-insight.md)

基本指引：

- 适合在完成基础复现理解之后再读。
- 重点不再是“方案怎么搭”，而是“实验结果说明了什么规律”。
- 这份文档会把模型表现、架构比较、训练策略和集成结论系统整理出来。
- 读完后，你应该能说清楚本项目最重要的研究结论，例如为什么小模型更强、为什么 SE-ResNet 表现最好、为什么 5-6 个模型的集成最优。

## `setup/`

### [setup/ProjectSetup_CN.md](setup/ProjectSetup_CN.md)

基本指引：

- 适合在准备真正动手时阅读。
- 重点是把理论方案拆成可执行的阶段任务，包括环境、EDA、可视化、分割、定位、补丁分类和最终提交。
- 这份文档最像项目实施手册。
- 读完后，你应该知道项目该按什么顺序落地，以及每个阶段该产出什么中间结果来做验证。

### [setup/ProjectSetup.md](setup/ProjectSetup.md)

基本指引：

- 与 `ProjectSetup_CN.md` 对应，是英文版本。
- 适合做英文技术记录、术语对照，或者需要英文任务说明时使用。
- 如果你的主要工作语言是中文，这份文档可以作为补充，而不是首读材料。

### [setup/InferenceSetup_CN.md](setup/InferenceSetup_CN.md)

基本指引：

- 适合在理解训练阶段后阅读。
- 重点解释一个常见疑问：为什么最终只提交分类概率，却还要使用定位数据。
- 它会从“大海捞针”问题出发，说明定位信号如何帮助分类模型聚焦候选区域。
- 读完后，你应该能清楚解释定位数据在训练中的角色，以及它如何转化为最终推理流程。

### [setup/InferenceSetup.md](setup/InferenceSetup.md)

基本指引：

- 与 `InferenceSetup_CN.md` 对应，是英文版本。
- 适合用来做英文表达或作为中英文术语对照。
- 如果你已经理解中文版本，这份文档更多是辅助材料。

## `results/`

### [results/MODEL_DATABASE_CN.md](results/MODEL_DATABASE_CN.md)

基本指引：

- 适合在完成训练后查看，或者做模型选型时查看。
- 重点是单模型维度的对比，包括不同架构家族、超参数和训练策略的表现差异。
- 如果你在问“下一轮应该优先训练哪个模型”，先看这份文档。
- 读完后，你应该能提炼出单模型层面的结论，比如哪些结构值得继续投入，哪些路线应该停止。

### [results/MODEL_DATABASE.md](results/MODEL_DATABASE.md)

基本指引：

- 与 `MODEL_DATABASE_CN.md` 对应，是英文版本。
- 适合做英文总结、对外交流或保留英文实验档案。

### [results/ENSEMBLE_RESULTS_CN.md](results/ENSEMBLE_RESULTS_CN.md)

基本指引：

- 适合在单模型结果已经稳定后阅读。
- 重点是比较不同集成策略，包括模型数量、简单平均、加权平均、TTA 和多样性组合。
- 如果你在问“最终上线或提交该用哪套集成”，这份文档最直接。
- 读完后，你应该能判断哪些集成策略真正有效，哪些复杂设计并不值得。

### [results/ENSEMBLE_RESULTS.md](results/ENSEMBLE_RESULTS.md)

基本指引：

- 与 `ENSEMBLE_RESULTS_CN.md` 对应，是英文版本。
- 适合做英文实验记录、对照或对外展示。

## 推荐阅读路径

### 路线 1：快速理解项目

1. [overview/introduction.md](overview/introduction.md)
2. [overview/recording.md](overview/recording.md)
3. [overview/training.md](overview/training.md)
4. [setup/InferenceSetup_CN.md](setup/InferenceSetup_CN.md)

### 路线 2：准备动手复现

1. [overview/FirstSummary_CN.md](overview/FirstSummary_CN.md)
2. [setup/ProjectSetup_CN.md](setup/ProjectSetup_CN.md)
3. [setup/InferenceSetup_CN.md](setup/InferenceSetup_CN.md)

### 路线 3：准备做实验决策

1. [overview/deep-insight.md](overview/deep-insight.md)
2. [results/MODEL_DATABASE_CN.md](results/MODEL_DATABASE_CN.md)
3. [results/ENSEMBLE_RESULTS_CN.md](results/ENSEMBLE_RESULTS_CN.md)
