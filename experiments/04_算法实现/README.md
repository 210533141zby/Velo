# 04_算法实现

这里放的是论文 RAG 实验实际调用的实现代码，不是演示版伪代码。

## 关键文件定位

- `retrieval_pipeline/datasets.py`
  - 数据读取
  - 负责读取 `CRUD` 与 `RGB` 数据
- `retrieval_pipeline/pipeline.py`
  - 主实验链路
  - 负责 Dense + BM25 -> RRF -> Rerank -> 证据组织 -> 作答 的整体调度
- `retrieval_pipeline/adaptive_evidence.py`
  - 证据加工与答案组织
  - 这里能找到论文中的：
  - 分项重排
  - 覆盖取证
  - 按题作答
- `retrieval_pipeline/metrics.py`
  - 指标计算
  - 负责消融实验、证伪实验中用到的准确率、命中率、综合质量指标等
- `retrieval_pipeline/common.py`
  - 通用组件
  - 包括嵌入、重排、提示词调用和缓存工具

## 论文方法与代码的对应关系

- “分项重排”
  - 主要看 `adaptive_evidence.py` 中与 `order_evidence_units_for_query*` 相关的函数
- “覆盖取证”
  - 主要看 `adaptive_evidence.py` 中证据候选压缩、覆盖控制、槽位组织相关函数
- “按题作答”
  - 主要看 `pipeline.py` 中 `answer_prompt_style`、复杂度判断和最终路由逻辑

## 说明

- 包名仍保留为 `retrieval_pipeline`
- 这是为了保证实验脚本导入路径稳定
- 对外展示时，按本 README 的中文说明来找代码位置即可
