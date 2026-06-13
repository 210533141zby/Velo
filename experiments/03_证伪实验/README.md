# 03_证伪实验

这一层对应论文第 3.3 节及相关补充对比，重点用于回答“为什么主线方案这样定”的问题。

## 与论文小节的对应关系

- `3.3.1 主线替代路线`
  - notebook：`scripts/运行_主线替代路线对比.ipynb`
  - 脚本：`scripts/运行_主线替代路线对比.py`
  - 结果：`results/01_主线替代路线/`
- `3.3.2 生成侧补充对比实验`
  - notebook：
  - `scripts/运行_生成侧提示路线对比.ipynb`
  - `scripts/运行_支持修补对比.ipynb`
  - 脚本：
  - `scripts/运行_生成侧提示路线对比.py`
  - `scripts/运行_支持修补对比.py`
  - 结果：`results/02_生成侧补充对比/`
- `3.3.3 检索侧补充对比实验`
  - notebook：`scripts/运行_检索侧补充对比.ipynb`
  - 脚本：`scripts/运行_检索侧补充对比.py`
  - 结果：`results/03_检索侧补充对比/`
- `TRACE 方法补充对比`
  - notebook：`scripts/运行_TRACE方法对比.ipynb`
  - 脚本：`scripts/运行_TRACE方法对比.py`
  - 结果：`results/05_TRACE方法对比/`

## 补充材料

- `scripts/运行_回答模型对比.py`
  - 用于固定检索链路后比较回答模型
  - 主要服务答辩问答，不作为 3.3 节主表的核心结论
- `scripts/运行_回答模型对比.ipynb`
  - 回答模型对比的查阅版 notebook
- `文字说明/`
  - 放论文 3.3 节相关的整理说明

## 建议展示路径

1. `results/01_主线替代路线/双片段路线_20260517/`
2. `results/01_主线替代路线/候选证据路线_20260517/`
3. `results/01_主线替代路线/结构化生成路线_20260517/`
4. `results/02_生成侧补充对比/`
5. `results/03_检索侧补充对比/`
6. `results/05_TRACE方法对比/`

## 代码位置

- 主链路执行：`../04_算法实现/retrieval_pipeline/pipeline.py`
- 分项重排与覆盖取证相关逻辑：`../04_算法实现/retrieval_pipeline/adaptive_evidence.py`
- 指标计算：`../04_算法实现/retrieval_pipeline/metrics.py`

## 说明

- `.ipynb` 为查阅版，适合现场展示代码和结果
- `.py` 为命令行运行版，适合重新复现实验
