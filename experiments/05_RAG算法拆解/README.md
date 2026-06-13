# RAG算法拆解说明

这个目录是为了答辩阅读重新整理出来的“论文主线拆解版”，不会改动原始实现目录 [04_算法实现](/root/Velo/experiments/04_算法实现)。

## 目录作用

原始实现更强调“可运行”和“多实验共用”，因此很多功能会分散在 `common.py`、`adaptive_evidence.py`、`pipeline.py`、`metrics.py` 之间。  
这个目录的目标相反：优先保证答辩时“按论文流程找代码”，也就是：

1. 共享检索主干看一个文件；
2. 分项重排看一个文件；
3. 覆盖取证看一个文件；
4. 按题作答看一个文件；
5. 主流程串联看一个文件；
6. 数据处理和指标评测也各自独立。

## 文件对应关系

1. [m00_公共基础.py](/root/Velo/experiments/05_RAG算法拆解/m00_公共基础.py)
   放通用常量、清洗、分词、相似度和底层服务调用，不放论文核心创新逻辑。
2. [m01_共享检索主干.py](/root/Velo/experiments/05_RAG算法拆解/m01_共享检索主干.py)
   对应 `Dense + BM25 -> RRF -> Rerank -> 证据单元构造`。
3. [m02_分项重排.py](/root/Velo/experiments/05_RAG算法拆解/m02_分项重排.py)
   对应论文里的“分项重排”。
4. [m03_覆盖取证.py](/root/Velo/experiments/05_RAG算法拆解/m03_覆盖取证.py)
   对应论文里的“覆盖取证”。
5. [m04_按题作答.py](/root/Velo/experiments/05_RAG算法拆解/m04_按题作答.py)
   对应论文里的“按题作答”。
6. [m05_主流程串联.py](/root/Velo/experiments/05_RAG算法拆解/m05_主流程串联.py)
   把主线算法怎样串起来单独放在一处。
7. [m06_数据处理.py](/root/Velo/experiments/05_RAG算法拆解/m06_数据处理.py)
   对应 RGB / CRUD 数据载入与语料构造。
8. [m07_指标评测.py](/root/Velo/experiments/05_RAG算法拆解/m07_指标评测.py)
   对应复杂问答指标计算与结果汇总。
9. [分项重排.md](/root/Velo/experiments/05_RAG算法拆解/分项重排.md)
   直接按论文口径解释公式、指标含义与代码对应。
10. [覆盖取证.md](/root/Velo/experiments/05_RAG算法拆解/覆盖取证.md)
   直接按论文口径解释公式、指标含义与代码对应。
11. [按题作答.md](/root/Velo/experiments/05_RAG算法拆解/按题作答.md)
   直接按论文口径解释任务识别、并列要求估计与提示词组织。

## 使用原则

1. 这里保留的是“论文主线最值得讲的实现”，不是把所有证伪路线全量重抄一遍；
2. 为了便于答辩阅读，部分文件会保留少量通用依赖导入，但核心算法代码都会放在对应主题文件里；
3. 如果需要看完整、可运行、全分支版本，仍以 [04_算法实现](/root/Velo/experiments/04_算法实现) 为准。
