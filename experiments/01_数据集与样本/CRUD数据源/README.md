# CRUD 数据源

这里保留论文实验实际读取的 `CRUD` 原始数据文件。

## 保留内容

- `crud_rag_official/data/crud_split/split_merged.json`
  - 官方切分文件
- `crud_rag_official/data/80000_docs/`
  - 80k 文档语料
- `crud_rag_official/data/crud/`
  - 原始压缩包备份

## 与论文相关的子集

- 论文复杂问答部分主要使用 `CRUD子样本（共1594条）`
- 具体由下列两部分组成：
  - `questanswer_2docs = 797`
  - `questanswer_3docs = 797`

## 代码读取入口

- `../../04_算法实现/retrieval_pipeline/datasets.py`

## 说明

- 当前目录只作为数据源保留
- 不在此目录放实验脚本与结果文件
