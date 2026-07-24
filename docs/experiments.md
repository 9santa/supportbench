# Retrieval experiments

Все метрики вычисляются на одинаковом наборе запросов. Recall@k — среднее значение recall по запросам.

Query count: 200

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 3.400 | 0.7825 | 0.9600 | 0.5907 |
| BM25 | 0.3525 | 0.8025 | 0.9525 | 0.6005 |



# Проверка гипотезы. Абляции по параметру k1 и b для BM25 retriever.
Dataset: dev-v1
Documents: текущий documents.jsonl
Queries: текущий dev split
Tokenizer: regex lowercase, без стоп-слов и лемматизации
Indexed fields: title + text
Metrics: Recall@1, Recall@3, Recall@5, MRR
Tie-breaking: score DESC, doc_id ASC
