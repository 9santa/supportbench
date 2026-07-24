# Retrieval experiments

Все метрики вычисляются на одинаковом наборе запросов. Recall@k — среднее значение recall по запросам.

Query count: 200

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 3.400 | 0.7825 | 0.9600 | 0.5907 |
| BM25 | 0.3525 | 0.8025 | 0.9525 | 0.6005 |
