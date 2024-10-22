# HNSW

## SIFT 10K Comparison

| EF  | M0  | M   | Recall (Original HNSW) | Recall (Modified HNSW) | Avg Calc (Original HNSW) | Avg Calc (Modified HNSW) |
| --- | --- | --- | ---------------------- | ---------------------- | ------------------------ | ------------------------ |
| 64  | 32  | 16  | 0.99                   | **1**                  | 649.56                   | **608.95**               |
| 64  | 64  | 32  | 0.99                   | **1**                  | 906.21                   | **858.46**               |
| 64  | 16  | 8   | 0.988                  | **0.998**              | 460.48                   | **447.98**               |
| 64  | 16  | 16  | 0.994                  | **0.998**              | 457.58                   | **446.68**               |

## SIFT 1M Comparison

| EF  | M0  | M   | Recall (Original HNSW) | Recall (Modified HNSW) | Avg Calc (Original HNSW) | Avg Calc (Modified HNSW) |
| --- | --- | --- | ---------------------- | ---------------------- | ------------------------ | ------------------------ |
| 64  | 16  | 16  | 0.91                   | 0.91                   | 836.4                    | **833.87**               |

![download](https://github.com/user-attachments/assets/dc4b14f0-74c4-49d3-b1fe-e94da9ea6692)
