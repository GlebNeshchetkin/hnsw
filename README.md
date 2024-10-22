# HNSW

## Modifications
> Some candidates in neighborhood_construction are randomly selected (lower lever - less randomly selected candidates) <br>

> Ensure diversity by considering angular difference between candidates in neighborhood_construction <br>

> Start point is selected as median point

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

## SIFT 10K M0=32, M=16, EF = [0,...,100]
> Recall == 1.0 for Modified algorithm achieved at EF=41 <br>
> Recall == 1.0 for Original algorithm achieved at EF=54 <br>
![download](https://github.com/user-attachments/assets/dc4b14f0-74c4-49d3-b1fe-e94da9ea6692)
