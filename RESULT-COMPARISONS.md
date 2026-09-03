# Comparison of time to run BridGE 2.0 vs BridGE 3.0

| Module                | old              | new                | speed up/difference |
| ------------------    | ---------------- | ------------------ | ------------------- |
| 1000G file size       | 44 GB            | 4.6 GB             | x0.9 size reduction |
| example file size     | 600 MB           | 66 MB              | x0.9 size reduction |
| check populations.sh  | 
| data_removeoutlier.sh | 
| preprocessgwas.sh     | 
| DataProcess           | 11 min           | 4 min              | 2.74x               |
| ComputeInteraction    | 1 hr per network | 6 min per network  | 10x                 |
| ComputeStats          | 3 hr per network | 20 min per network | 9x                  |
| ComputeFDR            | 4 hr             | 3 sec              | 4800x               |
| Summarize             | 5 min            | 30 sec             | 10x                 |
