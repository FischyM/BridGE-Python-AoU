# Comparison of time to run BridGE 2.0 vs BridGE 3.0

Preprocessing Changes

| step/file             | old       | new       | effect             |
| --------------------- | --------- | --------- | ------------------ |
| 1000G file size       | 44 GB     | 4.6 GB    | 90% size reduction |
| example file size     | 600 MB    | 66 MB     | 90% size reduction |
| check populations.sh  | 5 min     | 2.5 min   | 2x speed up        |
| data_removeoutlier.sh | 5 sec     | 2 sec     | 2.5x speed up      |
| preprocessgwas.sh     | 1 min     | 1.25 min  | 0.75x speed down   |
| example N samples     | 328       | 318       | 10 less            |
| example N SNPs        | 12745     | 28217     | 15472 more         |

BridGE Changes

| Module             | old               | new                   | speed up |
| -------------------| ----------------- | --------------------- | -------- |
| DataProcess        | 11 min            | 4 min                 | 2.74x    |
| ComputeInteraction | 1 hr per network  | 6 min per network     | 10x      |
| ComputeStats       | 3 hr per network  | 20 min per network    | 9x       |
| ComputeFDR         | 4 hr              | 3 sec                 | 4800x    |
| Summarize          | 5 min             | 30 sec                | 10x      |
