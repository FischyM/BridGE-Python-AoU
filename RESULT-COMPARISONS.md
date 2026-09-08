# Comparison of time to run BridGE 2.0 vs BridGE 3.0

Preprocessing Changes

| step/file             | old    | new      | effect             |
| --------------------- | ------ | -------- | ------------------ |
| gg1000G file size     | 44 GB  | 4.6 GB   | 90% size reduction |
| example file size     | 600 MB | 66 MB    | 90% size reduction |
| check populations.sh  | 5 min  | 2.5 min  | 2x speed up        |
| data_removeoutlier.sh | 5 sec  | 2 sec    | 2.5x speed up      |
| preprocessgwas.sh     | 1 min  | 1.25 min | 0.75x speed down   |
| example N samples     | 328    | 318      | 10 less            |
| example N SNPs        | 12745  | 28217    | 15472 more         |

BridGE Changes (old uses the same genotype file and filtered gene sets to be comparable)

| Module             | old            | new                 | speed up |
| ------------------ | -------------- | ------------------- | -------- |
| DataProcess        | 13 min         | 1 min               | 13x      |
| ComputeInteraction | hr per network | 3 min per network   |          |
| ComputeStats       | hr per network | 2.6 min per network |          |
| ComputeFDR         | hr             | 25 sec              |          |
| Summarize          | min            | 1 min               |          |
