# Improvements to BridGE 3.0 (BridGE-WGS, BridGE-BB) for All of Us data

This file identifies areas of the AoU pipeline that could be improved, or at the very list this file will track changes/decisions that were made that could be improved.

- Allows missing variant data in BridGE
  - The easiest data format to use in AoU (Plink2's pgen) has variant entries that are set to missing if it did not pass quality filtering.
  - In order to fill those in, an expensive operation to extract either from vcf sharded files or from a Hail Matrix Table has to be done, but this is estimated to cost quite a bit and increases the complexity of the pipeline.
  - Ideally, your input data should be statistically phased and/or imputed with a modern imputation software/service.
  - Now, BridGE 3.0 has been configured so that missing data can be present
  - Therefore, missing data simply will not contribute any calculations (treated as a zero value entry).
  - This is a change from BridGE 2.0 that would use the mode to impute per variant. This would be fine for a set of samples as long as they are ancestrally similar, which is not the case for AoU data
    - It is noted that you could filter by predicted ancestry, impute with the mode per variant, then recombine the date
    - However, this functionality allows for flexibility and for someone to determine the extent that imputation would have on BridGE analysis.
