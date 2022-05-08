# REF M1

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       76 |       77
| FFT 1 |       76 |       79
| FFT 2 |       85 |      102
| FFT 3 |      122 |      128
| FFT 4 |      167 |      177
| FFT 5 |      249 |      261
| FFT 6 |      412 |      424
| FFT 7 |      763 |      847
| FFT 8 |     1634 |     1811
| FFT 9 |     3649 |     3930
| FFT 10 |     8010 |     8541


| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |     6613 |     6448
| NTT 10 |    13803 |    13335


| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |       75
| poly_add |        1 |       76
| poly_add |        2 |       83
| poly_add |        3 |       86
| poly_add |        4 |       95
| poly_add |        5 |      113
| poly_add |        6 |      155
| poly_add |        7 |      247
| poly_add |        8 |      398
| poly_add |        9 |      653
| poly_add |       10 |     1165

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |       75
| poly_sub |        1 |       76
| poly_sub |        2 |       83
| poly_sub |        3 |       86
| poly_sub |        4 |       95
| poly_sub |        5 |      109
| poly_sub |        6 |      155
| poly_sub |        7 |      248
| poly_sub |        8 |      396
| poly_sub |        9 |      653
| poly_sub |       10 |     1165

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |       75
| poly_neg |        1 |       76
| poly_neg |        2 |       82
| poly_neg |        3 |       86
| poly_neg |        4 |       93
| poly_neg |        5 |      109
| poly_neg |        6 |      149
| poly_neg |        7 |      237
| poly_neg |        8 |      388
| poly_neg |        9 |      643
| poly_neg |       10 |     1160

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |       76
| poly_adj_fft |        1 |       76
| poly_adj_fft |        2 |       77
| poly_adj_fft |        3 |       82
| poly_adj_fft |        4 |       77
| poly_adj_fft |        5 |       78
| poly_adj_fft |        6 |       83
| poly_adj_fft |        7 |       90
| poly_adj_fft |        8 |      183
| poly_adj_fft |        9 |      157
| poly_adj_fft |       10 |      275

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |       76
| poly_mul_fft |        1 |       78
| poly_mul_fft |        2 |       78
| poly_mul_fft |        3 |       83
| poly_mul_fft |        4 |       87
| poly_mul_fft |        5 |       95
| poly_mul_fft |        6 |      115
| poly_mul_fft |        7 |      162
| poly_mul_fft |        8 |      336
| poly_mul_fft |        9 |      483
| poly_mul_fft |       10 |      756

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |       76
| poly_invnorm2_fft |        1 |       85
| poly_invnorm2_fft |        2 |       85
| poly_invnorm2_fft |        3 |       85
| poly_invnorm2_fft |        4 |       90
| poly_invnorm2_fft |        5 |      100
| poly_invnorm2_fft |        6 |      132
| poly_invnorm2_fft |        7 |      189
| poly_invnorm2_fft |        8 |      304
| poly_invnorm2_fft |        9 |      457
| poly_invnorm2_fft |       10 |      756

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |       76
| poly_mul_autoadj_fft |        1 |       76
| poly_mul_autoadj_fft |        2 |       77
| poly_mul_autoadj_fft |        3 |       78
| poly_mul_autoadj_fft |        4 |       80
| poly_mul_autoadj_fft |        5 |       86
| poly_mul_autoadj_fft |        6 |       96
| poly_mul_autoadj_fft |        7 |      125
| poly_mul_autoadj_fft |        8 |      264
| poly_mul_autoadj_fft |        9 |      422
| poly_mul_autoadj_fft |       10 |      758

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |       76
| poly_LDL_fft |        1 |       99
| poly_LDL_fft |        2 |       99
| poly_LDL_fft |        3 |      105
| poly_LDL_fft |        4 |      122
| poly_LDL_fft |        5 |      157
| poly_LDL_fft |        6 |      217
| poly_LDL_fft |        7 |      323
| poly_LDL_fft |        8 |      596
| poly_LDL_fft |        9 |     1019
| poly_LDL_fft |       10 |     1890

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |       76
| poly_LDLmv_fft |        1 |      103
| poly_LDLmv_fft |        2 |       99
| poly_LDLmv_fft |        3 |      108
| poly_LDLmv_fft |        4 |      122
| poly_LDLmv_fft |        5 |      156
| poly_LDLmv_fft |        6 |      219
| poly_LDLmv_fft |        7 |      324
| poly_LDLmv_fft |        8 |      565
| poly_LDLmv_fft |        9 |      983
| poly_LDLmv_fft |       10 |     1848

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |       76
| poly_split_fft |        1 |       76
| poly_split_fft |        2 |       82
| poly_split_fft |        3 |       86
| poly_split_fft |        4 |       93
| poly_split_fft |        5 |      109
| poly_split_fft |        6 |      140
| poly_split_fft |        7 |      199
| poly_split_fft |        8 |      295
| poly_split_fft |        9 |      486
| poly_split_fft |       10 |      891

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       76
| poly_merge_fft |        1 |       76
| poly_merge_fft |        2 |       83
| poly_merge_fft |        3 |       96
| poly_merge_fft |        4 |      104
| poly_merge_fft |        5 |      123
| poly_merge_fft |        6 |      161
| poly_merge_fft |        7 |      228
| poly_merge_fft |        8 |      336
| poly_merge_fft |        9 |      575
| poly_merge_fft |       10 |     1016



| degree |  kg(kc)   | ek(kc) |  sd(kc) | sdc(kc) |  st(kc) | stc(kc) |  vv(kc) | vvc(kc)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512: | 15627.24 |  154.31 |  655.72 |  698.30 |  436.83 |  480.04  |  43.41  |  91.02
| 1024: | 48677.96 |  318.49 | 1309.62 | 1395.65 |  863.20 |  952.89  |  88.85  | 182.97

keygen in milliseconds, other values in microseconds

| degree | kg(ms) |  ek(us) |  sd(us) | sdc(us) |  st(us) | stc(us) |  vv(us) | vvc(us)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512: |    5.25 |   48.23 |  207.26  | 217.71  | 136.16 |  152.52 |   13.82 |   28.42
| 1024: |   16.54 |   99.55 |  412.57  | 436.11  | 269.85 |  297.50 |   28.04 |   58.27