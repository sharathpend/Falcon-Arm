# NEON A72

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       21 |        9
| FFT 1 |       21 |        9
| FFT 2 |       25 |       20
| FFT 3 |       54 |       49
| FFT 4 |       92 |       88
| FFT 5 |      221 |      217
| FFT 6 |      540 |      543
| FFT 7 |     1155 |     1216
| FFT 8 |     2770 |     2913
| FFT 9 |     5951 |     6135
| FFT 10 |    14060 |    14705

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |     3518 |     3936
| NTT 10 |     7579 |     9021

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |       18
| poly_add |        1 |        9
| poly_add |        2 |        9
| poly_add |        3 |       18
| poly_add |        4 |       33
| poly_add |        5 |       63
| poly_add |        6 |      123
| poly_add |        7 |      243
| poly_add |        8 |      483
| poly_add |        9 |      987
| poly_add |       10 |     1952

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |       18
| poly_sub |        1 |        9
| poly_sub |        2 |        9
| poly_sub |        3 |       18
| poly_sub |        4 |       33
| poly_sub |        5 |       63
| poly_sub |        6 |      123
| poly_sub |        7 |      243
| poly_sub |        8 |      483
| poly_sub |        9 |      987
| poly_sub |       10 |     1949

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |       14
| poly_neg |        1 |        9
| poly_neg |        2 |        9
| poly_neg |        3 |       14
| poly_neg |        4 |       25
| poly_neg |        5 |       47
| poly_neg |        6 |       91
| poly_neg |        7 |      185
| poly_neg |        8 |      377
| poly_neg |        9 |      730
| poly_neg |       10 |     1557

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |       16
| poly_adj_fft |        1 |        9
| poly_adj_fft |        2 |        9
| poly_adj_fft |        3 |        9
| poly_adj_fft |        4 |       16
| poly_adj_fft |        5 |       26
| poly_adj_fft |        6 |       46
| poly_adj_fft |        7 |       90
| poly_adj_fft |        8 |      186
| poly_adj_fft |        9 |      378
| poly_adj_fft |       10 |      784

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |       10
| poly_mul_fft |        1 |       12
| poly_mul_fft |        2 |        9
| poly_mul_fft |        3 |       17
| poly_mul_fft |        4 |       48
| poly_mul_fft |        5 |       85
| poly_mul_fft |        6 |      161
| poly_mul_fft |        7 |      313
| poly_mul_fft |        8 |      617
| poly_mul_fft |        9 |     1225
| poly_mul_fft |       10 |     2459

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |       10
| poly_invnorm2_fft |        1 |        9
| poly_invnorm2_fft |        2 |       16
| poly_invnorm2_fft |        3 |       29
| poly_invnorm2_fft |        4 |      137
| poly_invnorm2_fft |        5 |      280
| poly_invnorm2_fft |        6 |      566
| poly_invnorm2_fft |        7 |     1138
| poly_invnorm2_fft |        8 |     2282
| poly_invnorm2_fft |        9 |     4570
| poly_invnorm2_fft |       10 |     9146

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |        9
| poly_mul_autoadj_fft |        1 |        9
| poly_mul_autoadj_fft |        2 |        9
| poly_mul_autoadj_fft |        3 |       10
| poly_mul_autoadj_fft |        4 |       30
| poly_mul_autoadj_fft |        5 |       55
| poly_mul_autoadj_fft |        6 |      105
| poly_mul_autoadj_fft |        7 |      205
| poly_mul_autoadj_fft |        8 |      405
| poly_mul_autoadj_fft |        9 |      805
| poly_mul_autoadj_fft |       10 |     1635

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |       12
| poly_LDL_fft |        1 |       34
| poly_LDL_fft |        2 |       48
| poly_LDL_fft |        3 |       84
| poly_LDL_fft |        4 |      208
| poly_LDL_fft |        5 |      418
| poly_LDL_fft |        6 |      838
| poly_LDL_fft |        7 |     1678
| poly_LDL_fft |        8 |     3392
| poly_LDL_fft |        9 |     6783
| poly_LDL_fft |       10 |    13829

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |       12
| poly_LDLmv_fft |        1 |       31
| poly_LDLmv_fft |        2 |       39
| poly_LDLmv_fft |        3 |       85
| poly_LDLmv_fft |        4 |      203
| poly_LDLmv_fft |        5 |      406
| poly_LDLmv_fft |        6 |      812
| poly_LDLmv_fft |        7 |     1624
| poly_LDLmv_fft |        8 |     3260
| poly_LDLmv_fft |        9 |     6542
| poly_LDLmv_fft |       10 |    13598

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |       10
| poly_split_fft |        1 |        9
| poly_split_fft |        2 |       13
| poly_split_fft |        3 |       21
| poly_split_fft |        4 |       53
| poly_split_fft |        5 |       86
| poly_split_fft |        6 |      158
| poly_split_fft |        7 |      302
| poly_split_fft |        8 |      591
| poly_split_fft |        9 |     1188
| poly_split_fft |       10 |     2658

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       10
| poly_merge_fft |        1 |       10
| poly_merge_fft |        2 |       14
| poly_merge_fft |        3 |       23
| poly_merge_fft |        4 |       40
| poly_merge_fft |        5 |       77
| poly_merge_fft |        6 |      138
| poly_merge_fft |        7 |      260
| poly_merge_fft |        8 |      504
| poly_merge_fft |        9 |     1004
| poly_merge_fft |       10 |     2029


| degree | kg(ms) |  ek(us) |  sd(us) | sdc(us) |  st(us) | stc(us) |  vv(us) | vvc(us)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512:  |  19.03 |  116.69 |  583.41 |  626.47 |  414.85 |  453.67 |   33.43 |   77.12
| 1024:  |  55.40 |  250.61 | 1189.01 | 1292.99 |  836.92 |  913.47 |   69.16 |  151.11


| degree |  kg(kc)   | ek(kc) |  sd(kc) | sdc(kc) |  st(kc) | stc(kc) |  vv(kc) | vvc(kc)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512:  | 31417.68  | 209.54 | 1044.61 | 1128.44 |  744.35 |  816.24 |   61.26 |  138.64
| 1024:  | 88141.58  | 452.12 | 2137.04 | 2320.12 | 1507.48 | 1640.51 |  125.20 |  271.60
