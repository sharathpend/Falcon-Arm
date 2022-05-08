# NEON M1 

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       77 |       76
| FFT 1 |       77 |       76
| FFT 2 |       82 |       85
| FFT 3 |       91 |       88
| FFT 4 |      110 |      103
| FFT 5 |      147 |      144
| FFT 6 |      232 |      228
| FFT 7 |      404 |      401
| FFT 8 |      789 |      794
| FFT 9 |     1577 |     1609
| FFT 10 |     3489 |     3547

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |      847 |      860
| NTT 10 |     1693 |     1756

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |       79
| poly_add |        1 |       76
| poly_add |        2 |       76
| poly_add |        3 |       79
| poly_add |        4 |       82
| poly_add |        5 |       87
| poly_add |        6 |      103
| poly_add |        7 |      134
| poly_add |        8 |      256
| poly_add |        9 |      375
| poly_add |       10 |      677

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |       79
| poly_sub |        1 |       76
| poly_sub |        2 |       76
| poly_sub |        3 |       79
| poly_sub |        4 |       82
| poly_sub |        5 |       87
| poly_sub |        6 |      103
| poly_sub |        7 |      134
| poly_sub |        8 |      256
| poly_sub |        9 |      373
| poly_sub |       10 |      599

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |       76
| poly_neg |        1 |       76
| poly_neg |        2 |       75
| poly_neg |        3 |       76
| poly_neg |        4 |       82
| poly_neg |        5 |       83
| poly_neg |        6 |       90
| poly_neg |        7 |      111
| poly_neg |        8 |      241
| poly_neg |        9 |      361
| poly_neg |       10 |      633

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |       77
| poly_adj_fft |        1 |       75
| poly_adj_fft |        2 |       78
| poly_adj_fft |        3 |       75
| poly_adj_fft |        4 |       77
| poly_adj_fft |        5 |       83
| poly_adj_fft |        6 |       86
| poly_adj_fft |        7 |       91
| poly_adj_fft |        8 |      181
| poly_adj_fft |        9 |      164
| poly_adj_fft |       10 |      315

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |       77
| poly_mul_fft |        1 |       76
| poly_mul_fft |        2 |       76
| poly_mul_fft |        3 |       77
| poly_mul_fft |        4 |       83
| poly_mul_fft |        5 |       91
| poly_mul_fft |        6 |      110
| poly_mul_fft |        7 |      153
| poly_mul_fft |        8 |      283
| poly_mul_fft |        9 |      410
| poly_mul_fft |       10 |      659

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |       77
| poly_invnorm2_fft |        1 |       77
| poly_invnorm2_fft |        2 |       79
| poly_invnorm2_fft |        3 |       80
| poly_invnorm2_fft |        4 |       84
| poly_invnorm2_fft |        5 |       93
| poly_invnorm2_fft |        6 |      120
| poly_invnorm2_fft |        7 |      174
| poly_invnorm2_fft |        8 |      293
| poly_invnorm2_fft |        9 |      454
| poly_invnorm2_fft |       10 |      733

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |       77
| poly_mul_autoadj_fft |        1 |       77
| poly_mul_autoadj_fft |        2 |       75
| poly_mul_autoadj_fft |        3 |       79
| poly_mul_autoadj_fft |        4 |       82
| poly_mul_autoadj_fft |        5 |       87
| poly_mul_autoadj_fft |        6 |       99
| poly_mul_autoadj_fft |        7 |      134
| poly_mul_autoadj_fft |        8 |      261
| poly_mul_autoadj_fft |        9 |      400
| poly_mul_autoadj_fft |       10 |      722

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |       76
| poly_LDL_fft |        1 |       92
| poly_LDL_fft |        2 |       91
| poly_LDL_fft |        3 |       95
| poly_LDL_fft |        4 |      100
| poly_LDL_fft |        5 |      128
| poly_LDL_fft |        6 |      187
| poly_LDL_fft |        7 |      274
| poly_LDL_fft |        8 |      536
| poly_LDL_fft |        9 |      889
| poly_LDL_fft |       10 |     1574

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |       79
| poly_LDLmv_fft |        1 |       92
| poly_LDLmv_fft |        2 |       91
| poly_LDLmv_fft |        3 |       95
| poly_LDLmv_fft |        4 |       99
| poly_LDLmv_fft |        5 |      131
| poly_LDLmv_fft |        6 |      189
| poly_LDLmv_fft |        7 |      274
| poly_LDLmv_fft |        8 |      505
| poly_LDLmv_fft |        9 |      856
| poly_LDLmv_fft |       10 |     1546

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |       77
| poly_split_fft |        1 |       75
| poly_split_fft |        2 |       80
| poly_split_fft |        3 |       79
| poly_split_fft |        4 |       90
| poly_split_fft |        5 |      109
| poly_split_fft |        6 |      145
| poly_split_fft |        7 |      209
| poly_split_fft |        8 |      323
| poly_split_fft |        9 |      549
| poly_split_fft |       10 |     1004

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       77
| poly_merge_fft |        1 |       77
| poly_merge_fft |        2 |       79
| poly_merge_fft |        3 |       81
| poly_merge_fft |        4 |       86
| poly_merge_fft |        5 |       99
| poly_merge_fft |        6 |      128
| poly_merge_fft |        7 |      186
| poly_merge_fft |        8 |      273
| poly_merge_fft |        9 |      449
| poly_merge_fft |       10 |      801


| degree | kg(ms) |  ek(us) |  sd(us) | sdc(us) |  st(us) | stc(us) |  vv(us) | vvc(us)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
| 512: |    5.25 |   14.58 |  140.85 |  155.77 |  113.82 |  129.08  |   6.77  |  21.69
|1024: |   17.22 |   31.33 |  281.35 |  303.76 |  222.11 |  249.59  |  13.30  |  42.50

| degree |  kg(kc)   | ek(kc) |  sd(kc) | sdc(kc) |  st(kc) | stc(kc) |  vv(kc) | vvc(kc)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
| 512: | 15756.22  |  45.59 |  442.74 |  486.31  | 360.77 |  403.56 |   22.59 |   68.31
|1024: | 48659.27  |  97.93 |  884.49 |  972.24  | 711.44 |  800.75 |   42.85 |  136.11