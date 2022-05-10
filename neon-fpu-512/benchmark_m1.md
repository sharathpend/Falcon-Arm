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
| poly_add |        9 |      369
| poly_add |       10 |      612

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
| poly_sub |        9 |      370
| poly_sub |       10 |      636

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
| poly_adj_fft |        2 |       76
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
| poly_mul_fft |        9 |      411
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
| poly_invnorm2_fft |        8 |      292
| poly_invnorm2_fft |        9 |      455
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
| poly_LDL_fft |        7 |      277
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
| poly_split_fft |        3 |       82
| poly_split_fft |        4 |       90
| poly_split_fft |        5 |      110
| poly_split_fft |        6 |      145
| poly_split_fft |        7 |      208
| poly_split_fft |        8 |      323
| poly_split_fft |        9 |      549
| poly_split_fft |       10 |     1004

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       77
| poly_merge_fft |        1 |       76
| poly_merge_fft |        2 |       77
| poly_merge_fft |        3 |       81
| poly_merge_fft |        4 |       87
| poly_merge_fft |        5 |      101
| poly_merge_fft |        6 |      126
| poly_merge_fft |        7 |      182
| poly_merge_fft |        8 |      265
| poly_merge_fft |        9 |      443
| poly_merge_fft |       10 |      769


| degree | kg(ms) |  ek(us) |  sd(us) | sdc(us) |  st(us) | stc(us) |  vv(us) | vvc(us)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512: |    5.33  |  14.65 |  141.44 |  155.03  | 115.11 |  128.66  |   6.71  |  21.72
| 1024: |   17.61  |  31.60 |  282.57 |  309.18  | 228.03 |  255.70  |  13.58  |  43.23

| degree | kg(kc)  | ek(kc) |  sd(kc) | sdc(kc) |  st(kc) | stc(kc)  | vv(kc) | vvc(kc)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512:  | 15754.82  |  45.86 |  442.83 |  485.81 |  359.06 |  404.54  |  22.62 |   68.20
| 1024:  | 48673.33  |  98.64 |  883.50 |  970.90 |  712.15 |  798.06  |  42.84 |  135.55


| degree | kg(Ghz)  | ek(Ghz)  | sd(Ghz) | sdc(Ghz)  | st(Ghz)  | stc(Ghz)  | vv(Ghz)  | vvc(Ghz)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512:  |    3.20  |    3.16  |    3.15 |     3.14  |    3.14  |     3.14  |    3.15  |     3.14
| 1024:  |    3.20  |    3.13  |    3.14 |     3.13  |    3.14  |     3.13  |    3.15  |     3.14
