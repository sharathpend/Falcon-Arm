# AVX2 

```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          8
On-line CPU(s) list:             0-7
Thread(s) per core:              2
Core(s) per socket:              4
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           140
Model name:                      11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Stepping:                        1
CPU MHz:                         2800.000
CPU max MHz:                     4700,0000
CPU min MHz:                     400,0000
BogoMIPS:                        5606.40
Virtualization:                  VT-x
L1d cache:                       192 KiB
L1i cache:                       128 KiB
L2 cache:                        5 MiB
L3 cache:                        12 MiB
NUMA node0 CPU(s):               0-7
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl a
                                 nd seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer s
                                 anitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca 
                                 cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht t
                                 m pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_
                                 perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpui
                                 d aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor 
                                 ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse

```

## AVX2 disable

### First Run 

|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 13884.02|   176.19|   584.02|   614.50|   359.80|   395.09|    51.23|    83.29|
| 1024:| 42237.31|   357.13|  1176.50|  1239.99|   705.43|   764.92|   102.59|   166.26|


|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     7.47|    63.16|   226.45|   221.61|   127.72|   136.30|    21.87|    29.54|
| 1024:|    14.91|   126.36|   412.05|   461.37|   253.96|   272.64|    37.44|    61.72|


### Second Run 

|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 15028.97|   189.60|   595.98|   653.06|   369.76|   413.90|    57.67|    94.78|
| 1024:| 43500.22|   355.47|  1187.07|  1246.55|   710.46|   786.32|   104.46|   168.77|


|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     6.00|    62.95|   211.67|   221.19|   128.79|   137.02|    18.80|    30.31|
| 1024:|    20.28|   127.00|   418.09|   452.03|   254.60|   273.36|    37.68|    60.77|


### Third Run 


|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 13852.58|   176.19|   580.36|   611.33|   353.38|   384.51|    50.77|    82.92|
| 1024:| 43056.73|   355.59|  1184.61|  1245.73|   710.91|   772.31|   104.00|   169.15|


|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     4.86|    62.89|   209.68|   222.97|   132.06|   135.30|    20.68|    33.99|
| 1024:|    15.18|   126.83|   413.84|   434.64|   253.51|   276.66|    37.37|    63.23|

### Frequency

./avx_speed_ghz
|degree|  kg(Ghz)|   ek(Ghz)|   sd(Ghz)|  sdc(Ghz)|   st(Ghz)|  stc(Ghz)|   vv(Ghz)|  vvc(Ghz)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|      2.82|      2.50|      2.80|      2.79|      2.80|      2.81|      2.74|      2.80|
| 1024:|      2.78|      2.81|      2.79|      2.61|      2.78|      2.80|      2.80|      2.79|


### FFT 

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       20 |       18
| FFT 1 |       19 |       18
| FFT 2 |       25 |       30
| FFT 3 |       49 |       53
| FFT 4 |       99 |      106
| FFT 5 |      201 |      215
| FFT 6 |      436 |      401
| FFT 7 |      864 |      906
| FFT 8 |     1795 |     1951
| FFT 9 |     3860 |     3674
| FFT 10 |     7162 |     7608

### NTT

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |     8486 |     9057
| NTT 10 |    17948 |    21985


## AVX2 enable

### First Run 

|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 13832.64|   167.25|   423.13|   453.90|   216.64|   246.84|    49.68|    82.94|
| 1024:| 41752.26|   344.23|   854.18|   915.86|   435.36|   496.47|   101.71|   165.78|


|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     6.15|    59.57|   151.05|   163.08|    76.95|    88.00|    20.59|    33.89|
| 1024:|    20.04|   122.89|   341.29|   325.08|   156.48|   182.90|    37.83|    59.18|

### Second Run 

|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 13943.66|   167.15|   427.10|   457.86|   216.44|   246.90|    50.56|    82.66|
| 1024:| 41079.54|   342.98|   865.91|   925.39|   441.13|   492.58|   102.44|   165.52|


|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     5.30|    67.06|   153.37|   165.54|    87.79|   102.01|    29.64|    34.49|
| 1024:|    15.07|   122.86|   307.66|   329.30|   152.35|   176.50|    37.36|    65.05|

### Third Run 

|degree|  kg(kc)|   ek(kc)|  sd(kc)| sdc(kc)|  st(kc)| stc(kc)|  vv(kc)| vvc(kc)|
| ---- | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:| 14207.52|   176.28|   426.25|   454.98|   217.51|   252.13|    50.95|    82.66|
| 1024:| 42764.18|   342.92|   872.93|   935.09|   443.10|   530.86|   103.35|   168.54|

|degree|  kg(ms)|  ek(us)|  sd(us)| sdc(us)|  st(us)| stc(us)|  vv(us)| vvc(us)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|     9.03|    61.07|   149.50|   163.89|    80.28|    89.32|    18.59|    31.01|
| 1024:|    13.60|   123.40|   344.59|   363.66|   174.76|   198.01|    42.31|    67.90|

### Frequency

|degree|  kg(Ghz)|   ek(Ghz)|   sd(Ghz)|  sdc(Ghz)|   st(Ghz)|  stc(Ghz)|   vv(Ghz)|  vvc(Ghz)|
| ---- |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  512:|      2.70|      2.80|      2.80|      2.80|      2.80|      2.81|      2.80|      2.80|
| 1024:|      2.78|      2.81|      2.81|      2.79|      3.00|      2.76|      3.14|      2.81|



### FFT 

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       20 |       23
| FFT 1 |       22 |       22
| FFT 2 |       30 |       32
| FFT 3 |       45 |       47
| FFT 4 |       73 |       76
| FFT 5 |      134 |      142
| FFT 6 |      228 |      253
| FFT 7 |      509 |      519
| FFT 8 |     1016 |     1065
| FFT 9 |     2107 |     2216
| FFT 10 |     4544 |     4741

### NTT

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |     8448 |     9123
| NTT 10 |    17847 |    19193