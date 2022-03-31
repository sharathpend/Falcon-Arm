# Falcon ARMv8

## Overall status

Complete Sign and Verify. 


## Sign 

List of vectorized files: 

- common.c
- fft_tree.c
- fft.c
- fpr.h
- poly_float.c
- sign_sampler.c
- sign.c

Bottlenecks are floating point operation, serial hashing and sampling. 

Todo: Maybe change FFT storage structure. (see TODO in fft_tree.c)

## Verify

List of vectorized files: 

- ntt_consts.c
- ntt_consts.h
- ntt_consts9.c
- ntt_consts10.c
- ntt.c
- poly_int.c
- poly.h
- vrfy.c

Bottleneck is serial hashing.

## TODO: 

NTT and FFT can be further enhance by Radix-4:

- In FFT, Radix-4 can be applied to `FFT_log2` and `IFFT_log2` function. 
- In NTT, Radix-4 can be applied to Forward and Inverse NTT. 


Radix-4 NTT for Forward NTT:  a +- bw

```c
Radix-2:

- Layer 8:
a_0' = a_0 + a_1*w_1 
a_1' = a_0 - a_1*w_1 

a_2' = a_2 + a_3*w_1 
a_2' = a_2 - a_3*w_1 

- Layer 7: 
a_0'' = a_0' + a_2'*w_2
a_2'' = a_0' - a_2'*w_2

a_1'' = a_1' + a_3'*w_3
a_3'' = a_1' - a_3'*w_3

Expand: 

a_0'' = a_0 + a_1*w_1 + (a_2 + a_3*w_1)*w_2 = a_0 + a_1*w_1 + (a_2*w_2 + a_3*w_3)
a_2'' = a_0 + a_1*w_1 - (a_2 + a_3*w_1)*w_2 = a_0 + a_1*w_1 - (a_2*w_2 + a_3*w_3)

a_1'' = a_0 - a_1*w_1 + (a_2 - a_3*w_1)*w_3 = a_0 - a_1*w_1 + (a_2*w_3 - a_3*w_4)
a_3'' = a_0 - a_1*w_1 - (a_2 - a_3*w_1)*w_3 = a_0 - a_1*w_1 - (a_2*w_3 - a_3*w_4)

The equation above is Radix-4. 

We can group pattern of operation base on index like this: 

0: (+, +, +)
2: (+, -, -)
1: (-, +, -)
3: (-, -, +)

Similarly for other indcies (distance of 4): 

0, 4, 8, 12, 16, 20, 24, 28: (+, +, +)

so on ... 


Recall Barrett reduction: 

a*w % N = a*w - N*[ (a * [wR/N]) / R ]

where [wR/N] are precomputed, and with R= 2^16 the multiplication use multiply return high-only.

Since a % N + b % N = (a + b) % N. So we expand the Barrett reduction above to 2 elements: 

(a*w_1 + b*w_2) % N = (a*w_1 + b*w_2) - N*[ ( (a * [w_1*R/N]) + (b * w_2*R/N) ) / R ]

Expand to 4 elements, then we have: 

a_0 + a_1 * w_1 + a_2*w_2 + a_3*w_3 % N 

can be implement using 7 MUL instructions: 

z1 = mla(a_0, a_1, w_1)     (a0 + a1*w1)
z2 = mla(z1, a_2, w_2)      (a0 + a1*w1 + a2*w2)
z3 = mla(z2, a_3, w_3)      (a0 + a1*w1 + a2*w2 + a3*w3)

t1 = sqrdmulh(a1, w_1*R/N)           (a1 * [w1*R/N])/R
t2 = sqrdmlah(t1, a2, w_2*R/N)       (a1 * [w1*R/N] + a2 * [w2*R/N])/R
t3 = sqrdmlah(t2, a3, w_3*R/N)       (a1 * [w1*R/N] + a2 * [w2*R/N] + a3 * [w3*R/N])/R

z = vmls(z3, t3, N) 

```


Compare Radix-4 versus Radix-2: 

| Radix-4 | Radix-2 | 
| 7 MUL   | 6 MUL + 4 ADD | 

Less number of additions and reduce to N. This lead to fewer Barrett points reduction.  

TODO: generalize this to other indexes: 

0, 4, 8,  12, 16, 20, 24, 28: (+, +, +)  -- Example show above
1, 5, 9,  13, 17, 21, 25, 29: (+, -, -)
2, 6, 10, 14, 18, 22, 26, 30: (-, +, -)
3, 7, 11, 15, 19, 23, 27, 31: (-, -, +)



