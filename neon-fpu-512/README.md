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
