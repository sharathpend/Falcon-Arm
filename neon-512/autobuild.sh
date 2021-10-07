#!/bin/bash
set -e 
gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c -O0 -g3 -Wall
./neon-fft