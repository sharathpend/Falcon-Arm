#!/bin/bash
set -e 
gcc -o ref-fft util.c fpr.c fft.c ref-fft.c -O0 -g3; 
./ref-fft
