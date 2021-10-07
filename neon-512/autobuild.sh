#!/bin/bash

gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c
./neon-fft