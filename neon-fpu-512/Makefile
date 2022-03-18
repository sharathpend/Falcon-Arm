# This Makefile compiles the implementation in this directory along with
# the known answer tests generator located in the
# ../../../KAT/generator/ directory. The output is an executable program
# in the build/ subdirectory, whose name starts with 'kat', followed by
# the implementation name (e.g. 'kat512int' for the 'falcon512int'
# implementation). This program, when executed, generates the .req and
# .rsp files in the expected NIST format.

.POSIX:

# CC = gcc
CC = clang
CFLAGS = -W -Wall -O3 -Wextra -Wpedantic -g3
LD = c99
LDFLAGS = 
LIBS = 

OBJ = codec.c util.c common.c fft.c fft_tree.c fpr.c keygen.c rng.c katrng.c \
	  poly_float.c sign_sampler.c shake.c sign.c vrfy.c ntt.c ntt_consts.c \
	  poly_int.c nist.c

OBJ_SPEED = falcon.c speed.c

OBJ_KAT = PQCgenKAT_sign.c

HEAD = api.h fpr.h inner.h config.h katrng.h
HEAD1 = falcon.h

all: build build/kat512fpu build/kat1024fpu build/speed512 build/speed1024

build:
	-mkdir build

clean:
	-rm -f build/kat512fpu build/kat1024fpu

build/kat512fpu: $(HEAD) $(OBJ)
	$(CC) $(CFLAGS) -DFALCON_LOGN=9  -DALGNAME=falcon512fpu  -o $@ $(OBJ) $(OBJ_KAT)

build/kat1024fpu: $(HEAD) $(OBJ)
	$(CC) $(CFLAGS) -DFALCON_LOGN=10 -DALGNAME=falcon1024fpu -o $@ $(OBJ) $(OBJ_KAT)

build/speed512: $(HEAD1) $(HEAD) $(OBJ) $(OBJ_SPEED)
	$(CC) $(CFLAGS) -DFALCON_LOGN=9  -o $@ $(OBJ) $(OBJ_SPEED)

build/speed1024: $(HEAD1) $(HEAD) $(OBJ) $(OBJ_SPEED)
	$(CC) $(CFLAGS) -DFALCON_LOGN=10 -o $@ $(OBJ) $(OBJ_SPEED)