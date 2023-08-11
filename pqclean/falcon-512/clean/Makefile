# This Makefile can be used with GNU Make or BSD Make

LIB=libfalcon-512_clean.a

SOURCES = codec.c common.c fft.c fpr.c keygen.c pqclean.c rng.c sign.c vrfy.c ../../../common/randombytes.c ../../../common/fips202.c
OBJECTS = codec.o common.o fft.o fpr.o keygen.o pqclean.o rng.o sign.o vrfy.o ../../../common/randombytes.o ../../../common/fips202.o
HEADERS = api.h fpr.h inner.h

CFLAGS=-std=c99 -O2 -W -Wall -Wconversion -Wextra -Wpedantic -Wvla -Werror -Wmissing-prototypes -Wredundant-decls -I../../../common $(EXTRAFLAGS)

all: $(LIB)
test: test_api.o test_api

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

test_api.o: test_api.c
	$(CC) $(CFLAGS) -c -o $@ $<

test_api: $(HEAD) $(OBJECTS) test_api.c
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) test_api.o
	./$@

$(LIB): $(OBJECTS)
	$(AR) -r $@ $(OBJECTS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB)
