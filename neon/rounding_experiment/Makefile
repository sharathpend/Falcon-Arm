CC = clang
CFLAGS = -fomit-frame-pointer -W -Wall -O3 -Wextra -Wpedantic -Wshadow -Wundef


all: experiment

experiment: experiment.c
	$(CC) $(CFLAGS) -o $@ experiment.c
	./$@

clean:
	rm -f experiment


