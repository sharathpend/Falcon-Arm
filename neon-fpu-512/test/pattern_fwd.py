from ntt_br_1024 import *

FALCON_Q = 12289
FALCON_QINV = (-12287)
FALCON_N = 1024
FALCON_PADDING = 0


assert FALCON_Q * FALCON_QINV % pow(2, 16) == 1
assert len(ntt1024_br) == FALCON_N
assert len(ntt1024_qinv_br) == FALCON_N
assert len(intt1024_br) == FALCON_N
assert len(intt1024_qinv_br) == FALCON_N

def print_table(a, string):
    print("extern const int16_t %s[] = {" % string)
    for i in range(0, len(a), 8):
        for j in range(8):
            if a[i + j] == FALCON_PADDING:
                print("{:>8}".format("PADDING"), end=", ")
            else:
                print("{:8d}".format(a[i + j]), end=", ")
        print()
    print("}; // %s" % len(a), end="\n\n")

def dup(a, x):
    b = []
    for i in a:
        for j in range(x):
            b.append(i)
    return b

def gen_table97_ntt(zetas):
    bar9 = 1 << 0
    bar8 = 1 << 1
    bar7 = 1 << 2
    final_zetas = []

    final_zetas += [0]

    # Layer 9 = Distance 512
    final_zetas += zetas[bar9 : bar9 + 1]

    # Layer 8 = Distance 256
    final_zetas += zetas[bar8 : bar8 + 2]

    # Layer 7 = Distance 128
    final_zetas += zetas[bar7 : bar7 + 4]

    # print(final_zetas, len(final_zetas))
    assert len(final_zetas) % 8 == 0

    return final_zetas

def gen_table87_ntt(zetas):
    bar8 = 1 << 0
    bar7 = 1 << 1

    final_zetas = []
    final_zetas += [0]

    # Layer 8 = Distance 256
    final_zetas += zetas[bar8: bar8 + 1]
    
    # Layer 7 = Distance 128
    final_zetas += zetas[bar7: bar7 + 2]

    assert len(final_zetas) % 8 == 0

    return zetas

# 1024
def gen_table60_ntt(zetas):
    final_zetas = []

    bar6 = 1 << 3
    bar5 = 1 << 4
    bar4 = 1 << 5
    bar3 = 1 << 6
    bar2 = 1 << 7
    bar1 = 1 << 8
    bar0 = 1 << 9

    for iter in range(0, FALCON_N, 128):

        # Layer 6 = Distance 64
        pool = zetas[bar6 : bar6 + 1]

        block0 = pool

        final_zetas += block0

        bar6 += 1

        # Layer 5 = Distance 32
        pool = zetas[bar5 : bar5 + 2]

        block1 = pool

        final_zetas += block1

        bar5 += 2

        # Layer 4 = Distance 16
        pool = zetas[bar4 : bar4 + 4]

        block2 = pool

        final_zetas += block2

        bar4 += 4

        # Layer 3 = Distance 8
        pool = zetas[bar3 : bar3 + 8]

        block3 = pool

        final_zetas += block3

        bar3 += 8

        # Padding
        final_zetas += [0]

        assert len(final_zetas) % 8 == 0

        # Layer 2 = Distance 4
        pool = zetas[bar2 : bar2 + 16]

        block4 = pool[0::2]
        block5 = pool[1::2]

        block4 = dup(block4, 4)
        block5 = dup(block5, 4)

        final_zetas += block4
        final_zetas += block5

        bar2 += 16

        # Layer 1 = Distance 2
        pool = zetas[bar1 : bar1 + 32]

        block5 = pool

        final_zetas += block5

        bar1 += 32

        # Layer 0 = Distance 1
        pool = zetas[bar0 : bar0 + 64]

        block6 = pool[0::2]
        block7 = pool[1::2]

        final_zetas += block6
        final_zetas += block7

        bar0 += 64

    assert len(final_zetas) % 8 == 0

    return final_zetas


table_ntt1024_br = gen_table97_ntt(ntt1024_br)
table_ntt1024_br += gen_table60_ntt(ntt1024_br)

# We compute twisted value for qinv
table_ntt1024_qinv_br = gen_table97_ntt(ntt1024_qinv_br)
table_ntt1024_qinv_br += gen_table60_ntt(ntt1024_qinv_br)

print_table(table_ntt1024_br, "ntt_br")
print_table(table_ntt1024_qinv_br, "ntt_qinv_br")

