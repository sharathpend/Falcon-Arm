from ntt_mont_1024 import *

FALCON_Q = 12289
FALCON_QINV = 53249
FALCON_N = 1024
FALCON_PADDING = 0

assert len(ntt_mont) == FALCON_N
assert len(ntt_qinv_mont) == FALCON_N


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


# 0 - 127
# 128 - 255
# 256 - 383
# 384 - 511
# 512 - 639
# 640 - 767
# 768 - 895
# 896 - 1023
def gen_table06_invntt(zetas):
    inv_zetas = zetas[:]
    inv_zetas = inv_zetas[::-1]

    bar0 = 0
    bar1 = FALCON_N // 2
    bar2 = FALCON_N // 2 + FALCON_N // 4
    bar3 = FALCON_N // 2 + FALCON_N // 4 + FALCON_N // 8
    bar4 = FALCON_N // 2 + FALCON_N // 4 + FALCON_N // 8 + FALCON_N // 16
    bar5 = (
        FALCON_N // 2 + FALCON_N // 4 + FALCON_N // 8 + FALCON_N // 16 + FALCON_N // 32
    )
    bar6 = (
        FALCON_N // 2
        + FALCON_N // 4
        + FALCON_N // 8
        + FALCON_N // 16
        + FALCON_N // 32
        + FALCON_N // 64
    )

    final_inv_zetas = []

    for iter in range(0, FALCON_N, 128):
        # Layer 0 = Distance 1
        pool = inv_zetas[bar0 : bar0 + 64]
        block1 = pool[0::2]
        block2 = pool[1::2]

        final_inv_zetas += block1
        final_inv_zetas += block2

        bar0 += 64

        # Layer 1 = Distance 2
        # 0 - 127
        pool = inv_zetas[bar1 : bar1 + 32]
        block3 = pool

        final_inv_zetas += block3

        bar1 += 32

        # Layer 2 = Distance 4
        pool = inv_zetas[bar2 : bar2 + 16]
        block5 = pool[0::2]
        block6 = pool[1::2]

        block5 = dup(block5, 4)
        block6 = dup(block6, 4)

        final_inv_zetas += block5
        final_inv_zetas += block6

        bar2 += 16

        # Layer 3 = Distance 8
        pool = inv_zetas[bar3 : bar3 + 8]
        block7 = pool

        block7 = dup(block7, 4)

        final_inv_zetas += block7
        bar3 += 8

        # Layer 4 = Distance 16
        pool = inv_zetas[bar4 : bar4 + 4]

        block8 = pool

        final_inv_zetas += block8
        bar4 += 4

        # Layer 5 = Distance 32
        pool = inv_zetas[bar5 : bar5 + 2]

        block9 = pool

        final_inv_zetas += block9
        bar5 += 2

        # Layer 6 = Distance 64
        pool = inv_zetas[bar6 : bar6 + 1]

        block10 = pool

        final_inv_zetas += block10
        bar6 += 1

        final_inv_zetas += [0]

        assert len(final_inv_zetas) % 8 == 0

    return final_inv_zetas


def gen_table79_invntt(zetas, twisted=False):
    inv_zetas = zetas[:]
    inv_zetas = inv_zetas[::-1]

    bar7 = (
        FALCON_N // 2
        + FALCON_N // 4
        + FALCON_N // 8
        + FALCON_N // 16
        + FALCON_N // 32
        + FALCON_N // 64
        + FALCON_N // 128
    )
    bar8 = (
        FALCON_N // 2
        + FALCON_N // 4
        + FALCON_N // 8
        + FALCON_N // 16
        + FALCON_N // 32
        + FALCON_N // 64
        + FALCON_N // 128
        + FALCON_N // 256
    )
    bar9 = (
        FALCON_N // 2
        + FALCON_N // 4
        + FALCON_N // 8
        + FALCON_N // 16
        + FALCON_N // 32
        + FALCON_N // 64
        + FALCON_N // 128
        + FALCON_N // 256
        + FALCON_N // 512
    )

    final_inv_zetas = []

    # Layer 7 = Distance 128
    final_inv_zetas = inv_zetas[bar7 : bar7 + 4]

    # Layer 8 = Distance 256
    final_inv_zetas += inv_zetas[bar8 : bar8 + 2]

    # Layer 9 = Distance 512, Embed N^-1 * Mont to the last layer for N = 1024
    # print("{} -> {}".format(bar9, bar9 + 1))
    if twisted:
        temp = [-18211]
    else:
        temp = [-10461]

    final_inv_zetas += temp

    # N^-1
    if twisted:
        temp = [n1024_inv_twisted]
    else:
        temp = [n1024_inv_root]

    final_inv_zetas += temp

    assert len(final_inv_zetas) % 8 == 0

    return final_inv_zetas


# Input table already in Montgomery domain, thus `mont = False`
invntt_mont = gen_table06_invntt(ntt_mont)
invntt_mont += gen_table79_invntt(ntt_mont, twisted=False)

# We compute twisted value for qinv
invntt_qinv_mont = gen_table06_invntt(ntt_qinv_mont)
invntt_qinv_mont += gen_table79_invntt(ntt_qinv_mont, twisted=True)

# Sanity check
for index, item in enumerate(invntt_mont):
    if item % 2 == 0 and item != 0:
        print(f"At {index}: {item}")


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


# Input table already in Montgomery domain, thus `mont = False`
table_ntt_mont = gen_table97_ntt(ntt_mont)
table_ntt_mont += gen_table60_ntt(ntt_mont)

# We compute twisted value for qinv
table_ntt_qinv_mont = gen_table97_ntt(ntt_qinv_mont)
table_ntt_qinv_mont += gen_table60_ntt(ntt_qinv_mont)

# Sanity check
for index, item in enumerate(invntt_mont):
    if item % 2 == 0 and item != 0:
        print(f"At {index}: {item}")

# print_table(invntt_mont, "invntt_mont")
# print_table(invntt_qinv_mont, "invntt_qinv_mont")

print_table(table_ntt_mont, "ntt_mont")
# print_table(table_ntt_qinv_mont, "ntt_qinv_mont")

# TODO: seprate Table for Forward and Inverse NTT
