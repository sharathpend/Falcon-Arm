
FALCON_N = 1024
global_reduce = 0

def reduce(array, start, end):
    global global_reduce
    for i in range(start, end):
        array[i] = 1
        global_reduce += 1

def reduce_signed(array, start, end):
    global global_reduce
    for i in range(start, end):
        array[i][0] = -0.5
        array[i][1] = 0.5
        global_reduce += 1

def printa(a, gap):
    count = 0
    for i, j in enumerate(a):
        if j == 5:
            print(i, end=', ')
            count += 1
            if count % gap == 0 and count != 0:
                print()

def ntt_unsigned(a):
    global global_reduce
    len = FALCON_N//2
    j = 0    
    layer = 0
    while len > 0:
        print("layer = ", layer)
        start = 0
        while start < FALCON_N:
            j = start
            while j < start + len:
                bound_t = 1
                a[j + len] = a[j] - bound_t + 1
                a[j] = a[j] + bound_t

                j += 1

            start = j + len
        
        # Layer 0
        if len == FALCON_N//2:
            pass

        # Layer 1
        if len == FALCON_N//4:
            pass
        
        # Layer 2
        if len == FALCON_N//8:
            pass
        
        # Layer 3
        if len == FALCON_N//16:
            reduce(a, 0, 64)

        # Layer 4
        if len == FALCON_N//32:
            reduce(a, 64, 64 + 32)
            reduce(a, 128, 128 + 32)
            reduce(a, 256, 256 + 32)
            reduce(a, 512, 512 + 32)

        # Layer 5
        if len == FALCON_N//64:
            reduce(a, 96, 96 + 16)
            reduce(a, 160, 160 + 16)
            reduce(a, 192, 192 + 16)
            reduce(a, 288, 288 + 16)
            reduce(a, 320, 320 + 16)

            reduce(a, 384, 384 + 16)
            reduce(a, 544, 544 + 16)
            reduce(a, 576, 576 + 16)
            reduce(a, 640, 640 + 16)
            reduce(a, 768, 768 + 16)

        # Layer 6
        if len == FALCON_N//128:
            reduce(a, 112, 112 + 8)
            reduce(a, 176, 176 + 8)
            reduce(a, 208, 208 + 8)
            reduce(a, 224, 224 + 8)
            reduce(a, 304, 304 + 8)
            reduce(a, 336, 336 + 8)
            reduce(a, 352, 352 + 8)
            reduce(a, 400, 400 + 8)
            reduce(a, 416, 416 + 8)
            reduce(a, 448, 448 + 8)
            reduce(a, 560, 560 + 8)
            reduce(a, 592, 592 + 8)
            reduce(a, 608, 608 + 8)
            reduce(a, 656, 656 + 8)
            reduce(a, 672, 672 + 8)
            reduce(a, 704, 704 + 8)
            reduce(a, 784, 784 + 8)
            reduce(a, 800, 800 + 8)
            reduce(a, 832, 832 + 8)
            reduce(a, 896, 896 + 8)

        # Layer 7
        if len == FALCON_N//256:
            pass
            reduce(a, 0, 0 + 4)
            reduce(a, 120, 120 + 4)
            reduce(a, 184, 184 + 4)
            reduce(a, 216, 216 + 4)
            reduce(a, 232, 232 + 4)
            reduce(a, 240, 240 + 4)
            reduce(a, 312, 312 + 4)
            reduce(a, 344, 344 + 4)
            reduce(a, 360, 360 + 4)
            reduce(a, 368, 368 + 4)
            reduce(a, 408, 408 + 4)
            reduce(a, 424, 424 + 4)
            reduce(a, 432, 432 + 4)
            reduce(a, 456, 456 + 4)
            reduce(a, 464, 464 + 4)
            reduce(a, 480, 480 + 4)
            reduce(a, 568, 568 + 4)
            reduce(a, 600, 600 + 4)
            reduce(a, 616, 616 + 4)
            reduce(a, 624, 624 + 4)
            reduce(a, 664, 664 + 4)
            reduce(a, 680, 680 + 4)
            reduce(a, 688, 688 + 4)
            reduce(a, 712, 712 + 4)
            reduce(a, 720, 720 + 4)
            reduce(a, 736, 736 + 4)
            reduce(a, 792, 792 + 4)
            reduce(a, 808, 808 + 4)
            reduce(a, 816, 816 + 4)
            reduce(a, 840, 840 + 4)
            reduce(a, 848, 848 + 4)
            reduce(a, 864, 864 + 4)
            reduce(a, 904, 904 + 4)
            reduce(a, 912, 912 + 4)
            reduce(a, 928, 928 + 4)
            reduce(a, 960, 960 + 4)

        # Layer 8
        if len == FALCON_N//512:
            reduce(a, 4, 4 + 2)
            reduce(a, 8, 8 + 2)
            reduce(a, 16, 16 + 2)
            reduce(a, 32, 32 + 2)
            reduce(a, 64, 64 + 2)
            reduce(a, 124, 124 + 2)
            reduce(a, 128, 128 + 2)
            reduce(a, 188, 188 + 2)
            reduce(a, 220, 220 + 2)
            reduce(a, 236, 236 + 2)
            reduce(a, 244, 244 + 2)
            reduce(a, 248, 248 + 2)
            reduce(a, 256, 256 + 2)
            reduce(a, 316, 316 + 2)
            reduce(a, 348, 348 + 2)
            reduce(a, 364, 364 + 2)
            reduce(a, 372, 372 + 2)
            reduce(a, 376, 376 + 2)
            reduce(a, 412, 412 + 2)
            reduce(a, 428, 428 + 2)
            reduce(a, 436, 436 + 2)
            reduce(a, 440, 440 + 2)
            reduce(a, 460, 460 + 2)
            reduce(a, 468, 468 + 2)
            reduce(a, 472, 472 + 2)
            reduce(a, 484, 484 + 2)
            reduce(a, 488, 488 + 2)
            reduce(a, 496, 496 + 2)
            reduce(a, 512, 512 + 2)
            reduce(a, 572, 572 + 2)
            reduce(a, 604, 604 + 2)
            reduce(a, 620, 620 + 2)
            reduce(a, 628, 628 + 2)
            reduce(a, 632, 632 + 2)
            reduce(a, 668, 668 + 2)
            reduce(a, 684, 684 + 2)
            reduce(a, 692, 692 + 2)
            reduce(a, 696, 696 + 2)
            reduce(a, 716, 716 + 2)
            reduce(a, 724, 724 + 2)
            reduce(a, 728, 728 + 2)
            reduce(a, 740, 740 + 2)
            reduce(a, 744, 744 + 2)
            reduce(a, 752, 752 + 2)
            reduce(a, 796, 796 + 2)
            reduce(a, 812, 812 + 2)
            reduce(a, 820, 820 + 2)
            reduce(a, 824, 824 + 2)
            reduce(a, 844, 844 + 2)
            reduce(a, 852, 852 + 2)
            reduce(a, 856, 856 + 2)
            reduce(a, 868, 868 + 2)
            reduce(a, 872, 872 + 2)
            reduce(a, 880, 880 + 2)
            reduce(a, 908, 908 + 2)
            reduce(a, 916, 916 + 2)
            reduce(a, 920, 920 + 2)
            reduce(a, 932, 932 + 2)
            reduce(a, 936, 936 + 2)
            reduce(a, 944, 944 + 2)
            reduce(a, 964, 964 + 2)
            reduce(a, 968, 968 + 2)
            reduce(a, 976, 976 + 2)
            reduce(a, 992, 992 + 2)
            pass
        
        if len != 1:
            if all(0 <= i < 5 for i in a) != True:
                print(a[:32], all(0 <= i <= 5 for i in a))
                print("stop at len = ", len)
                printa(a, 2)
                return None

        layer += 1
        len >>= 1

    print("Success", global_reduce)

a = [1 for i in range(FALCON_N)]
# ntt_unsigned(a)

global_reduce = 0
a = [[-0.5, 0.5] for i in range(FALCON_N)]

import copy

def add(t, c):
    temp = copy.deepcopy(t)
    temp[0] += c[0]
    temp[1] += c[1]
    return temp

def sub(t, c):
    temp = copy.deepcopy(t)
    temp[0] -= c[0]
    temp[1] -= c[1]
    return temp

def check(a):
    for index, item in enumerate(a):
        i, j = item
        if i < -2.5 or i > 2.5 or j < -2.5 or j > 2.5:
            # print("[{}] = {},{}".format(index, i, j))
            print(f"{index}", end= ', ')
            # return False
    print()
    return True

def ntt_signed(a):
    global global_reduce
    len = FALCON_N//2
    j = 0    
    layer = 0
    while len > 0:
        print("layer = ", layer)
        start = 0
        while start < FALCON_N:
            j = start
            while j < start + len:
                bound = [-1.5, 1.5]
                
                a[j + len] = add(a[j], bound)
                a[j] = add(a[j], bound)
                
                # print(a[j], bound, add(a[j], bound))
                j += 1

            start = j + len

        if len == FALCON_N//2:
            # check(a)
            reduce_signed(a, 0, FALCON_N)
            pass

        if len == FALCON_N//4:
            reduce_signed(a, 0, FALCON_N)
            pass

        if len == FALCON_N//8:
            reduce_signed(a, 0, FALCON_N)
            pass

        if len == FALCON_N//16:
            reduce_signed(a, 0, FALCON_N)
            # print(a)
        
        if len == FALCON_N//32:
            reduce_signed(a, 0, FALCON_N)
            # print(a)

        if len == FALCON_N//64:
            # print(a)
            reduce_signed(a, 0, FALCON_N)
        
        if len == FALCON_N//128:
            # print(a)
            reduce_signed(a, 0, FALCON_N)

        if len == FALCON_N//256:
            # print(a)
            reduce_signed(a, 0, FALCON_N)
        
        if len == FALCON_N//512:
            # print(a)
            reduce_signed(a, 0, FALCON_N)
            
        if len != 1:
            if check(a) != True:
                # print(a[:4])
                # print("stop at len = ", len)
                return None

        layer += 1
        len >>= 1

    print("Success", global_reduce)

global_reduce = 0
a = [[-0.5, 0.5] for i in range(FALCON_N)]
ntt_signed(a)

# TODO: convert to Unsigned again. 
# Create barrett for Unsigned 
# Create Mont for Unsigned

