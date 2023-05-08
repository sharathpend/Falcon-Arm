#define STACK_SIZE 16

.macro save_gprs // slothy:no-unfold
        sub sp, sp, #(16*6)
        stp x19, x20, [sp, #16*0]
        stp x19, x20, [sp, #16*0]
        stp x21, x22, [sp, #16*1]
        stp x23, x24, [sp, #16*2]
        stp x25, x26, [sp, #16*3]
        stp x27, x28, [sp, #16*4]
        str x29, [sp, #16*5]
.endm

.macro restore_gprs // slothy:no-unfold
        ldp x19, x20, [sp, #16*0]
        ldp x21, x22, [sp, #16*1]
        ldp x23, x24, [sp, #16*2]
        ldp x25, x26, [sp, #16*3]
        ldp x27, x28, [sp, #16*4]
        ldr x29, [sp, #16*5]
        add sp, sp, #(16*6)
.endm

.macro save_vregs // slothy:no-unfold
        sub sp, sp, #(16*4)
        stp  d8,  d9, [sp, #16*0]
        stp d10, d11, [sp, #16*1]
        stp d12, d13, [sp, #16*2]
        stp d14, d15, [sp, #16*3]
.endm

.macro restore_vregs // slothy:no-unfold
        ldp  d8,  d9, [sp, #16*0]
        ldp d10, d11, [sp, #16*1]
        ldp d12, d13, [sp, #16*2]
        ldp d14, d15, [sp, #16*3]
        add sp, sp, #(16*4)
.endm

.macro push_stack // slothy:no-unfold
        save_gprs
        save_vregs
        sub sp, sp, #STACK_SIZE
.endm

.macro pop_stack // slothy:no-unfold
        add sp, sp, #STACK_SIZE
        restore_vregs
        restore_gprs
.endm

// v0: 0, 1, 2, 3
// v1: 4, 5, 6, 7
// v2: 8, 9, 10, 11
// v3: 12, 13, 14, 15
// zl, zll: 16, 17
// zh, zhh: 18, 19
// neon_qmvq: 23*
// t: 24, 25, 26, 27
// t2: 28, 29, 30, 31


.macro gsbf_top a, b, t
    // t = a - b
    sub.s16 \t, \a, \b
    // a = a + b
    add.s16 \a, \a, \b
.endm

.macro gsbf_br_bot b, zl, zh, QMVQ, t
    sqrdmulh.s16 \b, \t, \zh
    mul.s16 \t, \t, \zl
    mls.s16 \t, \b, \QMVQ[0]
.endm

.macro gsbf_bri_bot b, zl, zh, i, QMVQ, t
    sqrdmulh.s16 \b, \t, \zh[i]
    mul.s16 \t, \t, \zl[i]
    mls.s16 \t, \b, \QMVQ[0]
.endm

.macro barrett a, QMVQ, t
    sqdmulh.s16 \t, \a, \QMVQ[4]
    srshr.s16 \t, \t, 11
    mls.s16 \a, \t, \QMVQ[0]
.endm

.macro transpose v0, v1, v2, v3, t0, t1, t2, t3
    trn1.s16 \t0, \v0, \v1
    trn2.s16 \t1, \v0, \v1
    trn1.s16 \t2, \v2, \v3
    trn2.s16 \t3, \v2, \v3

    trn1.s32 \v0, \t0, \t2
    trn2.s32 \v2, \t0, \t2
    trn1.s32 \v1, \t1, \t3
    trn2.s32 \v3, \t1, \t3
.endm

// Arrange and map to new number
.macro arrange t0, t1, t2, t3, v0, v1, v2, v3
    trn1.s64 \t0, \v0, \v1
    trn2.s64 \t2, \v0, \v1
    trn1.s64 \t1, \v2, \v3
    trn2.s64 \t3, \v2, \v3
.endm

.macro barmul_invntt a, zl, zh, i, QMVQ, t
    sqrdmulh.s16 \t, \a, \zh[i]
    mul.s16 \a, \a, \zl[i]
    mls.s16 \a, \t, \QMVQ[0]
.endm

neon_falcon_inner_poly_invntt_1st_loop:

    // Layer 0
    ld4.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly]
    ld4.s16 {v4.8h, v5.8h, v6.8h, v7.8h}, [poly, #(32*2)]
    ld4.s16 {v8.8h, v9.8h, v10.8h, v11.8h}, [poly, #(64*2)]
    ld4.s16 {v12.8h, v13.8h, v14.8h, v15.8h}, [poly, #(96*2)]

    gsbf_top v0.8h, v1.8h, v24.8h
    gsbf_top v4.8h, v5.8h, v25.8h
    gsbf_top v8.8h, v9.8h, v26.8h
    gsbf_top v12.8h, v13.8h, v27.8h

    gsbf_top v2.8h, v3.8h, v28.8h
    gsbf_top v6.8h, v7.8h, v29.8h
    gsbf_top v10.8h, v11.8h, v30.8h
    gsbf_top v14.8h, v15.8h, v31.8h
    
    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_br_bot v1.8h, v16.8h, v18.8h, v23.8h, v24.8h
    gsbf_br_bot v5.8h, v17.8h, v19.8h, v23.8h, v25.8h

    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_br_bot v9.8h,  v16.8h, v18.8h, v23.8h, v26.8h
    gsbf_br_bot v13.8h, v17.8h, v19.8h, v23.8h, v27.8h

    ld1.s16 {v16.8h, v17.8h, v18.8h, v19.8h}, [ptr_invntt_br], #(32*2)
    ld1.s16 {v24.8h, v25.8h, v26.8h, v27.8h}, [ptr_invntt_qinv_br], #(32*2)

    gsbf_br_bot v3.8h, v16.8h, v24.8h,  v23.8h, v28.8h
    gsbf_br_bot v7.8h, v17.8h, v25.8h,  v23.8h, v29.8h
    gsbf_br_bot v11.8h, v18.8h, v26.8h, v23.8h, v30.8h
    gsbf_br_bot v15.8h, v19.8h, v27.8h, v23.8h, v31.8h

    barrett v0.8h,  v23.8h, v24.8h
    barrett v4.8h,  v23.8h, v25.8h
    barrett v8.8h,  v23.8h, v26.8h
    barrett v12.8h, v23.8h, v27.8h

    // End Layer 0

    // Layer 1
    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_top v0.8h, v2.8h, v24.8h
    gsbf_top v1.8h, v3.8h, v25.8h
    gsbf_top v4.8h, v6.8h, v26.8h
    gsbf_top v5.8h, v7.8h, v27.8h

    gsbf_top v8.8h, v10.8h, v28.8h
    gsbf_top v9.8h, v11.8h, v29.8h
    gsbf_top v12.8h, v14.8h, v30.8h
    gsbf_top v13.8h, v15.8h, v31.8h

    gsbf_br_bot v2.8h, v16.8h, v18.8h, v23.8h, v24.8h
    gsbf_br_bot v3.8h, v17.8h, v19.8h, v23.8h, v25.8h
    gsbf_br_bot v6.8h,  v16.8h, v18.8h, v23.8h, v26.8h
    gsbf_br_bot v7.8h, v17.8h, v19.8h, v23.8h, v27.8h

    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_br_bot v10.8h, v16.8h, v18.8h, v23.8h, v28.8h
    gsbf_br_bot v11.8h, v17.8h, v19.8h, v23.8h, v29.8h
    gsbf_br_bot v14.8h, v16.8h, v18.8h, v23.8h, v30.8h
    gsbf_br_bot v15.8h, v17.8h, v19.8h, v23.8h, v31.8h

    // Reduction
    barrett v0.8h, v23.8h, v24.8h
    barrett v1.8h, v23.8h, v25.8h
    barrett v2.8h, v23.8h, v26.8h
    barrett v3.8h, v23.8h, v27.8h

    barrett v4.8h, v23.8h, v28.8h
    barrett v5.8h, v23.8h, v29.8h
    barrett v6.8h, v23.8h, v30.8h
    barrett v7.8h, v23.8h, v31.8h

    barrett v8.8h, v23.8h, v24.8h
    barrett v9.8h, v23.8h, v25.8h
    barrett v10.8h, v23.8h, v26.8h
    barrett v11.8h, v23.8h, v27.8h

    barrett v12.8h, v23.8h, v28.8h
    barrett v13.8h, v23.8h, v29.8h
    barrett v14.8h, v23.8h, v30.8h
    barrett v15.8h, v23.8h, v31.8h

    // End Layer 1

    // Layer 2
    transpose v0.8h, v1.8h, v2.8h, v3.8h, v24.8h, v25.8h, v26.8h, v27.8h
    transpose v4.8h, v5.8h, v6.8h, v7.8h, v28.8h, v29.8h, v30.8h, v31.8h
    transpose v8.8h, v9.8h, v10.8h, v11.8h, v24.8h, v25.8h, v26.8h, v27.8h
    transpose v12.8h, v13.8h, v14.8h, v15.8h, v28.8h, v29.8h, v30.8h, v31.8h

    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_top v0.8h, v1.8h, v24.8h
    gsbf_top v4.8h, v5.8h, v25.8h
    gsbf_top v8.8h, v9.8h, v26.8h
    gsbf_top v12.8h, v13.8h, v27.8h

    gsbf_top v2.8h, v3.8h, v28.8h
    gsbf_top v6.8h, v7.8h, v29.8h
    gsbf_top v10.8h, v11.8h, v30.8h
    gsbf_top v14.8h, v15.8h, v31.8h

    gsbf_br_bot v1.8h, v16.8h, v18.8h, v23.8h, v24.8h
    gsbf_br_bot v5.8h, v17.8h, v19.8h, v23.8h, v25.8h

    ld1.s16 {v16.8h, v17.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v18.8h, v19.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_br_bot v9.8h,  v16.8h, v18.8h, v23.8h, v26.8h
    gsbf_br_bot v13.8h, v17.8h, v19.8h, v23.8h, v27.8h

    ld1.s16 {v16.8h, v17.8h, v18.8h, v19.8h}, [ptr_invntt_br], #(32*2)
    ld1.s16 {v24.8h, v25.8h, v26.8h, v27.8h}, [ptr_invntt_qinv_br], #(32*2)

    gsbf_br_bot v3.8h, v16.8h, v24.8h, v23.8h, v28.8h
    gsbf_br_bot v7.8h, v17.8h, v25.8h, v23.8h, v29.8h
    gsbf_br_bot v11.8h, v18.8h, v26.8h, v23.8h, v30.8h
    gsbf_br_bot v15.8h, v19.8h, v27.8h, v23.8h, v31.8h

    // End Layer 2

    // Layer 3
    arrange v0.8h, v1.8h, v2.8h, v3.8h, v16.8h, v17.8h, v18.8h, v19.8h
    arrange v4.8h, v5.8h, v6.8h, v7.8h, v24.8h, v25.8h, v26.8h, v27.8h
    arrange v8.8h, v9.8h, v10.8h, v11.8h, v28.8h, v29.8h, v30.8h, v31.8h
    arrange v12.8h, v13.8h, v14.8h, v15.8h, v0.8h, v1.8h, v2.8h, v3.8h

    // v00 .req v16
    // v01 .req v17
    // v02 .req v18
    // v03 .req v19

    // v10 .req v24
    // v11 .req v25
    // v12 .req v26
    // v13 .req v27

    // v20 .req v28
    // v21 .req v29
    // v22 .req v30
    // v23 .req v31

    // v30 .req v0
    // v31 .req v1
    // v32 .req v2
    // v33 .req v3

    // t00 .req v4
    // t01 .req v5
    // t02 .req v6
    // t03 .req v7

    // t10 .req v8
    // t11 .req v9
    // t12 .req v10
    // t13 .req v11

    // neon_qmvq .req v23

    // v0 16-> 19
    // v1 24-> 27
    // v2 28 -> 31
    // v3 0 -> 3
    // t  4->7
    // t2 8->11
    // neon_qmvq v23

    ld1.s16 {v12.8h, v13.8h}, [ptr_invntt_br], #(16*2)
    ld1.s16 {v14.8h, v15.8h}, [ptr_invntt_qinv_br], #(16*2)

    gsbf_top v16.8h, v17.8h, v4.8h
    gsbf_top v18.8h, v19.8h, v5.8h
    gsbf_top v24.8h, v25.8h, v6.8h
    gsbf_top v26.8h, v27.8h, v7.8h

    gsbf_top v28.8h, v29.8h, v8.8h
    gsbf_top v30.8h, v31.8h, v9.8h
    gsbf_top v0.8h, v1.8h, v10.8h
    gsbf_top v2.8h, v3.8h, v11.8h

    gsbf_bri_bot v17.8h, v12.8h, v14.8h, 0, v23.8h, v4.8h
    gsbf_bri_bot v19.8h, v12.8h, v14.8h, 1, v23.8h, v5.8h
    gsbf_bri_bot v25.8h, v12.8h, v14.8h, 2, v23.8h, v6.8h
    gsbf_bri_bot v27.8h, v12.8h, v14.8h, 3, v23.8h, v7.8h

    gsbf_bri_bot v29.8h, v12.8h, v14.8h, 4, v23.8h, v8.8h
    gsbf_bri_bot v31.8h, v12.8h, v14.8h, 5, v23.8h, v9.8h
    gsbf_bri_bot v1.8h,  v12.8h, v14.8h, 6, v23.8h, v10.8h
    gsbf_bri_bot v3.8h,  v12.8h, v14.8h, 7, v23.8h, v11.8h

    // Reduction
    // v0
    barrett v16.8h, v23.8h, v4.8h
    
    // v1
    barrett v24.8h, v23.8h, v5.8h
    
    // v2
    barrett v28.8h, v23.8h, v6.8h
    
    // v3
    barrett v0.8h, v23.8h, v7.8h
    
    // End Layer 3

    // Layer 4
    // v0 16-> 19
    // v1 24-> 27
    // v2 28 -> 31
    // v3 0 -> 3
    // t  4->7
    // t2 8->11

    gsbf_top v16.8h, v18.8h, v4.8h
    gsbf_top v17.8h, v19.8h, v5.8h
    gsbf_top v24.8h, v26.8h, v6.8h
    gsbf_top v25.8h, v27.8h, v7.8h

    gsbf_top v28.8h, v30.8h, v8.8h
    gsbf_top v29.8h, v31.8h, v9.8h
    gsbf_top v0.8h, v2.8h, v10.8h
    gsbf_top v1.8h, v3.8h, v11.8h

    gsbf_bri_bot v18.8h, v13.8h, v15.8h, 0, v23.8h, v4.8h
    gsbf_bri_bot v19.8h, v13.8h, v15.8h, 0, v23.8h, v5.8h
    gsbf_bri_bot v26.8h, v13.8h, v15.8h, 1, v23.8h, v6.8h
    gsbf_bri_bot v27.8h, v13.8h, v15.8h, 1, v23.8h, v7.8h

    gsbf_bri_bot v30.8h, v13.8h, v15.8h, 2, v23.8h, v8.8h
    gsbf_bri_bot v31.8h, v13.8h, v15.8h, 2, v23.8h, v9.8h
    gsbf_bri_bot v2.8h, v13.8h, v15.8h, 3, v23.8h, v10.8h
    gsbf_bri_bot v3.8h, v13.8h, v15.8h, 3, v23.8h, v11.8h

    // v0
    barrett v16.8h, v23.8h, v4.8h
    barrett v17.8h, v23.8h, v5.8h
    barrett v18.8h, v23.8h, v6.8h
    barrett v19.8h, v23.8h, v7.8h
    
    // v1
    barrett v24.8h, v23.8h, v8.8h
    barrett v25.8h, v23.8h, v9.8h
    barrett v26.8h, v23.8h, v10.8h
    barrett v27.8h, v23.8h, v11.8h
    
    // v2
    barrett v28.8h, v23.8h, v4.8h
    barrett v29.8h, v23.8h, v5.8h
    barrett v30.8h, v23.8h, v6.8h
    barrett v31.8h, v23.8h, v7.8h
    
    // v3
    barrett v0.8h, v23.8h, v8.8h
    barrett v1.8h, v23.8h, v9.8h
    barrett v2.8h, v23.8h, v10.8h
    barrett v3.8h, v23.8h, v11.8h

    // End Layer 4

    // Layer 5
    // v0 16-> 19
    // v1 24-> 27
    // v2 28 -> 31
    // v3 0 -> 3
    // t  4->7
    // t2 8->11

    gsbf_top v16.8h, v24.8h, v4.8h
    gsbf_top v17.8h, v25.8h, v5.8h
    gsbf_top v18.8h, v26.8h, v6.8h
    gsbf_top v19.8h, v27.8h, v7.8h

    gsbf_top v28.8h, v0.8h, v8.8h
    gsbf_top v29.8h, v1.8h, v9.8h
    gsbf_top v30.8h, v2.8h, v10.8h
    gsbf_top v31.8h, v3.8h, v11.8h

    gsbf_bri_bot v24.8h, v13.8h, v15.8h, 4, v23.8h, v4.8h
    gsbf_bri_bot v25.8h, v13.8h, v15.8h, 4, v23.8h, v5.8h
    gsbf_bri_bot v26.8h, v13.8h, v15.8h, 4, v23.8h, v6.8h
    gsbf_bri_bot v27.8h, v13.8h, v15.8h, 4, v23.8h, v7.8h

    gsbf_bri_bot v0.8h, v13.8h, v15.8h, 5, v23.8h, v8.8h
    gsbf_bri_bot v1.8h, v13.8h, v15.8h, 5, v23.8h, v9.8h
    gsbf_bri_bot v2.8h, v13.8h, v15.8h, 5, v23.8h, v10.8h
    gsbf_bri_bot v3.8h, v13.8h, v15.8h, 5, v23.8h, v11.8h

    // End Layer 5

    // Layer 6
    // v0 16-> 19
    // v1 24-> 27
    // v2 28 -> 31
    // v3 0 -> 3
    // t  4->7
    // t2 8->11

    gsbf_top v16.8h, v28.8h, v4.8h
    gsbf_top v17.8h, v29.8h, v5.8h
    gsbf_top v18.8h, v30.8h, v6.8h
    gsbf_top v19.8h, v31.8h, v7.8h

    gsbf_top v24.8h, v0.8h, v8.8h
    gsbf_top v25.8h, v1.8h, v9.8h
    gsbf_top v26.8h, v2.8h, v10.8h
    gsbf_top v27.8h, v3.8h, v11.8h

    gsbf_bri_bot v28.8h, v13.8h, v15.8h, 6, v23.8h, v4.8h
    gsbf_bri_bot v29.8h, v13.8h, v15.8h, 6, v23.8h, v5.8h
    gsbf_bri_bot v30.8h, v13.8h, v15.8h, 6, v23.8h, v6.8h
    gsbf_bri_bot v31.8h, v13.8h, v15.8h, 6, v23.8h, v7.8h

    gsbf_bri_bot v0.8h, v13.8h, v15.8h, 6, v23.8h, v8.8h
    gsbf_bri_bot v1.8h, v13.8h, v15.8h, 6, v23.8h, v9.8h
    gsbf_bri_bot v2.8h, v13.8h, v15.8h, 6, v23.8h, v10.8h
    gsbf_bri_bot v3.8h, v13.8h, v15.8h, 6, v23.8h, v11.8h

    // Store

    st1.s16 {v16.8h, v17.8h, v18.8h, v19.8h}, [poly]
    st1.s16 {v24.8h, v25.8h, v26.8h, v27.8h}, [poly, #(32*2)]
    st1.s16 {v28.8h, v29.8h, v30.8h, v31.8h}, [poly, #(64*2)]
    st1.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly, #(96*2)]

    ret

// v0: 0, 1, 2, 3
// v1: 4, 5, 6, 7
// v2: 8, 9, 10, 11
// v3: 12, 13, 14, 15
// zl, zll: 16, 17
// zh, zhh: 18, 19
// neon_qmvq: 23*
// t: 24, 25, 26, 27
// t2: 28, 29, 30, 31
neon_falcon_inner_poly_invntt_2nd_loop:
    ld1.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly]
    ld1.s16 {v4.8h, v5.8h, v6.8h, v7.8h}, [poly, #(128*2)]
    ld1.s16 {v8.8h, v9.8h, v10.8h, v11.8h}, [poly, #(256*2)]
    ld1.s16 {v12.8h, v13.8h, v14.8h, v15.8h}, [poly, #(384*2)]

    // Reduction
    barrett v0.8h, v23.8h, v24.8h
    barrett v1.8h, v23.8h, v25.8h
    barrett v2.8h, v23.8h, v26.8h
    barrett v3.8h, v23.8h, v27.8h

    barrett v4.8h, v23.8h, v28.8h
    barrett v5.8h, v23.8h, v29.8h
    barrett v6.8h, v23.8h, v30.8h
    barrett v7.8h, v23.8h, v31.8h

    barrett v8.8h, v23.8h, v24.8h
    barrett v9.8h, v23.8h, v25.8h
    barrett v10.8h, v23.8h, v26.8h
    barrett v11.8h, v23.8h, v27.8h

    barrett v12.8h, v23.8h, v28.8h
    barrett v13.8h, v23.8h, v29.8h
    barrett v14.8h, v23.8h, v30.8h
    barrett v15.8h, v23.8h, v31.8h

    gsbf_top v0.8h, v4.8h, v24.8h
    gsbf_top v1.8h, v5.8h, v25.8h
    gsbf_top v2.8h, v6.8h, v26.8h
    gsbf_top v3.8h, v7.8h, v27.8h

    gsbf_top v8.8h, v12.8h, v28.8h
    gsbf_top v9.8h, v13.8h, v29.8h
    gsbf_top v10.8h, v14.8h, v30.8h
    gsbf_top v11.8h, v15.8h, v31.8h

    gsbf_bri_bot v4.8h, v16.8h, v18.8h, 0, v23.8h, v24.8h
    gsbf_bri_bot v5.8h, v16.8h, v18.8h, 0, v23.8h, v25.8h
    gsbf_bri_bot v6.8h, v16.8h, v18.8h, 0, v23.8h, v26.8h
    gsbf_bri_bot v7.8h, v16.8h, v18.8h, 0, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 1, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 1, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 1, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 1, v23.8h, v31.8h

    gsbf_top v0.8h, v8.8h, v24.8h
    gsbf_top v1.8h, v9.8h, v25.8h
    gsbf_top v2.8h, v10.8h, v26.8h
    gsbf_top v3.8h, v11.8h, v27.8h

    gsbf_top v4.8h, v12.8h, v28.8h
    gsbf_top v5.8h, v13.8h, v29.8h
    gsbf_top v6.8h, v14.8h, v30.8h
    gsbf_top v7.8h, v15.8h, v31.8h

    cmp ninv, 1
    b.eq 2nd_barmul

    gsbf_bri_bot v8.8h, v16.8h, v18.8h,  4, v23.8h, v24.8h
    gsbf_bri_bot v9.8h, v16.8h, v18.8h,  4, v23.8h, v25.8h
    gsbf_bri_bot v10.8h, v16.8h, v18.8h, 4, v23.8h, v26.8h
    gsbf_bri_bot v11.8h, v16.8h, v18.8h, 4, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 4, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 4, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 4, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 4, v23.8h, v31.8h

    b end_2nd_barmul

2nd_barmul:
    gsbf_bri_bot v8.8h, v16.8h, v18.8h, 2, v23.8h, v24.8h
    gsbf_bri_bot v9.8h, v16.8h, v18.8h, 2, v23.8h, v25.8h
    gsbf_bri_bot v10.8h, v16.8h, v18.8h, 2, v23.8h, v26.8h
    gsbf_bri_bot v11.8h, v16.8h, v18.8h, 2, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 2, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 2, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 2, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 2, v23.8h, v31.8h

    barmul_invntt v0.8h, v16.8h, v18.8h, 3, v24.8h
    barmul_invntt v1.8h, v16.8h, v18.8h, 3, v25.8h
    barmul_invntt v2.8h, v16.8h, v18.8h, 3, v26.8h
    barmul_invntt v3.8h, v16.8h, v18.8h, 3, v27.8h

    barmul_invntt v4.8h, v16.8h, v18.8h, 3, v28.8h
    barmul_invntt v5.8h, v16.8h, v18.8h, 3, v29.8h
    barmul_invntt v6.8h, v16.8h, v18.8h, 3, v30.8h
    barmul_invntt v7.8h, v16.8h, v18.8h, 3, v31.8h

end_2nd_barmul:

    barrett v0.8h, v23.8h, v24.8h
    barrett v1.8h, v23.8h, v25.8h
    barrett v2.8h, v23.8h, v26.8h
    barrett v3.8h, v23.8h, v27.8h

    barrett v4.8h, v23.8h, v28.8h
    barrett v5.8h, v23.8h, v29.8h
    barrett v6.8h, v23.8h, v30.8h
    barrett v7.8h, v23.8h, v31.8h

    st1.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly]
    st1.s16 {v4.8h, v5.8h, v6.8h, v7.8h}, [poly, #(128*2)]
    st1.s16 {v8.8h, v9.8h, v10.8h, v11.8h}, [poly, #(256*2)]
    st1.s16 {v12.8h, v13.8h, v14.8h, v15.8h}, [poly, #(384*2)]
    
    ret

neon_falcon_inner_poly_invntt_3rd_loop:
    ld1.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly]
    ld1.s16 {v4.8h, v5.8h, v6.8h, v7.8h}, [poly, #(128*2)]
    ld1.s16 {v8.8h, v9.8h, v10.8h, v11.8h}, [poly, #(256*2)]
    ld1.s16 {v12.8h, v13.8h, v14.8h, v15.8h}, [poly, #(384*2)]

    gsbf_top v0.8h, v4.8h, v24.8h
    gsbf_top v1.8h, v5.8h, v25.8h
    gsbf_top v2.8h, v6.8h, v26.8h
    gsbf_top v3.8h, v7.8h, v27.8h

    gsbf_top v8.8h, v12.8h, v28.8h
    gsbf_top v9.8h, v13.8h, v29.8h
    gsbf_top v10.8h, v14.8h, v30.8h
    gsbf_top v11.8h, v15.8h, v31.8h

    gsbf_bri_bot v4.8h, v16.8h, v18.8h, 0, v23.8h, v24.8h
    gsbf_bri_bot v5.8h, v16.8h, v18.8h, 0, v23.8h, v25.8h
    gsbf_bri_bot v6.8h, v16.8h, v18.8h, 0, v23.8h, v26.8h
    gsbf_bri_bot v7.8h, v16.8h, v18.8h, 0, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 1, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 1, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 1, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 1, v23.8h, v31.8h

    // Reduction
    barrett v0.8h, v23.8h, v24.8h
    barrett v1.8h, v23.8h, v25.8h
    barrett v2.8h, v23.8h, v26.8h
    barrett v3.8h, v23.8h, v27.8h

    barrett v4.8h, v23.8h, v28.8h
    barrett v5.8h, v23.8h, v29.8h
    barrett v6.8h, v23.8h, v30.8h
    barrett v7.8h, v23.8h, v31.8h

    barrett v8.8h, v23.8h, v24.8h
    barrett v9.8h, v23.8h, v25.8h
    barrett v10.8h, v23.8h, v26.8h
    barrett v11.8h, v23.8h, v27.8h

    barrett v12.8h, v23.8h, v28.8h
    barrett v13.8h, v23.8h, v29.8h
    barrett v14.8h, v23.8h, v30.8h
    barrett v15.8h, v23.8h, v31.8h

    gsbf_top v0.8h, v8.8h, v24.8h
    gsbf_top v1.8h, v9.8h, v25.8h
    gsbf_top v2.8h, v10.8h, v26.8h
    gsbf_top v3.8h, v11.8h, v27.8h

    gsbf_top v4.8h, v12.8h, v28.8h
    gsbf_top v5.8h, v13.8h, v29.8h
    gsbf_top v6.8h, v14.8h, v30.8h
    gsbf_top v7.8h, v15.8h, v31.8h

    cmp ninv, 1
    b.eq 3rd_barmul

    gsbf_bri_bot v8.8h, v16.8h, v18.8h,  4, v23.8h, v24.8h
    gsbf_bri_bot v9.8h, v16.8h, v18.8h,  4, v23.8h, v25.8h
    gsbf_bri_bot v10.8h, v16.8h, v18.8h, 4, v23.8h, v26.8h
    gsbf_bri_bot v11.8h, v16.8h, v18.8h, 4, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 4, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 4, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 4, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 4, v23.8h, v31.8h

    b end_3rd_barmul

3rd_barmul:
    gsbf_bri_bot v8.8h, v16.8h, v18.8h, 2, v23.8h, v24.8h
    gsbf_bri_bot v9.8h, v16.8h, v18.8h, 2, v23.8h, v25.8h
    gsbf_bri_bot v10.8h, v16.8h, v18.8h, 2, v23.8h, v26.8h
    gsbf_bri_bot v11.8h, v16.8h, v18.8h, 2, v23.8h, v27.8h

    gsbf_bri_bot v12.8h, v16.8h, v18.8h, 2, v23.8h, v28.8h
    gsbf_bri_bot v13.8h, v16.8h, v18.8h, 2, v23.8h, v29.8h
    gsbf_bri_bot v14.8h, v16.8h, v18.8h, 2, v23.8h, v30.8h
    gsbf_bri_bot v15.8h, v16.8h, v18.8h, 2, v23.8h, v31.8h

    barmul_invntt v0.8h, v16.8h, v18.8h, 3, v24.8h
    barmul_invntt v1.8h, v16.8h, v18.8h, 3, v25.8h
    barmul_invntt v2.8h, v16.8h, v18.8h, 3, v26.8h
    barmul_invntt v3.8h, v16.8h, v18.8h, 3, v27.8h

    barmul_invntt v4.8h, v16.8h, v18.8h, 3, v28.8h
    barmul_invntt v5.8h, v16.8h, v18.8h, 3, v29.8h
    barmul_invntt v6.8h, v16.8h, v18.8h, 3, v30.8h
    barmul_invntt v7.8h, v16.8h, v18.8h, 3, v31.8h

end_3rd_barmul:

    st1.s16 {v0.8h, v1.8h, v2.8h, v3.8h}, [poly]
    st1.s16 {v4.8h, v5.8h, v6.8h, v7.8h}, [poly, #(128*2)]
    st1.s16 {v8.8h, v9.8h, v10.8h, v11.8h}, [poly, #(256*2)]
    st1.s16 {v12.8h, v13.8h, v14.8h, v15.8h}, [poly, #(384*2)]
    
    ret

.text
.global neon_falcon_inner_poly_invntt
.global _neon_falcon_inner_poly_invntt



neon_falcon_inner_poly_invntt:
_neon_falcon_inner_poly_invntt
    
    push_stack

    ldp x19, x20, [sp, 48] // a
    ldp x21, x22, [sp, 32] // ntt_br
    ldp x23, x24, [sp, 16] // ntt_qinv_br
    ldp x25, x26, [sp]     // neon_qmvm

    ld1 {v23.8h}, 

    neon_falcon_inner_poly_invntt_1st_loop
    neon_falcon_inner_poly_invntt_1st_loop
    neon_falcon_inner_poly_invntt_1st_loop
    neon_falcon_inner_poly_invntt_1st_loop

    neon_falcon_inner_poly_invntt_2nd_loop
    neon_falcon_inner_poly_invntt_2nd_loop

    neon_falcon_inner_poly_invntt_3rd_loop
    neon_falcon_inner_poly_invntt_3rd_loop

    pop_stack

    ret