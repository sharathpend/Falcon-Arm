=========
// Layer 5
0, 1, 16, 17, 32, 33, 48, 49
64,65 80, 81, 128, 129, 160,161, 192,193, 224,225
RE =   0,  16 | IM = 256, 272 | gm =  32 
RE =   1,  17 | IM = 257, 273 | gm =  32 
RE =   2,  18 | IM = 258, 274 | gm =  32 
RE =   3,  19 | IM = 259, 275 | gm =  32 
RE =  32,  48 | IM = 288, 304 | gm =  34 
RE =  33,  49 | IM = 289, 305 | gm =  34 
RE =  34,  50 | IM = 290, 306 | gm =  34 
RE =  35,  51 | IM = 291, 307 | gm =  34 
=========

// Layer 6
RE =   0,  32 | IM = 256, 288 | gm =  16 
RE =   1,  33 | IM = 257, 289 | gm =  16 
RE =   2,  34 | IM = 258, 290 | gm =  16 
RE =   3,  35 | IM = 259, 291 | gm =  16 
RE =  16,  48 | IM = 272, 304 | gm =  16 
RE =  17,  49 | IM = 273, 305 | gm =  16 
RE =  18,  50 | IM = 274, 306 | gm =  16 
RE =  19,  51 | IM = 275, 307 | gm =  16 


for (int distance = 1 << 4; j < FALCON_N; distance<<= 2)
{
    for (int j = 0; j < 32; j+= 4)
    {
        // Layer 5
        x_re[0] <= 0...3 
        y_re[0] <= 16...19
        x_re[1] <= 32...35
        y_re[1] <= 48...51

        subx4(v_re, x_re, y_re);
        subx4(v_im, x_im, y_im);

        addx4(x_re, x_re, y_re);
        addx4(x_im, x_im, y_im);

        // complex mul

        // layer 6
        x_re[0] <= 0...3
        y_re[1] <= 32...35
        x_re[1] <= 16...19
        y_re[1] <= 48...51

        subx4(v_re, x_re, y_re);
        subx4(v_im, x_im, y_im);

        addx4(x_re, x_re, y_re);
        addx4(x_im, x_im, y_im);
        // Complex mul
    }
}


========
// Layer 7 
0, 1, 64, 65, 128, 129, 192, 193
RE =   0,  64 | IM = 256, 320 | gm =   8 
RE =   1,  65 | IM = 257, 321 | gm =   8 
RE =   2,  66 | IM = 258, 322 | gm =   8 
RE =   3,  67 | IM = 259, 323 | gm =   8 
RE = 128, 192 | IM = 384, 448 | gm =  10 
RE = 129, 193 | IM = 385, 449 | gm =  10 
RE = 130, 194 | IM = 386, 450 | gm =  10 
RE = 131, 195 | IM = 387, 451 | gm =  10 

// Layer 8 
RE =   0, 128 | IM = 256, 384 | gm =   4 
RE =   1, 129 | IM = 257, 385 | gm =   4 
RE =   2, 130 | IM = 258, 386 | gm =   4 
RE =   3, 131 | IM = 259, 387 | gm =   4 
RE =  64, 192 | IM = 320, 448 | gm =   4 
RE =  65, 193 | IM = 321, 449 | gm =   4 
RE =  66, 194 | IM = 322, 450 | gm =   4 
RE =  67, 195 | IM = 323, 451 | gm =   4 
========

for (int distance = 1 << 6; j < FALCON_N; distance<<= 2)
{
    for (int j = 0; j < 32; j+= 4)
    {
        // Layer 5
        x_re[0] <= 0...3 
        y_re[0] <= 64...67
        x_re[1] <= 128...131
        y_re[1] <= 192...195

        subx4(v_re, x_re, y_re);
        subx4(v_im, x_im, y_im);

        addx4(x_re, x_re, y_re);
        addx4(x_im, x_im, y_im);

        // complex mul

        // layer 6
        x_re[0] <= 0...3
        y_re[1] <= 128...131
        x_re[1] <= 64...67
        y_re[1] <= 192...195

        subx4(v_re, x_re, y_re);
        subx4(v_im, x_im, y_im);

        addx4(x_re, x_re, y_re);
        addx4(x_im, x_im, y_im);
        // Complex mul
    }
}
