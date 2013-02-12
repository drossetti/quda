// *** CUDA DSLASH DAGGER ***

#define DSLASH_SHARED_FLOATS_PER_THREAD 0

// NB! Don't trust any MULTI_GPU code

#if (CUDA_VERSION >= 4010)
#define VOLATILE
#else
#define VOLATILE volatile
#endif
// input spinor
#ifdef SPINOR_DOUBLE
#define spinorFloat double
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I1.x
#define i01_im I1.y
#define i02_re I2.x
#define i02_im I2.y
#define i10_re I3.x
#define i10_im I3.y
#define i11_re I4.x
#define i11_im I4.y
#define i12_re I5.x
#define i12_im I5.y
#define i20_re I6.x
#define i20_im I6.y
#define i21_re I7.x
#define i21_im I7.y
#define i22_re I8.x
#define i22_im I8.y
#define i30_re I9.x
#define i30_im I9.y
#define i31_re I10.x
#define i31_im I10.y
#define i32_re I11.x
#define i32_im I11.y
#else
#define spinorFloat float
#define i00_re I0.x
#define i00_im I0.y
#define i01_re I0.z
#define i01_im I0.w
#define i02_re I1.x
#define i02_im I1.y
#define i10_re I1.z
#define i10_im I1.w
#define i11_re I2.x
#define i11_im I2.y
#define i12_re I2.z
#define i12_im I2.w
#define i20_re I3.x
#define i20_im I3.y
#define i21_re I3.z
#define i21_im I3.w
#define i22_re I4.x
#define i22_im I4.y
#define i30_re I4.z
#define i30_im I4.w
#define i31_re I5.x
#define i31_im I5.y
#define i32_re I5.z
#define i32_im I5.w
#endif // SPINOR_DOUBLE

// output spinor
VOLATILE spinorFloat o00_re;
VOLATILE spinorFloat o00_im;
VOLATILE spinorFloat o01_re;
VOLATILE spinorFloat o01_im;
VOLATILE spinorFloat o02_re;
VOLATILE spinorFloat o02_im;
VOLATILE spinorFloat o10_re;
VOLATILE spinorFloat o10_im;
VOLATILE spinorFloat o11_re;
VOLATILE spinorFloat o11_im;
VOLATILE spinorFloat o12_re;
VOLATILE spinorFloat o12_im;
VOLATILE spinorFloat o20_re;
VOLATILE spinorFloat o20_im;
VOLATILE spinorFloat o21_re;
VOLATILE spinorFloat o21_im;
VOLATILE spinorFloat o22_re;
VOLATILE spinorFloat o22_im;
VOLATILE spinorFloat o30_re;
VOLATILE spinorFloat o30_im;
VOLATILE spinorFloat o31_re;
VOLATILE spinorFloat o31_im;
VOLATILE spinorFloat o32_re;
VOLATILE spinorFloat o32_im;

#ifdef SPINOR_DOUBLE
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 16 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 8 // to avoid bank conflicts on G80 and GT200
#endif
#else
#if (__COMPUTE_CAPABILITY__ >= 200)
#define SHARED_STRIDE 32 // to avoid bank conflicts on Fermi
#else
#define SHARED_STRIDE 16 // to avoid bank conflicts on G80 and GT200
#endif
#endif

//#include "read_gauge.h"
#include "io_spinor.h"

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= param.threads) return;

int X, x1, x2, x3, x4, xs;

int boundaryCrossing;

#ifdef MULTI_GPU
int face_idx;
if (kernel_type == INTERIOR_KERNEL) {
#endif

// Inline by hand for the moment and assume even dimensions
//coordsFromIndex(X, x1, x2, x3, x4, sid, param.parity);

boundaryCrossing = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h);

X = 2*sid + (boundaryCrossing + param.parity) % 2;
x1 = X % X1;
x2 = (X/X1) % X2;
x3 = (X/(X1*X2)) % X3;
x4 = (X/(X1*X2*X3)) % X4;
xs = X/(X1*X2*X3*X4);

 o00_re = 0; o00_im = 0;
 o01_re = 0; o01_im = 0;
 o02_re = 0; o02_im = 0;
 o10_re = 0; o10_im = 0;
 o11_re = 0; o11_im = 0;
 o12_re = 0; o12_im = 0;
 o20_re = 0; o20_im = 0;
 o21_re = 0; o21_im = 0;
 o22_re = 0; o22_im = 0;
 o30_re = 0; o30_im = 0;
 o31_re = 0; o31_im = 0;
 o32_re = 0; o32_im = 0;

#ifdef MULTI_GPU
} else { // exterior kernel

const int dim = static_cast<int>(kernel_type);
const int face_volume = (param.threads >> 1); // volume of one face
const int face_num = (sid >= face_volume); // is this thread updating face 0 or 1
face_idx = sid - face_num*face_volume; // index into the respective face

// ghostOffset is scaled to include body (includes stride) and number of FloatN arrays (SPINOR_HOP)
// face_idx not sid since faces are spin projected and share the same volume index (modulo UP/DOWN reading)
//sp_idx = face_idx + param.ghostOffset[dim];

coordsFromDWFaceIndex<1>(sid, x1, x2, x3, x4, xs, face_idx, face_volume, dim, face_num, param.parity);

boundaryCrossing = sid/X1h + sid/(X2*X1h) + sid/(X3*X2*X1h);

X = 2*sid + (boundaryCrossing + param.parity) % 2;

READ_INTERMEDIATE_SPINOR(INTERTEX, sp_stride, sid, sid);

 o00_re = i00_re; o00_im = i00_im;
 o01_re = i01_re; o01_im = i01_im;
 o02_re = i02_re; o02_im = i02_im;
 o10_re = i10_re; o10_im = i10_im;
 o11_re = i11_re; o11_im = i11_im;
 o12_re = i12_re; o12_im = i12_im;
 o20_re = i20_re; o20_im = i20_im;
 o21_re = i21_re; o21_im = i21_im;
 o22_re = i22_re; o22_im = i22_im;
 o30_re = i30_re; o30_im = i30_im;
 o31_re = i31_re; o31_im = i31_im;
 o32_re = i32_re; o32_im = i32_im;
}
#endif // MULTI_GPU

// 5th dimension -- NB: not partitionable!
#ifdef MULTI_GPU
if(kernel_type == INTERIOR_KERNEL)
#endif
{
// 2 P_L = 2 P_- = ( ( +1, -1 ), ( -1, +1 ) )
  {
     int sp_idx = ( xs == 0 ? X+(Ls-1)*2*Vh : X-2*Vh ) / 2;

// read spinor from device memory
     READ_SPINOR( SPINORTEX, sp_stride, sp_idx, sp_idx );

     if ( xs != 0 )
     {
   o00_re += +i00_re-i20_re;   o00_im += +i00_im-i20_im;
   o01_re += +i01_re-i21_re;   o01_im += +i01_im-i21_im;
   o02_re += +i02_re-i22_re;   o02_im += +i02_im-i22_im;

   o10_re += +i10_re-i30_re;   o10_im += +i10_im-i30_im;
   o11_re += +i11_re-i31_re;   o11_im += +i11_im-i31_im;
   o12_re += +i12_re-i32_re;   o12_im += +i12_im-i32_im;

   o20_re += -i00_re+i20_re;   o20_im += -i00_im+i20_im;
   o21_re += -i01_re+i21_re;   o21_im += -i01_im+i21_im;
   o22_re += -i02_re+i22_re;   o22_im += -i02_im+i22_im;

   o30_re += -i10_re+i30_re;   o30_im += -i10_im+i30_im;
   o31_re += -i11_re+i31_re;   o31_im += -i11_im+i31_im;
   o32_re += -i12_re+i32_re;   o32_im += -i12_im+i32_im;
    }
    else
    {
   o00_re += -mferm*(+i00_re-i20_re);   o00_im += -mferm*(+i00_im-i20_im);
   o01_re += -mferm*(+i01_re-i21_re);   o01_im += -mferm*(+i01_im-i21_im);
   o02_re += -mferm*(+i02_re-i22_re);   o02_im += -mferm*(+i02_im-i22_im);

   o10_re += -mferm*(+i10_re-i30_re);   o10_im += -mferm*(+i10_im-i30_im);
   o11_re += -mferm*(+i11_re-i31_re);   o11_im += -mferm*(+i11_im-i31_im);
   o12_re += -mferm*(+i12_re-i32_re);   o12_im += -mferm*(+i12_im-i32_im);

   o20_re += -mferm*(-i00_re+i20_re);   o20_im += -mferm*(-i00_im+i20_im);
   o21_re += -mferm*(-i01_re+i21_re);   o21_im += -mferm*(-i01_im+i21_im);
   o22_re += -mferm*(-i02_re+i22_re);   o22_im += -mferm*(-i02_im+i22_im);

   o30_re += -mferm*(-i10_re+i30_re);   o30_im += -mferm*(-i10_im+i30_im);
   o31_re += -mferm*(-i11_re+i31_re);   o31_im += -mferm*(-i11_im+i31_im);
   o32_re += -mferm*(-i12_re+i32_re);   o32_im += -mferm*(-i12_im+i32_im);
    } // end if ( xs != 0 )
  } // end P_L

 // 2 P_R = 2 P_+ = ( ( +1, +1 ), ( +1, +1 ) )
  {
    int sp_idx = ( xs == Ls-1 ? X-(Ls-1)*2*Vh : X+2*Vh ) / 2;

// read spinor from device memory
    READ_SPINOR( SPINORTEX, sp_stride, sp_idx, sp_idx );

    if ( xs != Ls-1 )
    {
   o00_re += +i00_re+i20_re;   o00_im += +i00_im+i20_im;
   o01_re += +i01_re+i21_re;   o01_im += +i01_im+i21_im;
   o02_re += +i02_re+i22_re;   o02_im += +i02_im+i22_im;

   o10_re += +i10_re+i30_re;   o10_im += +i10_im+i30_im;
   o11_re += +i11_re+i31_re;   o11_im += +i11_im+i31_im;
   o12_re += +i12_re+i32_re;   o12_im += +i12_im+i32_im;

   o20_re += +i00_re+i20_re;   o20_im += +i00_im+i20_im;
   o21_re += +i01_re+i21_re;   o21_im += +i01_im+i21_im;
   o22_re += +i02_re+i22_re;   o22_im += +i02_im+i22_im;

   o30_re += +i10_re+i30_re;   o30_im += +i10_im+i30_im;
   o31_re += +i11_re+i31_re;   o31_im += +i11_im+i31_im;
   o32_re += +i12_re+i32_re;   o32_im += +i12_im+i32_im;
    }
    else
    {
   o00_re += -mferm*(+i00_re+i20_re);   o00_im += -mferm*(+i00_im+i20_im);
   o01_re += -mferm*(+i01_re+i21_re);   o01_im += -mferm*(+i01_im+i21_im);
   o02_re += -mferm*(+i02_re+i22_re);   o02_im += -mferm*(+i02_im+i22_im);

   o10_re += -mferm*(+i10_re+i30_re);   o10_im += -mferm*(+i10_im+i30_im);
   o11_re += -mferm*(+i11_re+i31_re);   o11_im += -mferm*(+i11_im+i31_im);
   o12_re += -mferm*(+i12_re+i32_re);   o12_im += -mferm*(+i12_im+i32_im);

   o20_re += -mferm*(+i00_re+i20_re);   o20_im += -mferm*(+i00_im+i20_im);
   o21_re += -mferm*(+i01_re+i21_re);   o21_im += -mferm*(+i01_im+i21_im);
   o22_re += -mferm*(+i02_re+i22_re);   o22_im += -mferm*(+i02_im+i22_im);

   o30_re += -mferm*(+i10_re+i30_re);   o30_im += -mferm*(+i10_im+i30_im);
   o31_re += -mferm*(+i11_re+i31_re);   o31_im += -mferm*(+i11_im+i31_im);
   o32_re += -mferm*(+i12_re+i32_re);   o32_im += -mferm*(+i12_im+i32_im);
    } // end if ( xs != Ls-1 )
  } // end P_R
} // end 5th dimension


#if defined MULTI_GPU && defined DSLASH_XPAY

int incomplete = 0; // Have all 8 contributions been computed for this site?

switch(kernel_type) { // intentional fall-through
case INTERIOR_KERNEL:
incomplete = incomplete || (param.commDim[3] && (x4==0 || x4==X4m1));
case EXTERIOR_KERNEL_T:
incomplete = incomplete || (param.commDim[2] && (x3==0 || x3==X3m1));
case EXTERIOR_KERNEL_Z:
incomplete = incomplete || (param.commDim[1] && (x2==0 || x2==X2m1));
case EXTERIOR_KERNEL_Y:
incomplete = incomplete || (param.commDim[0] && (x1==0 || x1==X1m1));
}

if (!incomplete)
#endif // MULTI_GPU
{

#ifdef DSLASH_XPAY

 READ_ACCUM(ACCUMTEX, sp_stride)

#ifdef SPINOR_DOUBLE
 o00_re = a*o00_re + accum0.x;
 o00_im = a*o00_im + accum0.y;
 o01_re = a*o01_re + accum1.x;
 o01_im = a*o01_im + accum1.y;
 o02_re = a*o02_re + accum2.x;
 o02_im = a*o02_im + accum2.y;
 o10_re = a*o10_re + accum3.x;
 o10_im = a*o10_im + accum3.y;
 o11_re = a*o11_re + accum4.x;
 o11_im = a*o11_im + accum4.y;
 o12_re = a*o12_re + accum5.x;
 o12_im = a*o12_im + accum5.y;
 o20_re = a*o20_re + accum6.x;
 o20_im = a*o20_im + accum6.y;
 o21_re = a*o21_re + accum7.x;
 o21_im = a*o21_im + accum7.y;
 o22_re = a*o22_re + accum8.x;
 o22_im = a*o22_im + accum8.y;
 o30_re = a*o30_re + accum9.x;
 o30_im = a*o30_im + accum9.y;
 o31_re = a*o31_re + accum10.x;
 o31_im = a*o31_im + accum10.y;
 o32_re = a*o32_re + accum11.x;
 o32_im = a*o32_im + accum11.y;
#else
 o00_re = a*o00_re + accum0.x;
 o00_im = a*o00_im + accum0.y;
 o01_re = a*o01_re + accum0.z;
 o01_im = a*o01_im + accum0.w;
 o02_re = a*o02_re + accum1.x;
 o02_im = a*o02_im + accum1.y;
 o10_re = a*o10_re + accum1.z;
 o10_im = a*o10_im + accum1.w;
 o11_re = a*o11_re + accum2.x;
 o11_im = a*o11_im + accum2.y;
 o12_re = a*o12_re + accum2.z;
 o12_im = a*o12_im + accum2.w;
 o20_re = a*o20_re + accum3.x;
 o20_im = a*o20_im + accum3.y;
 o21_re = a*o21_re + accum3.z;
 o21_im = a*o21_im + accum3.w;
 o22_re = a*o22_re + accum4.x;
 o22_im = a*o22_im + accum4.y;
 o30_re = a*o30_re + accum4.z;
 o30_im = a*o30_im + accum4.w;
 o31_re = a*o31_re + accum5.x;
 o31_im = a*o31_im + accum5.y;
 o32_re = a*o32_re + accum5.z;
 o32_im = a*o32_im + accum5.w;
#endif // SPINOR_DOUBLE

#endif // DSLASH_XPAY
}

// write spinor field back to device memory
WRITE_SPINOR(sp_stride);

// undefine to prevent warning when precision is changed
#undef spinorFloat
#undef SHARED_STRIDE

#undef i00_re
#undef i00_im
#undef i01_re
#undef i01_im
#undef i02_re
#undef i02_im
#undef i10_re
#undef i10_im
#undef i11_re
#undef i11_im
#undef i12_re
#undef i12_im
#undef i20_re
#undef i20_im
#undef i21_re
#undef i21_im
#undef i22_re
#undef i22_im
#undef i30_re
#undef i30_im
#undef i31_re
#undef i31_im
#undef i32_re
#undef i32_im

#undef VOLATILE
