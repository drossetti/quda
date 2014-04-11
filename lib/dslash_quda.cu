#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_INTER
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
//#define DIRECT_ACCESS_CLOVER
#endif // GPU_WILSON_DIRAC

//these are access control for staggered action
#ifdef GPU_STAGGERED_DIRAC
#if (__COMPUTE_CAPABILITY__ >= 300) // Kepler works best with texture loads only
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#elif (__COMPUTE_CAPABILITY__ >= 200)
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#else
#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#endif
#endif // GPU_STAGGERED_DIRAC

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>

enum KernelType {
  INTERIOR_KERNEL = 5,
  EXTERIOR_KERNEL_X = 0,
  EXTERIOR_KERNEL_Y = 1,
  EXTERIOR_KERNEL_Z = 2,
  EXTERIOR_KERNEL_T = 3
};

namespace quda {

  struct DslashParam {
    int threads; // the desired number of active threads
    int parity;  // Even-Odd or Odd-Even
    int commDim[QUDA_MAX_DIM]; // Whether to do comms or not
    int ghostDim[QUDA_MAX_DIM]; // Whether a ghost zone has been allocated for a given dimension
    int ghostOffset[QUDA_MAX_DIM+1];
    int ghostNormOffset[QUDA_MAX_DIM+1];
    int X[4];
    KernelType kernel_type; //is it INTERIOR_KERNEL, EXTERIOR_KERNEL_X/Y/Z/T
    int sp_stride; // spinor stride
#ifdef GPU_STAGGERED_DIRAC
    int gauge_stride;
    int long_gauge_stride;
    int fat_link_max;
#endif 

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t inTex;
    cudaTextureObject_t inTexNorm;
    cudaTextureObject_t xTex;
    cudaTextureObject_t xTexNorm;
    cudaTextureObject_t outTex;
    cudaTextureObject_t outTexNorm;
    cudaTextureObject_t gauge0Tex; // also applies to fat gauge
    cudaTextureObject_t gauge1Tex; // also applies to fat gauge
    cudaTextureObject_t longGauge0Tex;
    cudaTextureObject_t longGauge1Tex;
    cudaTextureObject_t longPhase0Tex;
    cudaTextureObject_t longPhase1Tex;
    cudaTextureObject_t cloverTex;
    cudaTextureObject_t cloverNormTex;
    cudaTextureObject_t cloverInvTex;
    cudaTextureObject_t cloverInvNormTex;
#endif
  };

  DslashParam dslashParam;

  // these are set in initDslashConst
  int Vspatial;

  static cudaEvent_t packEnd[Nstream];
  static cudaEvent_t gatherStart[Nstream];
  static cudaEvent_t gatherEnd[Nstream];
  static cudaEvent_t scatterStart[Nstream];
  static cudaEvent_t scatterEnd[Nstream];
  static cudaEvent_t dslashStart;
  static cudaEvent_t dslashEnd;

  static FaceBuffer *face[2];
  static cudaColorSpinorField *inSpinor;
  static FullClover *inClover = NULL;
  static FullClover *inCloverInv = NULL;

  // For tuneLaunch() to uniquely identify a suitable set of launch parameters, we need copies of a few of
  // the constants set by initDslashConstants().
  static struct {
    int x[4];
    int Ls;
    unsigned long long VolumeCB() { return x[0]*x[1]*x[2]*x[3]/2; }
    // In the future, we may also want to add gauge_fixed, sp_stride, ga_stride, cl_stride, etc.
  } dslashConstants;

  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple calls to cudaMemcpy()
  static bool kernelPackT = false;

  void setKernelPackT(bool packT) { kernelPackT = packT; }

  bool getKernelPackT() { return kernelPackT; }


  //these params are needed for twisted mass (in particular, for packing twisted spinor)
  static bool twistPack = false;

  void setTwistPack(bool flag) { twistPack = flag; }
  bool getTwistPack() { return twistPack; }

#ifdef MULTI_GPU
  static double twist_a = 0.0;
  static double twist_b = 0.0;
#endif

#include <dslash_textures.h>
#include <dslash_constants.h>

#if defined(DIRECT_ACCESS_LINK) || defined(DIRECT_ACCESS_WILSON_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_ACCUM) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_INTER) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_CLOVER)

  static inline __device__ float short2float(short a) {
    return (float)a/MAX_SHORT;
  }

  static inline __device__ short float2short(float c, float a) {
    return (short)(a*c*MAX_SHORT);
  }

  static inline __device__ short4 float42short4(float c, float4 a) {
    return make_short4(float2short(c, a.x), float2short(c, a.y), float2short(c, a.z), float2short(c, a.w));
  }

  static inline __device__ float4 short42float4(short4 a) {
    return make_float4(short2float(a.x), short2float(a.y), short2float(a.z), short2float(a.w));
  }

  static inline __device__ float2 short22float2(short2 a) {
    return make_float2(short2float(a.x), short2float(a.y));
  }
#endif // DIRECT_ACCESS inclusions

  // Enable shared memory dslash for Fermi architecture
  #define SHARED_WILSON_DSLASH
  //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#include <pack_face_def.h>        // kernels for packing the ghost zones and general indexing
#include <staggered_dslash_def.h> // staggered Dslash kernels
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#include <dw_dslash_def.h>        // Domain Wall kernels
#include <tm_dslash_def.h>        // Twisted Mass kernels
#include <tm_core.h>              // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone
#include <tm_ndeg_dslash_def.h>   // Non-degenerate twisted Mass
#include <tmc_dslash_def.h>       // Twisted Clover kernels

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef CLOVER_SHARED_FLOATS_PER_THREAD
#define CLOVER_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef NDEGTM_SHARED_FLOATS_PER_THREAD
#define NDEGTM_SHARED_FLOATS_PER_THREAD 0
#endif




  void setFace(const FaceBuffer &Face1, const FaceBuffer &Face2) {
    face[0] = (FaceBuffer*)&(Face1); 
    face[1] = (FaceBuffer*)&(Face2); // nasty
  }

  static int it = 0;

  void createDslashEvents()
  {
    // add cudaEventDisableTiming for lower sync overhead
    for (int i=0; i<Nstream; i++) {
      cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&dslashStart, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&dslashEnd, cudaEventDisableTiming);

    checkCudaError();
  }


  void destroyDslashEvents()
  {
    for (int i=0; i<Nstream; i++) {
      cudaEventDestroy(packEnd[i]);
      cudaEventDestroy(gatherStart[i]);
      cudaEventDestroy(gatherEnd[i]);
      cudaEventDestroy(scatterStart[i]);
      cudaEventDestroy(scatterEnd[i]);
    }

    cudaEventDestroy(dslashStart);
    cudaEventDestroy(dslashEnd);

    checkCudaError();
  }


#define MORE_GENERIC_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x==0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  }


#define MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x==0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 13 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {								\
      FUNC ## 9 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_13) {			\
      FUNC ## 13 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_9) {			\
      FUNC ## 9 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }                                                                   \
  }


#ifndef MULTI_GPU

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  default:								\
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#define GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  default:								\
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }


#else

#define GENERIC_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_X:						\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Y:						\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Z:						\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_T:						\
    MORE_GENERIC_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  }

#define GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_X:						\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Y:						\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Z:						\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_T:						\
    MORE_GENERIC_STAGGERED_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  }


#endif

  // macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      } else {								\
    GENERIC_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      }

  // macro used for staggered dslash
#define STAGGERED_DSLASH(gridDim, blockDim, shared, stream, param, ...)	\
  GENERIC_STAGGERED_DSLASH(staggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__)

#define IMPROVED_STAGGERED_DSLASH(gridDim, blockDim, shared, stream, param, ...) \
  GENERIC_STAGGERED_DSLASH(improvedStaggeredDslash, , Axpy, gridDim, blockDim, shared, stream, param, __VA_ARGS__) 

#define MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
    FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
    FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
    FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
  }									

#ifndef MULTI_GPU

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  default:								\
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#else

#define GENERIC_ASYM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_X:						\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Y:						\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Z:						\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_T:						\
    MORE_GENERIC_ASYM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  }

#endif

  // macro used for dslash types with dagger kernel defined (Wilson, domain wall, etc.)
#define ASYM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...) \
  if (!dagger) {							\
    GENERIC_ASYM_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      } else {								\
    GENERIC_ASYM_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      }



//macro used for twisted mass dslash:

#define MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, kernel_type, gridDim, blockDim, shared, stream, param,  ...) \
  if (x == 0 && d == 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else {								\
      FUNC ## 8 ## DAG ## Twist ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x != 0 && d == 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Twist ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## Twist ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else if (x == 0 && d != 0) {								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__ , param); \
    } else {								\
      FUNC ## 8 ## DAG ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  } else{								\
    if (reconstruct == QUDA_RECONSTRUCT_NO) {				\
      FUNC ## 18 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_12) {			\
      FUNC ## 12 ## DAG ## X ## Kernel<kernel_type><<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    } else if (reconstruct == QUDA_RECONSTRUCT_8) {			\
      FUNC ## 8 ## DAG ## X ## Kernel<kernel_type> <<<gridDim, blockDim, shared, stream>>> ( __VA_ARGS__, param); \
    }									\
  }

#ifndef MULTI_GPU

#define GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  default:								\
    errorQuda("KernelType %d not defined for single GPU", param.kernel_type); \
  }

#else

#define GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, gridDim, blockDim, shared, stream, param,  ...) \
  switch(param.kernel_type) {						\
  case INTERIOR_KERNEL:							\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, INTERIOR_KERNEL,   gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_X:						\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_X, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Y:						\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Y, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_Z:						\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_Z, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  case EXTERIOR_KERNEL_T:						\
    MORE_GENERIC_NDEG_TM_DSLASH(FUNC, DAG, X, EXTERIOR_KERNEL_T, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      break;								\
  }

#endif

#define NDEG_TM_DSLASH(FUNC, gridDim, blockDim, shared, stream, param, ...)	\
  if (!dagger) {							\
    GENERIC_NDEG_TM_DSLASH(FUNC, , Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      } else {								\
    GENERIC_NDEG_TM_DSLASH(FUNC, Dagger, Xpay, gridDim, blockDim, shared, stream, param, __VA_ARGS__) \
      }
//end of tm dslash macro


  // Use an abstract class interface to drive the different CUDA dslash
  // kernels. All parameters are curried into the derived classes to
  // allow a simple interface.
  class DslashCuda : public Tunable {
  protected:
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    const cudaColorSpinorField *x;
    char *saveOut, *saveOutNorm;

    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

  public:
    DslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
	       const cudaColorSpinorField *x) 
      : out(out), in(in), x(x), saveOut(0), saveOutNorm(0) { }
    virtual ~DslashCuda() { }
    virtual TuneKey tuneKey() const;
    std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }
    virtual int Nface() { return 2; }

    virtual void preTune()
    {
      if (dslashParam.kernel_type < 5) { // exterior kernel
	saveOut = new char[in->Bytes()];
	cudaMemcpy(saveOut, out->V(), in->Bytes(), cudaMemcpyDeviceToHost);
	if (out->Precision() == QUDA_HALF_PRECISION) {
	  saveOutNorm = new char[in->NormBytes()];
	  cudaMemcpy(saveOutNorm, out->Norm(), in->NormBytes(), cudaMemcpyDeviceToHost);
	}
      }
    }

    virtual void postTune()
    {
      if (dslashParam.kernel_type < 5) { // exterior kernel
	cudaMemcpy(out->V(), saveOut, in->Bytes(), cudaMemcpyHostToDevice);
	delete[] saveOut;
	if (out->Precision() == QUDA_HALF_PRECISION) {
	  cudaMemcpy(out->Norm(), saveOutNorm, in->NormBytes(), cudaMemcpyHostToDevice);
	  delete[] saveOutNorm;
	}
      }
    }

  };

  TuneKey DslashCuda::tuneKey() const
  {
    std::stringstream vol, aux;
  
    vol << dslashConstants.x[0] << "x";
    vol << dslashConstants.x[1] << "x";
    vol << dslashConstants.x[2] << "x";
    vol << dslashConstants.x[3];

    aux << "type=";
#ifdef MULTI_GPU
    char comm[5], ghost[5];
    switch (dslashParam.kernel_type) {
    case INTERIOR_KERNEL: aux << "interior"; break;
    case EXTERIOR_KERNEL_X: aux << "exterior_x"; break;
    case EXTERIOR_KERNEL_Y: aux << "exterior_y"; break;
    case EXTERIOR_KERNEL_Z: aux << "exterior_z"; break;
    case EXTERIOR_KERNEL_T: aux << "exterior_t"; break;
    }
    for (int i=0; i<4; i++) {
      comm[i] = (dslashParam.commDim[i] ? '1' : '0');
      ghost[i] = (dslashParam.ghostDim[i] ? '1' : '0');
    }
    comm[4] = '\0'; ghost[4] = '\0';
    aux << ",comm=" << comm;
    if (dslashParam.kernel_type == INTERIOR_KERNEL) {
      aux << ",ghost=" << ghost;
    }
#else
    aux << "single-GPU";
#endif // MULTI_GPU

    return TuneKey(vol.str(), typeid(*this).name(), aux.str());
  }

  /** This derived class is specifically for driving the Dslash kernels
      that use shared memory blocking.  This only applies on Fermi and
      upwards, and only for the interior kernels. */
#if (__COMPUTE_CAPABILITY__ >= 200 && defined(SHARED_WILSON_DSLASH)) 
  class SharedDslashCuda : public DslashCuda {
  protected:
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; } // FIXME: this isn't quite true, but works
    bool advanceSharedBytes(TuneParam &param) const { 
      if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceSharedBytes(param);
      else return false;
    } // FIXME - shared memory tuning only supported on exterior kernels

    /** Helper function to set the shared memory size from the 3-d block size */
    int sharedBytes(const dim3 &block) const { 
      int warpSize = 32; // FIXME - query from device properties
      int block_xy = block.x*block.y;
      if (block_xy % warpSize != 0) block_xy = ((block_xy / warpSize) + 1)*warpSize;
      return block_xy*block.z*sharedBytesPerThread();
    }

    /** Helper function to set the 3-d grid size from the 3-d block size */
    dim3 createGrid(const dim3 &block) const {
      unsigned int gx = ((dslashConstants.x[0]/2)*dslashConstants.x[3] + block.x - 1) / block.x;
      unsigned int gy = (dslashConstants.x[1] + block.y - 1 ) / block.y;	
      unsigned int gz = (dslashConstants.x[2] + block.z - 1) / block.z;
      return dim3(gx, gy, gz);
    }

    /** Advance the 3-d block size. */
    bool advanceBlockDim(TuneParam &param) const {
      if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::advanceBlockDim(param);
      const unsigned int min_threads = 2;
      const unsigned int max_threads = 512; // FIXME: use deviceProp.maxThreadsDim[0];
      const unsigned int max_shared = 16384*3; // FIXME: use deviceProp.sharedMemPerBlock;
    
      // set the x-block dimension equal to the entire x dimension
      bool set = false;
      dim3 blockInit = param.block;
      blockInit.z++;
      for (unsigned bx=blockInit.x; bx<=dslashConstants.x[0]/2; bx++) {
	//unsigned int gx = (dslashConstants.x[0]*dslashConstants.x[3] + bx - 1) / bx;
	for (unsigned by=blockInit.y; by<=dslashConstants.x[1]; by++) {
	  unsigned int gy = (dslashConstants.x[1] + by - 1 ) / by;	
	
	  if (by > 1 && (by%2) != 0) continue; // can't handle odd blocks yet except by=1
	
	  for (unsigned bz=blockInit.z; bz<=dslashConstants.x[2]; bz++) {
	    unsigned int gz = (dslashConstants.x[2] + bz - 1) / bz;
	  
	    if (bz > 1 && (bz%2) != 0) continue; // can't handle odd blocks yet except bz=1
	    if (bx*by*bz > max_threads) continue;
	    if (bx*by*bz < min_threads) continue;
	    // can't yet handle the last block properly in shared memory addressing
	    if (by*gy != dslashConstants.x[1]) continue;
	    if (bz*gz != dslashConstants.x[2]) continue;
	    if (sharedBytes(dim3(bx, by, bz)) > max_shared) continue;

	    param.block = dim3(bx, by, bz);	  
	    set = true; break;
	  }
	  if (set) break;
	  blockInit.z = 1;
	}
	if (set) break;
	blockInit.y = 1;
      }

      if (param.block.x > dslashConstants.x[0]/2 && param.block.y > dslashConstants.x[1] &&
	  param.block.z > dslashConstants.x[2] || !set) {
	//||sharedBytesPerThread()*param.block.x > max_shared) {
	param.block = dim3(dslashConstants.x[0]/2, 1, 1);
	return false;
      } else { 
	param.grid = createGrid(param.block);
	param.shared_bytes = sharedBytes(param.block);
	return true; 
      }
    
    }

  public:
    SharedDslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x) : DslashCuda(out, in, x) { ; }
    virtual ~SharedDslashCuda() { ; }
    std::string paramString(const TuneParam &param) const // override and print out grid as well
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "grid=(" << param.grid.x << "," << param.grid.y << "," << param.grid.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      if (dslashParam.kernel_type != INTERIOR_KERNEL) return DslashCuda::initTuneParam(param);

      param.block = dim3(dslashConstants.x[0]/2, 1, 1);
      param.grid = createGrid(param.block);
      param.shared_bytes = sharedBytes(param.block);
    }

    /** Sets default values for when tuning is disabled - this is guaranteed to work, but will be slow */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      if (dslashParam.kernel_type != INTERIOR_KERNEL) DslashCuda::defaultTuneParam(param);
      else initTuneParam(param);
    }
  };
#else /** For pre-Fermi architectures */
  class SharedDslashCuda : public DslashCuda {
  public:
    SharedDslashCuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x) : DslashCuda(out, in, x) { }
    virtual ~SharedDslashCuda() { }
  };
#endif


  template <typename sFloat, typename gFloat>
  class WilsonDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const int dagger;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200) // Fermi uses shared memory for common input
      if (dslashParam.kernel_type == INTERIOR_KERNEL) { // Interior kernels use shared memory for common iunput
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else { // Exterior kernels use no shared memory
	return 0;
      }
#else // Pre-Fermi uses shared memory only for pseudo-registers
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
    }

  public:
    WilsonDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		     const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x, const double a, const int dagger)
      : SharedDslashCuda(out, in, x), gauge0(gauge0), gauge1(gauge1), 
	reconstruct(reconstruct), dagger(dagger), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
    }

    virtual ~WilsonDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str();
      if (x) key.aux += ",Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DSLASH(dslash, tp.grid, tp.block, tp.shared_bytes, stream, 
	     dslashParam, (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a);
    }

    long long flops() const { return (x ? 1368ll : 1320ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
  };

  template <typename sFloat, typename gFloat, typename cFloat>
  class CloverDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const cFloat *clover;
    const float *cloverNorm;
    const int dagger;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
	return 0;
      }
#else
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
    }
  public:
    CloverDslashCuda(cudaColorSpinorField *out,  const gFloat *gauge0, const gFloat *gauge1, 
		     const QudaReconstructType reconstruct, const cFloat *clover, 
		     const float *cloverNorm, const cudaColorSpinorField *in, 
		     const cudaColorSpinorField *x, const double a, const int dagger)
      : SharedDslashCuda(out, in, x), gauge0(gauge0), gauge1(gauge1), clover(clover),
	cloverNorm(cloverNorm), reconstruct(reconstruct), dagger(dagger), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
    }
    virtual ~CloverDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str();
      if (x) key.aux += ",Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DSLASH(cloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cloverNorm, 
	     (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a);
    }

    long long flops() const { return (x ? 1872ll : 1824ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
  };

  template <typename sFloat, typename gFloat, typename cFloat>
  class AsymCloverDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const cFloat *clover;
    const float *cloverNorm;
    const int dagger;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
	return 0;
      }
#else
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
    }

  public:
    AsymCloverDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
			 const QudaReconstructType reconstruct, const cFloat *clover, 
			 const float *cloverNorm, const cudaColorSpinorField *in,
			 const cudaColorSpinorField *x, const double a, const int dagger)
      : SharedDslashCuda(out, in, x), gauge0(gauge0), gauge1(gauge1), clover(clover),
	cloverNorm(cloverNorm), reconstruct(reconstruct), dagger(dagger), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x);
      if (!x) errorQuda("Asymmetric clover dslash only defined for Xpay");
    }
    virtual ~AsymCloverDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str() + ",Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ASYM_DSLASH(asymCloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
		  (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cloverNorm, 
		  (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)x, (float*)x->Norm(), a);
    }

    long long flops() const { return 1872ll * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
  };

  void setTwistParam(double &a, double &b, const double &kappa, const double &mu, 
		     const int dagger, const QudaTwistGamma5Type twist) {
    if (twist == QUDA_TWIST_GAMMA5_DIRECT) {
      a = 2.0 * kappa * mu;
      b = 1.0;
    } else if (twist == QUDA_TWIST_GAMMA5_INVERSE) {
      a = -2.0 * kappa * mu;
      b = 1.0 / (1.0 + a*a);
    } else {
      errorQuda("Twist type %d not defined\n", twist);
    }
    if (dagger) a *= -1.0;

  }

  template <typename sFloat, typename gFloat>
  class TwistedDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const QudaTwistDslashType dslashType;
    const int dagger;
    double a, b, c, d;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
        int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
        return ((in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS) ? DSLASH_SHARED_FLOATS_PER_THREAD * reg_size : NDEGTM_SHARED_FLOATS_PER_THREAD * reg_size);
      } else {
        return 0;
      }
#else
     int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
     return ((in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS) ? DSLASH_SHARED_FLOATS_PER_THREAD * reg_size : NDEGTM_SHARED_FLOATS_PER_THREAD * reg_size);
#endif
    }

  public:
    TwistedDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		      const QudaReconstructType reconstruct, const cudaColorSpinorField *in,  const cudaColorSpinorField *x, 
                      const QudaTwistDslashType dslashType, const double kappa, const double mu, 
		      const double epsilon, const double k, const int dagger)
      : SharedDslashCuda(out, in, x),gauge0(gauge0), gauge1(gauge1), 
	reconstruct(reconstruct), dslashType(dslashType), dagger(dagger)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;
    }
    virtual ~TwistedDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon, dslash_type;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str();

      switch(dslashType){
        case QUDA_DEG_TWIST_INV_DSLASH:
        key.aux += ",TwistInvDslash";
        break;
        case QUDA_DEG_DSLASH_TWIST_INV:
        key.aux += ",";
        break;
        case QUDA_DEG_DSLASH_TWIST_XPAY:
        key.aux += ",DslashTwist";
        break;
        case QUDA_NONDEG_DSLASH:
        key.aux += ",NdegDslash";
        break;
      }
      if (x) key.aux += "Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
        errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  
      switch(dslashType){
        case QUDA_DEG_TWIST_INV_DSLASH:
          DSLASH(twistedMassTwistInvDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
        break;
        case QUDA_DEG_DSLASH_TWIST_INV:
          DSLASH(twistedMassDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
        break;
        case QUDA_DEG_DSLASH_TWIST_XPAY:
          DSLASH(twistedMassDslashTwist, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)x->V(), (float*)x->Norm());
        break;
        case QUDA_NONDEG_DSLASH:
          NDEG_TM_DSLASH(twistedNdegMassDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, c, d, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
        break;
        default: errorQuda("Invalid twisted mass dslash type");
      }
    }

    long long flops() const { return (x ? 1416ll : 1392ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
  };

  template <typename sFloat, typename gFloat, typename cFloat>
  class TwistedCloverDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const QudaTwistCloverDslashType dslashType;
    const int dagger;
    double a, b, c, d;
    const cFloat *clover;
    const float *cNorm;
    const cFloat *cloverInv;
    const float *cNrm2;

  protected:
    unsigned int sharedBytesPerThread() const
    {
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (dslashParam.kernel_type == INTERIOR_KERNEL) {
        int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
        return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else {
        return 0;
      }
#else
     int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
     return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
#endif
    }

  public:
    TwistedCloverDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		      const QudaReconstructType reconstruct, const cFloat *clover, const float *cNorm,
		      const cFloat *cloverInv, const float *cNrm2, const cudaColorSpinorField *in,
		      const cudaColorSpinorField *x, const QudaTwistCloverDslashType dslashType, const double kappa,
		      const double mu, const double epsilon, const double k, const int dagger)
      : SharedDslashCuda(out, in, x),gauge0(gauge0), gauge1(gauge1), clover(clover),
	cNorm(cNorm), cloverInv(cloverInv), cNrm2(cNrm2),
	reconstruct(reconstruct), dslashType(dslashType), dagger(dagger)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
      a = kappa;
      b = mu;
      c = epsilon;
      d = k;
    }
    virtual ~TwistedCloverDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon, dslash_type;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str();

      switch(dslashType){
        case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
        key.aux += ",CloverTwistInvDslash";
        break;
        case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
        key.aux += ",";
        break;
        case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
        key.aux += ",DslashCloverTwist";
        break;
      }
      if (x) key.aux += "Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
        errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
  
      switch(dslashType){

        case QUDA_DEG_CLOVER_TWIST_INV_DSLASH:
          DSLASH(twistedCloverInvDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
        break;
        case QUDA_DEG_DSLASH_CLOVER_TWIST_INV:
          DSLASH(twistedCloverDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0));
        break;
        case QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY:
          DSLASH(twistedCloverDslashTwist, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, clover, cNorm, cloverInv, cNrm2,
	     (sFloat*)in->V(), (float*)in->Norm(), a, b, (sFloat*)x->V(), (float*)x->Norm());
        break;
        default: errorQuda("Invalid twisted clover dslash type");
      }
    }

    long long flops() const { return (x ? 1416ll : 1392ll) * dslashConstants.VolumeCB(); } // FIXME for multi-GPU
  };

  template <typename sFloat, typename gFloat>
  class DomainWallDslashCuda : public DslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const QudaReconstructType reconstruct;
    const int dagger;
    const double mferm;
    const double a;

    bool checkGrid(TuneParam &param) const {
      if (param.grid.x > deviceProp.maxGridSize[0] || param.grid.y > deviceProp.maxGridSize[1]) {
	warningQuda("Autotuner is skipping blockDim=(%u,%u,%u), gridDim=(%u,%u,%u) because lattice volume is too large",
		    param.block.x, param.block.y, param.block.z, 
		    param.grid.x, param.grid.y, param.grid.z);
	return false;
      } else {
	return true;
      }
    }

  protected:
    bool advanceBlockDim(TuneParam &param) const
    {
      const unsigned int max_shared = 16384; // FIXME: use deviceProp.sharedMemPerBlock;
      const int step[2] = { deviceProp.warpSize, 1 };
      bool advance[2] = { false, false };

      // first try to advance block.x
      param.block.x += step[0];
      if (param.block.x > deviceProp.maxThreadsDim[0] || 
	  sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	advance[0] = false;
	param.block.x = step[0]; // reset block.x
      } else {
	advance[0] = true; // successfully advanced block.x
      }

      if (!advance[0]) {  // if failed to advance block.x, now try block.y
	param.block.y += step[1];

	if (param.block.y > in->X(4) || 
	  sharedBytesPerThread()*param.block.x*param.block.y > max_shared) {
	  advance[1] = false;
	  param.block.y = step[1]; // reset block.x
	} else {
	  advance[1] = true; // successfully advanced block.y
	}
      }

      if (advance[0] || advance[1]) {
	param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			   (in->X(4)+param.block.y-1) / param.block.y, 1);

	bool advance = true;
	if (!checkGrid(param)) advance = advanceBlockDim(param);
	return advance;
      } else {
	return false;
      }
    }

    unsigned int sharedBytesPerThread() const { return 0; }
  
  public:
    DomainWallDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
			 const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
			 const cudaColorSpinorField *x, const double mferm, 
			 const double a, const int dagger)
      : DslashCuda(out, in, x), gauge0(gauge0), gauge1(gauge1), mferm(mferm), 
	reconstruct(reconstruct), dagger(dagger), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }
    virtual ~DomainWallDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      bool ok = true;
      if (!checkGrid(param)) ok = advanceBlockDim(param);
      if (!ok) errorQuda("Lattice volume is too large for even the largest blockDim");
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (dslashParam.threads+param.block.x-1) / param.block.x, 
			 (in->X(4)+param.block.y-1) / param.block.y, 1);
      bool ok = true;
      if (!checkGrid(param)) ok = advanceBlockDim(param);
      if (!ok) errorQuda("Lattice volume is too large for even the largest blockDim");
    }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream ls, recon;
      ls << dslashConstants.Ls;
      recon << reconstruct;
      key.volume += "x" + ls.str();
      key.aux += ",reconstruct=" + recon.str();
      if (x) key.aux += ",Xpay";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      DSLASH(domainWallDslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam,
	     (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, 
	     (sFloat*)in->V(), (float*)in->Norm(), mferm, (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a);
    }

    long long flops() const { // FIXME for multi-GPU
      long long bulk = (dslashConstants.Ls-2)*(dslashConstants.VolumeCB()/dslashConstants.Ls);
      long long wall = 2*dslashConstants.VolumeCB()/dslashConstants.Ls;
      return (x ? 1368ll : 1320ll)*dslashConstants.VolumeCB()*dslashConstants.Ls + 96ll*bulk + 120ll*wall;
    }
  };


  template<typename T> struct RealType {};
  template<> struct RealType<double2> { typedef double type; };
  template<> struct RealType<float2> { typedef float type; };
  template<> struct RealType<float4> { typedef float type; };
  template<> struct RealType<short2> { typedef short type; };
  template<> struct RealType<short4> { typedef short type; };

  template <typename sFloat, typename fatGFloat, typename longGFloat, typename phaseFloat>
  class StaggeredDslashCuda : public DslashCuda {

  private:
    const fatGFloat *fat0, *fat1;
    const longGFloat *long0, *long1;
    const phaseFloat *phase0, *phase1;
    const QudaReconstructType reconstruct;
    const int dagger;
    const double a;
    QudaDslashType type;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return 6 * reg_size;
    }

  public:
    StaggeredDslashCuda(cudaColorSpinorField *out, const fatGFloat *fat0, const fatGFloat *fat1,
				const longGFloat *long0, const longGFloat *long1,
				const phaseFloat *phase0, const phaseFloat *phase1, 
				const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
				const cudaColorSpinorField *x, const double a, const int dagger)
      : DslashCuda(out, in, x), fat0(fat0), fat1(fat1), long0(long0), long1(long1), phase0(phase0), phase1(phase1), 
	reconstruct(reconstruct), dagger(dagger), a(a), type(long0 ? QUDA_ASQTAD_DSLASH : QUDA_STAGGERED_DSLASH)
    { 
      bindSpinorTex<sFloat>(in, out, x);
    }

    virtual ~StaggeredDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    TuneKey tuneKey() const
    {
      TuneKey key = DslashCuda::tuneKey();
      std::stringstream recon;
      recon << reconstruct;
      key.aux += ",reconstruct=" + recon.str();
      if (x) key.aux += ",Axpy";
      return key;
    }

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if (type == QUDA_STAGGERED_DSLASH) {
	STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
			 (sFloat*)out->V(), (float*)out->Norm(), fat0, fat1, 
			 (sFloat*)in->V(), (float*)in->Norm(), 
			 (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
      } else {
	IMPROVED_STAGGERED_DSLASH(gridDim, tp.block, tp.shared_bytes, stream, dslashParam,
				  (sFloat*)out->V(), (float*)out->Norm(), 
				  fat0, fat1, long0, long1, phase0, phase1, 
				  (sFloat*)in->V(), (float*)in->Norm(), 
				  (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a); 
      }

    }

    int Nface() { return type == QUDA_STAGGERED_DSLASH ? 2 : 6; } 

    long long flops() const { 
      long long flops;
      if (type == QUDA_STAGGERED_DSLASH) 
	flops = (x ? 666ll : 654ll) * dslashConstants.VolumeCB();
      else 
	flops = (x ? 1158ll : 1146ll) * dslashConstants.VolumeCB(); 
      return flops;
    } 
  };

  int gatherCompleted[Nstream];
  int previousDir[Nstream];
  int commsCompleted[Nstream];
  int dslashCompleted[Nstream];
  int commDimTotal;

  /**
   * Initialize the arrays used for the dynamic scheduling.
   */
  void inline initDslashCommsPattern() {
    for (int i=0; i<Nstream-1; i++) {
#ifndef GPU_COMMS
      gatherCompleted[i] = 0;
#else
      gatherCompleted[i] = 1;      
#endif
      commsCompleted[i] = 0;
      dslashCompleted[i] = 0;
    }
    gatherCompleted[Nstream-1] = 1;
    commsCompleted[Nstream-1] = 1;

    //   We need to know which was the previous direction in which
    //   communication was issued, since we only query a given event /
    //   comms call after the previous the one has successfully
    //   completed.
    for (int i=3; i>=0; i--) {
      if (dslashParam.commDim[i]) {
	int prev = Nstream-1;
	for (int j=3; j>i; j--) if (dslashParam.commDim[j]) prev = 2*j;
	previousDir[2*i + 1] = prev;
	previousDir[2*i + 0] = 2*i + 1; // always valid
      }
    }

    // this tells us how many events / comms occurances there are in
    // total.  Used for exiting the while loop
    commDimTotal = 0;
    for (int i=3; i>=0; i--) commDimTotal += dslashParam.commDim[i];
#ifndef GPU_COMMS
    commDimTotal *= 4; // 2 from pipe length, 2 from direction
#else
    commDimTotal *= 2; // 2 from pipe length, 2 from direction
#endif
  }

#define PROFILE(f, profile, idx)		\
  profile.Start(idx);				\
  f;						\
  profile.Stop(idx); 

  void dslashCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    profile.Start(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    initDslashCommsPattern();
    // Record the start of the dslash
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);

    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
	PROFILE(face[it]->recvStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
      } 
    }


    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
	{ pack = true; break; }

    // Initialize pack from source spinor

    if (inCloverInv == NULL) {
      PROFILE(face[it]->pack(*inSpinor, 1-parity, dagger, streams, twist_a, twist_b), 
	      profile, QUDA_PROFILE_PACK_KERNEL);
    } else {
      PROFILE(face[it]->pack(*inSpinor, *inClover, *inCloverInv, 1-parity, dagger,
	      streams, twist_a, twist_b), profile, QUDA_PROFILE_PACK_KERNEL);
    }

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[Nstream-1]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }

    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
	cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

	PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	// Initialize host transfer from source spinor
	PROFILE(face[it]->gather(*inSpinor, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

	// Record the end of the gathering
	PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
		profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }
#endif

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

    int completeSum = 0;
    while (completeSum < commDimTotal) {
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;
      
	for (int dir=1; dir>=0; dir--) {

	  // Query if gather has completed
	  if (!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]) { 
	    //CUresult event_test;
	    //event_test = cuEventQuery(gatherEnd[2*i+dir]);
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), 
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(face[it]->sendStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }
	
	  // Query if comms has finished
	  if (!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] &&
	      gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = face[it]->commsQuery(2*i+dir), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    
	      // Scatter into the end zone
	      // Both directions use the same stream
	      PROFILE(face[it]->scatter(*inSpinor, dagger, 2*i+dir), 
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }

	}
	 
	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
	if (!dslashCompleted[2*i] && commsCompleted[2*i] && commsCompleted[2*i+1] ) {
	  // Record the end of the scattering
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		  profile, QUDA_PROFILE_EVENT_RECORD);

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	  
	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  
	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  dslashCompleted[2*i] = 1;
	}

      }
    
    }
    it = (it^1);
#endif // MULTI_GPU

    profile.Stop(QUDA_PROFILE_TOTAL);
  }

  void dslashCuda2(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
		  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    profile.Start(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    // Record the start of the dslash if doing communication in T and not kernel packing
    if (dslashParam.commDim[3] && !(getKernelPackT() || getTwistPack())) {
      PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }
	
    initDslashCommsPattern();
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
        PROFILE(inSpinor->recvStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
      }
    }
    bool pack = false;
    for (int i=3; i>=0; i--) 
      if (dslashParam.commDim[i] && (i!=3 || getKernelPackT() || getTwistPack())) 
	{ pack = true; break; }

    // Initialize pack from source spinor
    if (inCloverInv == NULL) {
      PROFILE(inSpinor->pack(dslash.Nface()/2, 1-parity, dagger, streams, twist_a, twist_b),
	      profile, QUDA_PROFILE_PACK_KERNEL);
    } else {
      PROFILE(inSpinor->pack(*inClover, *inCloverInv, dslash.Nface()/2, 1-parity, dagger, streams, twist_a),
	      profile, QUDA_PROFILE_PACK_KERNEL);
    }

    if (pack) {
      // Record the end of the packing
      PROFILE(cudaEventRecord(packEnd[0], streams[Nstream-1]), 
	      profile, QUDA_PROFILE_EVENT_RECORD);
    }

#ifndef GPU_COMMS
    for(int i = 3; i >=0; i--){
      if (!dslashParam.commDim[i]) continue;

      for (int dir=1; dir>=0; dir--) {
	cudaEvent_t &event = (i!=3 || getKernelPackT() || getTwistPack()) ? packEnd[0] : dslashStart;

	PROFILE(cudaStreamWaitEvent(streams[2*i+dir], event, 0), 
		profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

	// Initialize host transfer from source spinor
	PROFILE(inSpinor->gather(dslash.Nface()/2, dagger, 2*i+dir), profile, QUDA_PROFILE_GATHER);

	// Record the end of the gathering
	PROFILE(cudaEventRecord(gatherEnd[2*i+dir], streams[2*i+dir]), 
		profile, QUDA_PROFILE_EVENT_RECORD);
      }
    }
#endif // GPU_COMMS

#endif // MULTI_GPU

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

#ifdef GPU_COMMS
    bool pack_event = false;
    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;

      if ((i!=3 || getKernelPackT() || getTwistPack()) && !pack_event) {
	cudaEventSynchronize(packEnd[0]);
	pack_event = true;
      } else {
	cudaEventSynchronize(dslashStart);
      }

      for (int dir=1; dir>=0; dir--) {	
	PROFILE(inSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	inSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger); // do a comms query to ensure MPI has begun
      }
    }
#endif

    int completeSum = 0;
    while (completeSum < commDimTotal) {
      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;
      
	for (int dir=1; dir>=0; dir--) {
	
#ifndef GPU_COMMS
	  // Query if gather has completed
	  if (!gatherCompleted[2*i+dir] && gatherCompleted[previousDir[2*i+dir]]) { 
	    PROFILE(cudaError_t event_test = cudaEventQuery(gatherEnd[2*i+dir]), 
		    profile, QUDA_PROFILE_EVENT_QUERY);

	    if (cudaSuccess == event_test) {
	      gatherCompleted[2*i+dir] = 1;
	      completeSum++;
	      PROFILE(inSpinor->sendStart(dslash.Nface()/2, 2*i+dir, dagger), profile, QUDA_PROFILE_COMMS_START);
	    }
	  }
#endif
	
	  // Query if comms has finished
	  if (!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]] &&
	      gatherCompleted[2*i+dir]) {
	    PROFILE(int comms_test = inSpinor->commsQuery(dslash.Nface()/2, 2*i+dir, dagger), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    
	      // Scatter into the end zone
	      // Both directions use the same stream
#ifndef GPU_COMMS
	      PROFILE(inSpinor->scatter(dslash.Nface()/2, dagger, 2*i+dir), 
		      profile, QUDA_PROFILE_SCATTER);
#endif
	    }
	  }

	} // dir=0,1
	 
	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
	if (!dslashCompleted[2*i] && commsCompleted[2*i] && commsCompleted[2*i+1] ) {
	  // Record the end of the scattering
#ifndef GPU_COMMS
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		  profile, QUDA_PROFILE_EVENT_RECORD);

	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
#endif

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	  
	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  dslashCompleted[2*i] = 1;
	}

      }
    
    }
    it = (it^1);
#endif // MULTI_GPU
    profile.Stop(QUDA_PROFILE_TOTAL);
  }

  /**
     Variation of multi-gpu dslash where the packing kernel writes
     buffers directly to host memory
  */
  void dslashZeroCopyCuda(DslashCuda &dslash, const size_t regSize, const int parity, const int dagger, 
			  const int volume, const int *faceVolumeCB, TimeProfile &profile) {
    profile.Start(QUDA_PROFILE_TOTAL);

    dslashParam.parity = parity;
    dslashParam.kernel_type = INTERIOR_KERNEL;
    dslashParam.threads = volume;

#ifdef MULTI_GPU
    initDslashCommsPattern();
    for(int i=3; i>=0; i--){
      if(!dslashParam.commDim[i]) continue;
      for(int dir=1; dir>=0; dir--){
	PROFILE(face[it]->recvStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);    
      }
    }


    setKernelPackT(true);

    // Record the end of the packing
    PROFILE(cudaEventRecord(dslashStart, streams[Nstream-1]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);

    PROFILE(cudaStreamWaitEvent(streams[0], dslashStart, 0), 
	    profile, QUDA_PROFILE_STREAM_WAIT_EVENT);

    // Initialize pack from source spinor
    PROFILE(face[it]->pack(*inSpinor, 1-parity, dagger, streams, true, twist_a, twist_b), 
	    profile, QUDA_PROFILE_PACK_KERNEL);

    // Record the end of the packing
    PROFILE(cudaEventRecord(packEnd[0], streams[0]), 
	    profile, QUDA_PROFILE_EVENT_RECORD);
#endif

    PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

#ifdef MULTI_GPU

    int doda=0;
    while (doda++>=0) {
      PROFILE(cudaError_t event_test = cudaEventQuery(packEnd[0]), 
	      profile, QUDA_PROFILE_EVENT_QUERY);
      if (event_test == cudaSuccess) doda=-1;
    }

    for (int i=3; i>=0; i--) {
      if (!dslashParam.commDim[i]) continue;
      for (int dir=1; dir>=0; dir--) {
	PROFILE(face[it]->sendStart(2*i+dir), profile, QUDA_PROFILE_COMMS_START);    
      }
    }


    int completeSum = 0;
    commDimTotal /= 2; // pipe is shorter for zero-variant

    while (completeSum < commDimTotal) {

      for (int i=3; i>=0; i--) {
	if (!dslashParam.commDim[i]) continue;
      
	for (int dir=1; dir>=0; dir--) {
	
	  // Query if comms have finished
	  if (!commsCompleted[2*i+dir] && commsCompleted[previousDir[2*i+dir]]) {
	    PROFILE(int comms_test = face[it]->commsQuery(2*i+dir), 
		    profile, QUDA_PROFILE_COMMS_QUERY);
	    if (comms_test) { 
	      commsCompleted[2*i+dir] = 1;
	      completeSum++;
	    
	      // Scatter into the end zone
	      // Both directions use the same stream
	      PROFILE(face[it]->scatter(*inSpinor, dagger, 2*i+dir), 
		      profile, QUDA_PROFILE_SCATTER);
	    }
	  }

	}
	 
	// enqueue the boundary dslash kernel as soon as the scatters have been enqueued
	if (!dslashCompleted[2*i] && commsCompleted[2*i] && commsCompleted[2*i+1] ) {
	  // Record the end of the scattering
	  PROFILE(cudaEventRecord(scatterEnd[2*i], streams[2*i]), 
		  profile, QUDA_PROFILE_EVENT_RECORD);

	  dslashParam.kernel_type = static_cast<KernelType>(i);
	  dslashParam.threads = dslash.Nface()*faceVolumeCB[i]; // updating 2 or 6 faces
	  
	  // wait for scattering to finish and then launch dslash
	  PROFILE(cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[2*i], 0), 
		  profile, QUDA_PROFILE_STREAM_WAIT_EVENT);
	  
	  // all faces use this stream
	  PROFILE(dslash.apply(streams[Nstream-1]), profile, QUDA_PROFILE_DSLASH_KERNEL);

	  dslashCompleted[2*i] = 1;
	}

      }
    
    }
    it = (it^1);
#endif // MULTI_GPU

    profile.Stop(QUDA_PROFILE_TOTAL);
  }

  // Wilson wrappers
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, 
			const int parity, const int dagger, const cudaColorSpinorField *x, const double &k, 
			const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_WILSON_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge %d and spinor %d precision not supported", 
		gauge.Precision(), in->Precision());

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new WilsonDslashCuda<double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
						      gauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new WilsonDslashCuda<float4, float4>(out, (float4*)gauge0, (float4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new WilsonDslashCuda<short4, short4>(out, (short4*)gauge0, (short4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    }
    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

  }

  void cloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover cloverInv,
			const cudaColorSpinorField *in, const int parity, const int dagger, 
			const cudaColorSpinorField *x, const double &a, const int *commOverride,
			TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_CLOVER_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *cloverP, *cloverNormP;
    QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

    if (in->Precision() != clover_prec)
      errorQuda("Mixing clover and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new CloverDslashCuda<double2, double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
							       gauge.Reconstruct(), (double2*)cloverP, 
							       (float*)cloverNormP, in, x, a, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new CloverDslashCuda<float4, float4, float4>(out, (float4*)gauge0, (float4*)gauge1,
							    gauge.Reconstruct(), (float4*)cloverP,
							    (float*)cloverNormP, in, x, a, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new CloverDslashCuda<short4, short4, short4>(out, (short4*)gauge0, (short4*)gauge1,
							    gauge.Reconstruct(), (short4*)cloverP,
							    (float*)cloverNormP, in, x, a, dagger);
    }

    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
    unbindGaugeTex(gauge);
    unbindCloverTex(cloverInv);

    checkCudaError();
#else
    errorQuda("Clover dslash has not been built");
#endif

  }


  void asymCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover cloverInv,
			    const cudaColorSpinorField *in, const int parity, const int dagger, 
			    const cudaColorSpinorField *x, const double &a, const int *commOverride,
			    TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_CLOVER_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *cloverP, *cloverNormP;
    QudaPrecision clover_prec = bindCloverTex(cloverInv, parity, &cloverP, &cloverNormP);

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

    if (in->Precision() != clover_prec)
      errorQuda("Mixing clover and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new AsymCloverDslashCuda<double2, double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
								   gauge.Reconstruct(), (double2*)cloverP, 
								   (float*)cloverNormP, in, x, a, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new AsymCloverDslashCuda<float4, float4, float4>(out, (float4*)gauge0, (float4*)gauge1, 
								gauge.Reconstruct(), (float4*)cloverP, 
								(float*)cloverNormP, in, x, a, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new AsymCloverDslashCuda<short4, short4, short4>(out, (short4*)gauge0, (short4*)gauge1, 
								gauge.Reconstruct(), (short4*)cloverP, 
								(float*)cloverNormP, in, x, a, dagger);
    }

    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
    unbindGaugeTex(gauge);
    unbindCloverTex(cloverInv);

    checkCudaError();
#else
    errorQuda("Clover dslash has not been built");
#endif

  }

  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			     const cudaColorSpinorField *in, const int parity, const int dagger, 
			     const cudaColorSpinorField *x, const QudaTwistDslashType type, const double &kappa, const double &mu, 
			     const double &epsilon, const double &k,  const int *commOverride,
			     TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
  #ifdef GPU_TWISTED_MASS_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  
    int ghost_threads[4] = {0};
    int bulk_threads = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->Volume() : in->Volume() / 2;
  
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
      ghost_threads[i] = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->GhostFace()[i] : in->GhostFace()[i] / 2;
    }

#ifdef MULTI_GPU
    if(type == QUDA_DEG_TWIST_INV_DSLASH){
        setTwistPack(true);
        twist_a = kappa; 
        twist_b = mu;
    }
#endif

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
        errorQuda("Mixing gauge and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new TwistedDslashCuda<double2,double2>(out, (double2*)gauge0,(double2*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedDslashCuda<float4,float4>(out, (float4*)gauge0,(float4*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);

    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedDslashCuda<short4,short4>(out, (short4*)gauge0,(short4*)gauge1, gauge.Reconstruct(), in, x, type, kappa, mu, epsilon, k, dagger);
    }

    dslashCuda(*dslash, regSize, parity, dagger, bulk_threads, ghost_threads, profile);

    delete dslash;
#ifdef MULTI_GPU
    if(type == QUDA_DEG_TWIST_INV_DSLASH){
        setTwistPack(false);
        twist_a = 0.0; 
        twist_b = 0.0;
    }
#endif

    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Twisted mass dslash has not been built");
#endif
  }

  void twistedCloverDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover *clover, const FullClover *cloverInv,
			     const cudaColorSpinorField *in, const int parity, const int dagger, 
			     const cudaColorSpinorField *x, const QudaTwistCloverDslashType type, const double &kappa, const double &mu, 
			     const double &epsilon, const double &k,  const int *commOverride,
			     TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inClover = (FullClover*) clover;
    inCloverInv = (FullClover*) cloverInv;
  #ifdef GPU_TWISTED_CLOVER_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
  
    int ghost_threads[4] = {0};
    int bulk_threads = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->Volume() : in->Volume() / 2;
  
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
      ghost_threads[i] = ((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) ? in->GhostFace()[i] : in->GhostFace()[i] / 2;
    }
    twist_a	= 2.*mu*kappa;
/*
#ifdef MULTI_GPU
    if(type == QUDA_DEG_CLOVER_TWIST_INV_DSLASH){
        setTwistPack(true);
        twist_a = kappa; 
        twist_b = mu;
    }
#endif
*/
    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    void *cloverP, *cloverNormP, *cloverInvP, *cloverInvNormP;
    QudaPrecision clover_prec = bindTwistedCloverTex(*clover, *cloverInv, parity, &cloverP, &cloverNormP, &cloverInvP, &cloverInvNormP);

    if (in->Precision() != clover_prec)
      errorQuda("Mixing clover and spinor precision not supported");

    if (in->Precision() != gauge.Precision())
        errorQuda("Mixing gauge and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new TwistedCloverDslashCuda<double2,double2,double2>(out, (double2*)gauge0,(double2*)gauge1, gauge.Reconstruct(), (double2*)cloverP, (float*)cloverNormP,
						     (double2*)cloverInvP, (float*)cloverInvNormP, in, x, type, kappa, mu, epsilon, k, dagger);

      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new TwistedCloverDslashCuda<float4,float4,float4>(out, (float4*)gauge0,(float4*)gauge1, gauge.Reconstruct(), (float4*)cloverP, (float*)cloverNormP,
						   (float4*)cloverInvP, (float*)cloverInvNormP, in, x, type, kappa, mu, epsilon, k, dagger);

    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new TwistedCloverDslashCuda<short4,short4,short4>(out, (short4*)gauge0,(short4*)gauge1, gauge.Reconstruct(), (short4*)cloverP, (float*)cloverNormP,
						   (short4*)cloverInvP, (float*)cloverInvNormP, in, x, type, kappa, mu, epsilon, k, dagger);
    }

//    dslashCuda(*dslash, regSize, parity, dagger, bulk_threads, ghost_threads, profile);
    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
/*
#ifdef MULTI_GPU
    if(type == QUDA_DEG_CLOVER_TWIST_INV_DSLASH){
        setTwistPack(false);
        twist_a = 0.0; 
        twist_b = 0.0;
    }
#endif
*/
    unbindGaugeTex(gauge);
    unbindTwistedCloverTex(*clover);

    checkCudaError();
#else
    errorQuda("Twisted clover dslash has not been built");
#endif
  }

  void domainWallDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			    const cudaColorSpinorField *in, const int parity, const int dagger, 
			    const cudaColorSpinorField *x, const double &m_f, const double &k2, 
			    const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

    dslashParam.parity = parity;

#ifdef GPU_DOMAIN_WALL_DIRAC
    //currently splitting in space-time is impelemented:
    int dirs = 4;
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i = 0;i < dirs; i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }  

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge and spinor precision not supported");

    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new DomainWallDslashCuda<double2,double2>(out, (double2*)gauge0, (double2*)gauge1, 
							 gauge.Reconstruct(), in, x, m_f, k2, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new DomainWallDslashCuda<float4,float4>(out, (float4*)gauge0, (float4*)gauge1, 
						       gauge.Reconstruct(), in, x, m_f, k2, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new DomainWallDslashCuda<short4,short4>(out, (short4*)gauge0, (short4*)gauge1, 
						       gauge.Reconstruct(), in, x, m_f, k2, dagger);
    }

    // the parameters passed to dslashCuda must be 4-d volume and 3-d
    // faces because Ls is added as the y-dimension in thread space
    int ghostFace[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) ghostFace[i] = in->GhostFace()[i] / in->X(4);
    dslashCuda(*dslash, regSize, parity, dagger, in->Volume() / in->X(4), ghostFace, profile);

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Domain wall dslash has not been built");
#endif
  }

  void staggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, 
			   const cudaColorSpinorField *in, const int parity, 
			   const int dagger, const cudaColorSpinorField *x,
			   const double &k, const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

  
    dslashParam.parity = parity;
    dslashParam.sp_stride = in->Stride();
    dslashParam.gauge_stride = gauge.Stride();
    dslashParam.fat_link_max = gauge.LinkMax(); // May need to use this in the preconditioning step 
                                                // in the solver for the improved staggered action


    for(int i=0;i<4;i++){
      dslashParam.X[i] = in->X()[i];
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }
    dslashParam.X[0] *= 2; // because color spinor fields are defined on a half lattice
    void *gauge0, *gauge1;
    bindFatGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision()) {
      errorQuda("Mixing precisions gauge=%d and spinor=%d not supported",
		gauge.Precision(), in->Precision());
    }
    
    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new StaggeredDslashCuda<double2, double2, double2, double>
	(out, (double2*)gauge0, (double2*)gauge1, 0, 0, 0, 0, gauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2, float4, float>
	(out, (float2*)gauge0, (float2*)gauge1, 0, 0, 0, 0, gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2, short4, short>
	(out, (short2*)gauge0, (short2*)gauge1, 0, 0, 0, 0, gauge.Reconstruct(), in, x, k, dagger);
    }

    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
    unbindFatGaugeTex(gauge);

    checkCudaError();
  
#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

  void
  improvedStaggeredDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &fatGauge, 
			      const cudaGaugeField &longGauge, const cudaColorSpinorField *in,
			      const int parity, const int dagger, const cudaColorSpinorField *x,
			      const double &k, const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL

#ifdef GPU_STAGGERED_DIRAC

#ifdef MULTI_GPU
    for(int i=0;i < 4; i++){
      if(commDimPartitioned(i) && (fatGauge.X()[i] < 6)){
	errorQuda("ERROR: partitioned dimension with local size less than 6 is not supported in staggered dslash\n");
      }    
    }
#endif

    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code

    dslashParam.sp_stride = in->Stride();
    dslashParam.parity = parity;
    dslashParam.gauge_stride = fatGauge.Stride();
    dslashParam.long_gauge_stride = longGauge.Stride();
    dslashParam.fat_link_max = fatGauge.LinkMax();
  
    for(int i=0;i<4;i++){
      dslashParam.X[i] = in->X()[i];
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
      dslashParam.ghostOffset[i] = Npad*(in->GhostOffset(i) + in->Stride());
      dslashParam.ghostNormOffset[i] = in->GhostNormOffset(i) + in->Stride();
      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }
    dslashParam.X[0] *= 2;

    void *fatGauge0, *fatGauge1;
    void* longGauge0, *longGauge1;
    bindFatGaugeTex(fatGauge, parity, &fatGauge0, &fatGauge1);
    bindLongGaugeTex(longGauge, parity, &longGauge0, &longGauge1);
    void *longPhase0 = (char*)longGauge0 + longGauge.PhaseOffset();
    void *longPhase1 = (char*)longGauge1 + longGauge.PhaseOffset();   

    if (in->Precision() != fatGauge.Precision() || in->Precision() != longGauge.Precision()){
      errorQuda("Mixing gauge and spinor precision not supported"
		"(precision=%d, fatlinkGauge.precision=%d, longGauge.precision=%d",
		in->Precision(), fatGauge.Precision(), longGauge.Precision());
    }
    
    DslashCuda *dslash = 0;
    size_t regSize = sizeof(float);

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      dslash = new StaggeredDslashCuda<double2, double2, double2, double>
	(out, (double2*)fatGauge0, (double2*)fatGauge1,
	 (double2*)longGauge0, (double2*)longGauge1,
	 (double*)longPhase0, (double*)longPhase1, 
	 longGauge.Reconstruct(), in, x, k, dagger);
      regSize = sizeof(double);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new StaggeredDslashCuda<float2, float2, float4, float>
	(out, (float2*)fatGauge0, (float2*)fatGauge1,
	 (float4*)longGauge0, (float4*)longGauge1, 
	 (float*)longPhase0, (float*)longPhase1,
	 longGauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {	
      dslash = new StaggeredDslashCuda<short2, short2, short4, short>
	(out, (short2*)fatGauge0, (short2*)fatGauge1,
	 (short4*)longGauge0, (short4*)longGauge1, 
	 (short*)longPhase0, (short*)longPhase1,
	 longGauge.Reconstruct(), in, x, k, dagger);
    }

    dslashCuda2(*dslash, regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);

    delete dslash;
    unbindFatGaugeTex(fatGauge);
    unbindLongGaugeTex(longGauge);

    checkCudaError();
  
#else
    errorQuda("Staggered dslash has not been built");
#endif  // GPU_STAGGERED_DIRAC
  }

  template <typename sFloat, typename cFloat>
  class CloverCuda : public Tunable {
  private:
    cudaColorSpinorField *out;
    float *outNorm;
    char *saveOut, *saveOutNorm;
    const cFloat *clover;
    const float *cloverNorm;
    const cudaColorSpinorField *in;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return CLOVER_SHARED_FLOATS_PER_THREAD * reg_size;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

  public:
    CloverCuda(cudaColorSpinorField *out, const cFloat *clover, const float *cloverNorm, 
	       const cudaColorSpinorField *in)
      : out(out), clover(clover), cloverNorm(cloverNorm), in(in)
    {
      bindSpinorTex<sFloat>(in);
    }
    virtual ~CloverCuda() { unbindSpinorTex<sFloat>(in); }
    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      cloverKernel<<<gridDim, tp.block, tp.shared_bytes, stream>>>
	((sFloat*)out->V(), (float*)out->Norm(), clover, cloverNorm, 
	 (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
    }
    virtual TuneKey tuneKey() const
    {
      std::stringstream vol, aux;
      vol << dslashConstants.x[0] << "x";
      vol << dslashConstants.x[1] << "x";
      vol << dslashConstants.x[2] << "x";
      vol << dslashConstants.x[3];
      return TuneKey(vol.str(), typeid(*this).name());
    }

    // Need to save the out field if it aliases the in field
    void preTune() {
      if (in == out) {
	saveOut = new char[out->Bytes()];
	cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
	if (typeid(sFloat) == typeid(short4)) {
	  saveOutNorm = new char[out->NormBytes()];
	  cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
	}
      }
    }

    // Restore if the in and out fields alias
    void postTune() {
      if (in == out) {
	cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
	delete[] saveOut;
	if (typeid(sFloat) == typeid(short4)) {
	  cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
	  delete[] saveOutNorm;
	}
      }
    }

    std::string paramString(const TuneParam &param) const // Don't bother printing the grid dim.
    {
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 504ll * dslashConstants.VolumeCB(); }
  };


  void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
		  const cudaColorSpinorField *in, const int parity) {

    dslashParam.parity = parity;
    dslashParam.threads = in->Volume();

#ifdef GPU_CLOVER_DIRAC
    Tunable *clov = 0;
    void *cloverP, *cloverNormP;
    QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

    if (in->Precision() != clover_prec)
      errorQuda("Mixing clover and spinor precision not supported");

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      clov = new CloverCuda<double2, double2>(out, (double2*)cloverP, (float*)cloverNormP, in);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      clov = new CloverCuda<float4, float4>(out, (float4*)cloverP, (float*)cloverNormP, in);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      clov = new CloverCuda<short4, short4>(out, (short4*)cloverP, (float*)cloverNormP, in);
    }
    clov->apply(0);

    unbindCloverTex(clover);
    checkCudaError();

    delete clov;
#else
    errorQuda("Clover dslash has not been built");
#endif
  }


  template <typename sFloat>
  class TwistGamma5Cuda : public Tunable {

  private:
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    double a;
    double b;
    double c;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

    char *saveOut, *saveOutNorm;

  public:
    TwistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		    double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type twist) :
      out(out), in(in) 
    {
      bindSpinorTex<sFloat>(in);
      if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS))
        setTwistParam(a, b, kappa, mu, dagger, twist);
      else{//twist doublet
        a = kappa, b = mu, c = epsilon;
      } 
    }
    virtual ~TwistGamma5Cuda() {
      unbindSpinorTex<sFloat>(in);    
    }

   TuneKey tuneKey() const {
     std::stringstream vol, aux;
     vol << dslashConstants.x[0] << "x";
     vol << dslashConstants.x[1] << "x";
     vol << dslashConstants.x[2] << "x";
     vol << dslashConstants.x[3];    
     aux << "TwistFlavor" << in->TwistFlavor();
     return TuneKey(vol.str(), typeid(*this).name(), aux.str());
   }  

  void apply(const cudaStream_t &stream) 
  {
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) {
      twistGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
	((sFloat*)out->V(), (float*)out->Norm(), a, b, 
	 (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
    } else {
      twistGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>>
	((sFloat*)out->V(), (float*)out->Norm(), a, b, c, 
	 (sFloat*)in->V(), (float*)in->Norm(), dslashParam);
    }
#endif
  }

  void preTune() {
    saveOut = new char[out->Bytes()];
    cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
    if (typeid(sFloat) == typeid(short4)) {
      saveOutNorm = new char[out->NormBytes()];
      cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
    }
  }

  void postTune() {
    cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
    delete[] saveOut;
    if (typeid(sFloat) == typeid(short4)) {
      cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
      delete[] saveOutNorm;
    }
  }

  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  long long flops() const { return 24ll * dslashConstants.VolumeCB(); }
  long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
 };

//!ndeg tm: 
  void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		     const int dagger, const double &kappa, const double &mu, const double &epsilon,   const QudaTwistGamma5Type twist)
  {
    if(in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS)
      dslashParam.threads = in->Volume();
    else //twist doublet    
      dslashParam.threads = in->Volume() / 2;

#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
    Tunable *twistGamma5 = 0;

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      twistGamma5 = new TwistGamma5Cuda<double2>(out, in, kappa, mu, epsilon, dagger, twist);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      twistGamma5 = new TwistGamma5Cuda<float4>(out, in, kappa, mu, epsilon, dagger, twist);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      twistGamma5 = new TwistGamma5Cuda<short4>(out, in, kappa, mu, epsilon, dagger, twist);
    }

    twistGamma5->apply(streams[Nstream-1]);
    checkCudaError();

    delete twistGamma5;
#else
    errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

#include "dslash_core/tmc_gamma_core.h"

  template <typename cFloat, typename sFloat>
  class TwistCloverGamma5Cuda : public Tunable {

  private:
    const cFloat *clover;
    const float *cNorm;
    const cFloat *cloverInv;
    const float *cNrm2;
    QudaTwistGamma5Type twist;
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;
    double a;
    double b;
    double c;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return dslashConstants.VolumeCB(); }

    char *saveOut, *saveOutNorm;

  public:
    TwistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		    double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type tw,
		    cFloat *clov, const float *cN, cFloat *clovInv, const float *cN2) :
      out(out), in(in)
    {
      bindSpinorTex<sFloat>(in);
      twist = tw;
      clover = clov;
      cNorm = cN;
      cloverInv = clovInv;
      cNrm2 = cN2;

      if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS))
        setTwistParam(a, b, kappa, mu, dagger, tw);
//	  a = 2.*kappa*mu;
      else{//twist doublet
        errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      } 
    }
    virtual ~TwistCloverGamma5Cuda() {
      unbindSpinorTex<sFloat>(in);    
    }

   TuneKey tuneKey() const {
     std::stringstream vol, aux;
     vol << dslashConstants.x[0] << "x";
     vol << dslashConstants.x[1] << "x";
     vol << dslashConstants.x[2] << "x";
     vol << dslashConstants.x[3];    
     aux << "TwistFlavor" << in->TwistFlavor();
     return TuneKey(vol.str(), typeid(*this).name(), aux.str());
   }  

  void apply(const cudaStream_t &stream)
  {
//A.S.: should this be GPU_TWISTED_CLOVER_DIRAC instead?
#if (defined GPU_TWISTED_CLOVER_DIRAC)
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
    if((in->TwistFlavor() == QUDA_TWIST_PLUS) || (in->TwistFlavor() == QUDA_TWIST_MINUS)) {	//Idea for the kernel, two spinor inputs (IN and clover applied IN), on output (Clover applied IN + ig5IN)
      if (twist == QUDA_TWIST_GAMMA5_DIRECT)
        twistCloverGamma5Kernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
	  ((sFloat*)out->V(), (float*)out->Norm(), a, 
	   (sFloat*)in->V(), (float*)in->Norm(), dslashParam,
	   clover, cNorm, cloverInv, cNrm2);
      else if (twist == QUDA_TWIST_GAMMA5_INVERSE)
        twistCloverGamma5InvKernel<<<gridDim, tp.block, tp.shared_bytes, stream>>> 
	  ((sFloat*)out->V(), (float*)out->Norm(), a, 
	   (sFloat*)in->V(), (float*)in->Norm(), dslashParam,
	   clover, cNorm, cloverInv, cNrm2);
    } else {
        errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
    }
#endif
  }

  void preTune() {
    saveOut = new char[out->Bytes()];
    cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
    if (typeid(sFloat) == typeid(short4)) {
      saveOutNorm = new char[out->NormBytes()];
      cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
    }
  }

  void postTune() {
    cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
    delete[] saveOut;
    if (typeid(sFloat) == typeid(short4)) {
      cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
      delete[] saveOutNorm;
    }
  }

 std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }

  long long flops() const { return 24ll * dslashConstants.VolumeCB(); }	//TODO FIX THIS NUMBER!!!
  long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
 };

  void twistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in, const int dagger, const double &kappa, const double &mu,
			     const double &epsilon, const QudaTwistGamma5Type twist, const FullClover *clov, const FullClover *clovInv, const int parity)
  {
    if(in->TwistFlavor() == QUDA_TWIST_PLUS || in->TwistFlavor() == QUDA_TWIST_MINUS)
      dslashParam.threads = in->Volume();
    else //twist doublet    
      errorQuda("Twisted doublet not supported in twisted clover dslash");

#ifdef GPU_TWISTED_CLOVER_DIRAC
    Tunable *tmClovGamma5 = 0;

    void *clover, *cNorm, *cloverInv, *cNorm2;
    QudaPrecision clover_prec = bindTwistedCloverTex(*clov, *clovInv, parity, &clover, &cNorm, &cloverInv, &cNorm2);

    if (in->Precision() != clover_prec)
      errorQuda("ERROR: Clover precision and spinor precision do not match\n");

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
#if (__COMPUTE_CAPABILITY__ >= 130)
      tmClovGamma5 = new TwistCloverGamma5Cuda<double2,double2>(out, in, kappa, mu, epsilon, dagger, twist, (double2 *) clover, (float *) cNorm, (double2 *) cloverInv, (float *) cNorm2);
#else
      errorQuda("Double precision not supported on this GPU");
#endif
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      tmClovGamma5 = new TwistCloverGamma5Cuda<float4,float4>(out, in, kappa, mu, epsilon, dagger, twist, (float4 *) clover, (float *) cNorm, (float4 *) cloverInv, (float *) cNorm2);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      tmClovGamma5 = new TwistCloverGamma5Cuda<short4,short4>(out, in, kappa, mu, epsilon, dagger, twist, (short4 *) clover, (float *) cNorm, (short4 *) cloverInv, (float *) cNorm2);
    }

    tmClovGamma5->apply(streams[Nstream-1]);
    checkCudaError();

    delete tmClovGamma5;
    unbindTwistedCloverTex(*clov);
#else
    errorQuda("Twisted clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

} // namespace quda

#include "misc_helpers.cu"


#if defined(GPU_FATLINK) || defined(GPU_GAUGE_FORCE) || defined(GPU_FERMION_FORCE) // || defined(GPU_UNITARIZE)
#include <force_common.h>
#endif

#ifdef GPU_FATLINK
#include "llfat_quda.cu"
#endif

#ifdef GPU_GAUGE_FORCE
#include "gauge_force_quda.cu"
#endif

#ifdef GPU_FERMION_FORCE
#include "fermion_force_quda.cu"
#endif



