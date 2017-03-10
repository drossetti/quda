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


#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>

#include "nvToolsExt.h"
static const uint32_t nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int nvtx_num_colors = sizeof(nvtx_colors)/sizeof(uint32_t);

#define MY_PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%nvtx_num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = nvtx_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.category = cid;\
    nvtxRangePushEx(&eventAttrib); \
}
#define MY_POP_RANGE nvtxRangePop();

namespace quda {

  namespace wilson {

#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>

  // Enable shared memory dslash for Fermi architecture
  //#define SHARED_WILSON_DSLASH
  //#define SHARED_8_BYTE_WORD_SIZE // 8-byte shared memory access

#ifdef GPU_WILSON_DIRAC
#define DD_CLOVER 0
#include <wilson_dslash_def.h>    // Wilson Dslash kernels (including clover)
#undef DD_CLOVER
#endif

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#include <dslash_quda.cuh>

  } // end namespace wilson

  // declare the dslash events
#include <dslash_events.cuh>
#include <typeinfo>

  using namespace wilson;

#ifdef GPU_WILSON_DIRAC
  template <typename sFloat, typename gFloat>
  class WilsonDslashCuda : public SharedDslashCuda {

  private:
    const gFloat *gauge0, *gauge1;
    const double a;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      if (dslashParam.kernel_type == INTERIOR_KERNEL) { // Interior kernels use shared memory for common iunput
	int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
	return DSLASH_SHARED_FLOATS_PER_THREAD * reg_size;
      } else { // Exterior kernels use no shared memory
	return 0;
      }
    }

  public:
    WilsonDslashCuda(cudaColorSpinorField *out, const gFloat *gauge0, const gFloat *gauge1, 
		     const QudaReconstructType reconstruct, const cudaColorSpinorField *in,
		     const cudaColorSpinorField *x, const double a, const int dagger)
      : SharedDslashCuda(out, in, x, reconstruct, dagger), gauge0(gauge0), gauge1(gauge1), a(a)
    { 
      bindSpinorTex<sFloat>(in, out, x); 
    }

    virtual ~WilsonDslashCuda() { unbindSpinorTex<sFloat>(in, out, x); }

    void apply(const cudaStream_t &stream)
    {
#ifdef SHARED_WILSON_DSLASH
      if (dslashParam.kernel_type == EXTERIOR_KERNEL_X) 
	errorQuda("Shared dslash does not yet support X-dimension partitioning");
#endif
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dslashParam.block[0] = tp.aux.x; dslashParam.block[1] = tp.aux.y; dslashParam.block[2] = tp.aux.z; dslashParam.block[3] = tp.aux.w;
      for (int i=0; i<4; i++) dslashParam.grid[i] = ( (i==0 ? 2 : 1) * in->X(i)) / dslashParam.block[i];
      DSLASH(dslash, tp.grid, tp.block, tp.shared_bytes, stream, dslashParam, (sFloat*)out->V(), (float*)out->Norm(), gauge0, gauge1, (sFloat*)in->V(), (float*)in->Norm(), (sFloat*)(x ? x->V() : 0), (float*)(x ? x->Norm() : 0), a)
    }

  };
#endif // GPU_WILSON_DIRAC

#include <dslash_policy.cuh>

  // Wilson wrappers
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, 
			const int parity, const int dagger, const cudaColorSpinorField *x, const double &k, 
			const int *commOverride, TimeProfile &profile)
  {
    inSpinor = (cudaColorSpinorField*)in; // EVIL
    inSpinor->allocateGhostBuffer(1);

#ifdef GPU_WILSON_DIRAC
    int Npad = (in->Ncolor()*in->Nspin()*2)/in->FieldOrder(); // SPINOR_HOP in old code
    for(int i=0;i<4;i++){
      dslashParam.ghostDim[i] = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
        
      dslashParam.ghostOffset[i][0] = in->GhostOffset(i,0)/in->FieldOrder();
      dslashParam.ghostOffset[i][1] = in->GhostOffset(i,1)/in->FieldOrder();

      if(in->GhostOffset(i,0)%in->FieldOrder()) errorQuda("ghostOffset(%d,0) %d is not a multiple of FloatN\n", i, in->GhostOffset(i,0));
      if(in->GhostOffset(i,1)%in->FieldOrder()) errorQuda("ghostOffset(%d,1) %d is not a multiple of FloatN\n", i, in->GhostOffset(i,1));

      dslashParam.ghostNormOffset[i][0] = in->GhostNormOffset(i,0);
      dslashParam.ghostNormOffset[i][1] = in->GhostNormOffset(i,1);

      dslashParam.commDim[i] = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
    }

    void *gauge0, *gauge1;
    bindGaugeTex(gauge, parity, &gauge0, &gauge1);

    if (in->Precision() != gauge.Precision())
      errorQuda("Mixing gauge %d and spinor %d precision not supported", 
		gauge.Precision(), in->Precision());

    DslashCuda *dslash = nullptr;
    size_t regSize = in->Precision() == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float);
    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      dslash = new WilsonDslashCuda<double2, double2>(out, (double2*)gauge0, (double2*)gauge1, 
						      gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      dslash = new WilsonDslashCuda<float4, float4>(out, (float4*)gauge0, (float4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      dslash = new WilsonDslashCuda<short4, short4>(out, (short4*)gauge0, (short4*)gauge1,
						    gauge.Reconstruct(), in, x, k, dagger);
    }

#ifndef GPU_COMMS
    DslashPolicyTune dslash_policy(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    dslash_policy.apply(0);
#else
    DslashPolicyImp* dslashImp;
    if (comm_use_async()) {
        comm_enable_async(true);
	if (comm_use_prepared()) {
	  dslashImp = DslashFactory::create(QUDA_GPU_ASYNC_PREPARED_COMMS_DSLASH);
	  //dslashImp = DslashFactory::create(QUDA_GPU_ASYNC_PREPARED_SIMPLE_COMMS_DSLASH);
	} else {
	  dslashImp = DslashFactory::create(QUDA_GPU_ASYNC_COMMS_DSLASH);
	}
    } else {
        dslashImp = DslashFactory::create(QUDA_GPU_COMMS_DSLASH);
    }
    MY_PUSH_RANGE("dslash_async", 2);
    (*dslashImp)(*dslash, const_cast<cudaColorSpinorField*>(in), regSize, parity, dagger, in->Volume(), in->GhostFace(), profile);
    MY_POP_RANGE;
    if (comm_use_async()) {
        //comm_flush();
        PROFILE(comm_progress(), profile, QUDA_PROFILE_COMMS_QUERY);
        comm_enable_async(false);
    }
    delete dslashImp;
#endif

    delete dslash;
    unbindGaugeTex(gauge);

    checkCudaError();
#else
    errorQuda("Wilson dslash has not been built");
#endif // GPU_WILSON_DIRAC

  }

}
