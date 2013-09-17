#include <cstdio>
#include <cstdlib>
#include <staggered_oprod.h>

#include <tune_quda.h>
#include <quda_internal.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>

namespace quda {

#include <texture.h>

  static bool kernelPackT = false;

  template<int N>
    void createEventArray(cudaEvent_t (&event)[N], unsigned int flags=cudaEventDefault)
    {
      for(int i=0; i<N; ++i)
        cudaEventCreate(&event[i],flags);
      return;
    }

  template<int N>
    void destroyEventArray(cudaEvent_t (&event)[N])
    {
      for(int i=0; i<N; ++i)
        cudaEventDestroy(event[i]);
    }


  static cudaEvent_t packEnd;
  static cudaEvent_t gatherStart[4];
  static cudaEvent_t gatherEnd[4];
  static cudaEvent_t scatterStart[4];
  static cudaEvent_t scatterEnd[4];
  static cudaEvent_t oprodStart;
  static cudaEvent_t oprodEnd;


  void createStaggeredOprodEvents(){
#ifdef MULTI_GPU
    cudaEventCreate(packEnd, cudaEventDisableTiming);
    createEventArray(gatherStart, cudaEventDisableTiming);
    createEventArray(gatherEnd, cudaEventDisableTiming);
    createEventArray(scatterStart, cudaEventDisableTiming);
    createEventArray(scatterEnd, cudaEventDisableTiming);
#endif
    cudaEventCreate(&oprodStart, cudaEventDisableTiming);
    cudaEventCreate(&oprodEnd, cudaEventDisableTiming);
    return;
  }

  void destroyStaggeredOprodEvents(){
#ifdef MULTI_GPU
    destroyEventArray(gatherStart);
    destroyEventArray(gatherEnd);
    destroyEventArray(scatterStart);
    destroyEventArray(scatterEnd);
    cudaEventDestroy(packEnd);
    cudaEventDestroy(oprodStart);
#endif
    cudaEventDestroy(oprodEnd);
    return;
  }


  enum KernelType {OPROD_INTERIOR_KERNEL, OPROD_EXTERIOR_KERNEL};

  template<typename Complex, typename Output, typename Input>
    struct StaggeredOprodArg {
      unsigned int length;
      unsigned int X[4];
      unsigned int parity;
      unsigned int dir;
      unsigned int ghostOffset;
      unsigned int displacement;
      KernelType kernelType;
      bool partitioned[4];
      Input in;
      Output out;

      StaggeredOprodArg(const unsigned int length,
          const unsigned int X[4],
          const unsigned int parity,
          const unsigned int dir,
          const unsigned int ghostOffset,
          const unsigned int displacement,   
          const KernelType& kernelType, 
          const Input& in,
          const Output& out) : length(length), parity(parity), ghostOffset(ghostOffset), displacement(displacement), kernelType(kernelType),
      in(in), out(out) 
      {
        for(int i=0; i<4; ++i) this->X[i] = X[i];
        for(int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
      }
    };



  template<typename Complex, typename Output, typename Input>
    __global__ void interiorOprodKernel(StaggeredOprodArg<Complex, Output, Input> arg)
    {
      unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      Complex x[3];
      Complex y[3];
      Matrix<Complex,3> result;


      while(idx<arg.length){
        int a_loaded = 0;
        for(int dir=0; dir<4; ++dir){
          int shift[4] = {0,0,0,0};
          shift[dir] = arg.shift;
          /*
             const int nbr_idx = neighborIndex(idx, shift, arg.partitioned, arg.parity, arg.X);
             if(nbr_idx>=0){
             arg.inB.load(y, nbr_idx);
             if(!a_loaded){ 
             arg.inA.load(x, idx);
             a_loaded=1;
             }
             outerProd(y,x,&result);
             result *= arg.coeff;
             arg.out.save(static_caste<real*>(result.data), idx, dir, arg.parity); 
             } // nbr_idx >= 0
           */
        } // dir
        idx += gridSize;
      }
      return;
    } // interiorOprodKernel



  template<typename Complex, typename Output, typename Input> 
    __global__ void exteriorOprodKernel(StaggeredOprodArg<Complex, Output, Input> arg)
    {
      unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = gridDim.x*blockDim.x;

      int shift[4] = {0,0,0,0};
      shift[arg.dir] = arg.shift;

      Complex a[3];
      Complex b[3];
      Matrix<Complex,3> result;
      typedef typename RealTypeId<Complex>::Type real;

      unsigned int x[4];
      /*
         while(cb_idx<arg.length){
         coordsFromIndex(x, idx, arg.X, arg.dir, arg.parity); 
      // determines coords from idx such that ghost-zone accesses are coalesced
      // Note, in general, loads from the bulk and stores are not guaranteed to be coalesced.
      // However, I don't believe this is a problem in practice if we don't partition the X direction.
      const unsigned int bulk_cb_idx = (((x[3]*arg.X[2] + x[2])*arg.X[1] + x[1])*arg.X[0] >> 1);
      arg.inA.load(a, bulk_cb_idx);

      const unsigned int ghost_idx = arg.ghostOffset + ghostIndexFromCoords<3,3>(x, arg.X, arg.dir, arg.shift);
      arg.inB.load(b, ghost_idx);

      outerProd(b,a,&result);
      result *= arg.coeff; // multiply the result by coeff

      arg.out.save(static_caste<real*>(result.data), bulk_cb_idx, arg.dir, arg.parity); 

      cb_idx += gridSize;
      }
       */
      return;
    }


  /*
     template<typename Complex, typename Output, typename Input>
     struct StaggeredOprodArg {
     unsigned int length;
     unsigned int X[4];
     unsigned int parity;
     unsigned int dir;
     unsigned int ghostOffset;
     unsigned int displacement;
     KernelType kernelType;
     bool partitioned[4];
     Input in;
     Output out;

     StaggeredOprodArg(const unsigned int length,
     const unsigned int X[4],
     const unsigned int parity,
     const unsigned int dir,
     const unsigned int ghostOffset,
     const unsigned int displacement,   
     const KernelType& kernelType, 
     const Input& in,
     const Output& out) : length(length), parity(parity), ghostOffset(ghostOffset), displacement(displacement), kernelType(kernelType),
     in(in), out(out) 
     {
     for(int i=0; i<4; ++i) this->X[i] = X[i];
     for(int i=0; i<4; ++i) this->partitioned[i] = commDimPartitioned(i) ? true : false;
     }
     };
   */



  template<typename Complex, typename Output, typename Input> 
    class StaggeredOprodField : public Tunable {

      private:
        StaggeredOprodArg<Complex,Output,Input> arg;
        QudaFieldLocation location; // location of the lattice fields

        unsigned int sharedBytesPerThread() const { return 0; }
        unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

        unsigned int minThreads() const { return arg.out.volumeCB; }
        bool tunedGridDim() const { return false; }

      public:
        StaggeredOprodField(const StaggeredOprodArg<Complex,Output,Input> &arg,
            QudaFieldLocation location)
          : arg(arg), location(location) {} 

        virtual ~StaggeredOprodField() {}

        void set(const StaggeredOprodArg<Complex,Output,Input> &arg, QudaFieldLocation location){
          this->arg = arg;
          this->location = location;
        } // set

        void apply(const cudaStream_t &stream){
          if(location == QUDA_CUDA_FIELD_LOCATION){
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            if(arg.kernelType == OPROD_INTERIOR_KERNEL){
              interiorOprodKernel(arg);
            }else if(arg.kernelType == OPROD_EXTERIOR_KERNEL){
              exteriorOprodKernel(arg);
            }else{
              errorQuda("Kernel type not supported\n");
            }
          }else{ // run the CPU code
            errorQuda("No CPU support for staggered outer-product calculation\n");
          }
        } // apply

        void preTune(){}
        void postTune(){}

        long long flops() const {
          return 0; // fix this
        }

        long long bytes() const { 
          return 0; // fix this
        }

        TuneKey tuneKey() const {
          std::stringstream vol, aux;
          vol << arg.X[0] << "x";
          vol << arg.X[1] << "x";
          vol << arg.X[2] << "x";
          vol << arg.X[3] << "x";

          aux << "threads=" << arg.length << ",prec=" << sizeof(Complex)/2;
          aux << "stride=" << arg.in.stride;
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    }; // StaggeredOprodField

  template<typename Complex, typename Output, typename Input>
    void computeStaggeredOprodCuda(Output out, Input& in, cudaColorSpinorField& src, FaceBuffer& faceBuffer,  const unsigned int parity, const int faceVolumeCB[4], const unsigned int ghostOffset[4], const unsigned int displacement)
    {
      assert(displacement == 1 || displacement == 3);
 

      cudaEventRecord(oprodStart, streams[Nstream-1]);

      const unsigned int dim[4] = {src.X(0)*2, src.X(1), src.X(2), src.X(3)};
      // Create the arguments for the interior kernel 
      StaggeredOprodArg<Complex,Output,Input> arg(out.volumeCB, dim, parity, 0, displacement, OPROD_INTERIOR_KERNEL, in, out);
      StaggeredOprodField<Complex,Output,Input> oprod(arg, QUDA_CUDA_FIELD_LOCATION);

      bool pack=false;
      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i) && (i!=3 || kernelPackT)){
          pack = true;
          break;
        }
      } // i=3,..,0

      // source, dir(+/-1), parity, dagger, stream_ptr
      faceBuffer.pack(in, -1, 1-parity, 0, streams); // packing is all done in streams[Nstream-1]

      if(pack){
        cudaEventRecord(packEnd, streams[Nstream-1]);
      }

      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i)){

          cudaEvent_t &event = (i!=3 || getKernelPackT()) ? packEnd : oprodStart;
          cudaStreamWaitEvent(streams[2*i], event, 0); // wait in stream 2*i for event to complete

          // Initialize the host transfer from the source spinor
          faceBuffer.gather(src, false, 2*i); 
          // record the end of the gathering 
          cudaEventRecord(gatherEnd[i], streams[2*i]);
        } // comDim(i)
      } // i=3,..,0

      // Should probably be able to reset the arguments
      oprod.apply(streams[Nstream-1]); // Need to change this so that there's a way to distinguish between interior and exterior kernels


      // compute gather completed 
      int gatherCompleted[5];
      int commsCompleted[5];
      int oprodCompleted[4];

      for(int i=0; i<4; ++i){
        gatherCompleted[i] = commsCompleted[i] = oprodCompleted[i] = 0;
      }
      gatherCompleted[4] = commsCompleted[4] = 1;

      // initialize commDimTotal 
      int commDimTotal = 0;
      for(int i=0; i<4; ++i){
        commDimTotal += commDimPartitioned(i);
      }
      commDimTotal *= 2;

      // initialize previousDir
      int previousDir[4];
      for(int i=3; i>=0; i--){
        if(commDimPartitioned(i)){
          int prev = 4;
          for(int j=3; j>i; j--){
            if(commDimPartitioned(j)){
              prev = j;
            }
            previousDir[i] = prev;
          }
        }
      } // set previous directions


      if(commDimTotal){
        arg.kernelType = OPROD_EXTERIOR_KERNEL;
        unsigned int completeSum=0;
        while(completeSum < commDimTotal){
          for(int i=3; i>=0; i--){
            if(!commDimPartitioned(i)) continue;

            if(!gatherCompleted[i] && gatherCompleted[previousDir[i]]){
              cudaError_t event_test = cudaEventQuery(gatherEnd[i]);

              if(event_test == cudaSuccess){
                gatherCompleted[i] = 1;
                completeSum++;
                faceBuffer.commsStart(2*i);
              }
            }

            // Query if comms has finished 
            if(!commsCompleted[i] && commsCompleted[previousDir[i]] && gatherCompleted[i]){
              int comms_test = faceBuffer.commsQuery(2*i);
              if(comms_test){
                commsCompleted[i] = 1;
                completeSum++;
                faceBuffer.scatter(src, false, 2*i);
              }
            }

            // enqueue the boundary oprod kernel as soon as the scatters have been enqueud
            if(!oprodCompleted[i] && commsCompleted[i]){
              cudaEventRecord(scatterEnd[i], streams[2*i]);
              cudaStreamWaitEvent(streams[Nstream-1], scatterEnd[i],0);

              // set arguments for the exterior kernel
              arg.dir = i;
              arg.length = displacement*faceVolumeCB[i];
              arg.ghostOffset = ghostOffset[i];
              oprod.set(arg,QUDA_CUDA_FIELD_LOCATION);

              // apply kernel in border region
              oprod.apply(streams[Nstream-1]);

              oprodCompleted[i] = 1;
            }
          } // i=3,..,0 
        } // completeSum < commDimTotal
      } // if commDimTotal
    } // computeStaggeredOprodCuda


  // At the moment, I pass an instance of FaceBuffer in. 
  // Soon, faceBuffer will be subsumed in cudaColorSpinorField.

  void computeStaggeredOprod(cudaGaugeField& out, cudaColorSpinorField& in,  
      FaceBuffer& faceBuffer,
      const unsigned int parity, const unsigned int displacement)
  {

    if(out.Order() != QUDA_FLOAT2_GAUGE_ORDER)
      errorQuda("Unsupported output ordering: %d\n", out.Order());    

    const unsigned int ghostOffset[4] = {0,0,0,0};
#ifdef MULTI_GPU
    const unsigned int Npad = in.Ncolor()*in.Nspin()*2/in.FieldOrder();
    for(int dir=0; dir<4; ++dir){
      ghostOffset[dir] = Npad*(in.GhostOffset(dir) + in.Stride()); 
    }
#endif

    if(in.Precision() != out.Precision()) errorQuda("Mixed precision not supported: %d %d\n", in.Precision(), out.Precision());
    
    if(in.Precision() == QUDA_DOUBLE_PRECISION){
      Spinor<double2, double2, double2, 3, 0, 0> spinor(in);
      computeStaggeredOprodCuda<double2>(FloatNOrder<double, 18, 2, 18>(out), spinor, in, faceBuffer, parity, in.GhostFace(), ghostOffset, displacement);
    }else if(in.Precision() == QUDA_SINGLE_PRECISION){
      Spinor<float2, float2, float2, 3, 0, 0> spinor(in);
      computeStaggeredOprodCuda<float2>(FloatNOrder<float, 18, 2, 18>(out), spinor, in, faceBuffer, parity, in.GhostFace(), ghostOffset, displacement);
    }else{
      errorQuda("Unsupported precision: %d\n", in.Precision());
    }
    return;
  } // computeStaggeredOprod



} // namespace quda
