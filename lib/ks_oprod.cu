#include <quda_internal.h>
#include <color_spinor_oprod.h>
#include <quda_matrix.h>

// In this file, we write the code to compute the outer product of two staggered quark fields.
// The result is a 3x3 color-matrix field.
namespace quda {

  template<typename Complex, typename Output, typename Input>
    struct StaggeredOprodArg {
      typedef typename RealTypeId<Complex>::Type real;
      Output out;
      real coeff;
      Input inA;
      Input inB;
      unsigned int dir;
      unsigned int parity;

      StaggeredOprodArg(const Output &out, 
          const real&  coeff, const Input& inA, const InputB& inB,
          const unsigned int parity, const unsigned int dir)
        : out(out), coeff(coeff), inA(inA), inB(inB), dir(dir), parity(parity) {}
    };



  template<typename Complex, typename Output, typename Input>
    __device__ __host__ void staggeredOprodCompute(StaggeredOprodArg& arg,  unsigned int idx)
    {

      typedef typename RealTypeId<Complex>::Type real;
      Matrix<Complex,3> result; 
      Complex x[3], y[3];

      arg.inA.load(x, idx);
      arg.inB.load(y, idx);

      outerProd(x,y, &result);
      arg.out.save((real*)(result.data), idx, arg.dir, arg.parity);  
    }


  // Host function
  template<typename Complex, typename Output, typename Input>
    void staggeredOprodField(StaggeredOprodArg<Complex,Output,Input> arg){
      for(unsigned int idx=0; idx<arg.out.volumeCB; ++idx){
        staggeredOprodCompute<Complex,Input>(arg, idx); 
      }
    }


  template<typename Complex, typename Output, typename Input>
    void staggeredOprodFieldKernel(StaggeredOprodArg<Complex,Output,Input> arg){
      unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
      const unsigned int gridSize = blockDim.x*gridDim.x;
      while(idx < arg.out.volumeCB){
        staggeredOprodCompute<Complex,Output,Input>(arg,idx);
        idx += gridSize;
      }
    }

  template<typename Complex, typename Output, typename Input> 
    class StaggeredOprodField : public Tunable {

      private: 
        StaggeredOprodArg<Complex,Output,Input> arg;
        const int *X; // pointer to lattice dimensions
        const QudaFieldLocation location; // location of the lattice fields

        unsigned int sharedBytesPerThread() const { return 0; }
        unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

        // NEED TO CHECK/CHANGE THIS!!!  
        unsigned int minThreads() const { return arg.out.volumeCB; }
        bool tuneGridDim() const { return false; }


      public: 
        StaggeredOprodField(const StaggeredOproArg<Complex,OutPut,Input> &arg,
            const int *X, QudaFieldLocation location) 
          : arg(arg), X(X), location(location) {}

        virtual ~StaggeredOprodField() {}


        void apply(const cudaStream_t &stream){
          if(location == QUDA_CUDA_FIELD_LOCATION){
            TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
            staggeredOprodFieldKernel<Complex,Output,Input><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
          }else{ // run the CPU code
            staggeredOprodField<Complex,Output,Input>(arg);
          }
        } // apply


        void preTune(){}
        void postTune(){}

        long long flops() const { return 0; } // FIX THIS!

        long long bytes() const { return 0; } // FIX THIS!

        TuneKey tunekey() const {
          std::stringstream vol, aux;
          vol << X[0] << "x";
          vol << X[1] << "x";
          vol << X[2] << "x";
          vol << X[3] << "x";
          aux << "threads=" << arg.out.volumeCB << ",prec=" << sizeof(Complex)/2;
          aux << "stride="  << arg.out.stride;
          return TuneKey(vol.str(), typeid(*this).name(), aux.str());
        }
    }; // StaggeredOprodField


  template<typename Float, typename Output, typename Input>
    void computeStaggeredOprod(Output& out, const Input& inA, const Input& inB,
        const int *X, 
        const double coeff, 
        const unsigned int dir, 
        const unsigned int parity, 
        QudaFieldLocation location)
    {
      StaggeredOprodArg<Complex, Output, Input> arg(out, coeff, inA, inB, dir, parity);
      StaggeredOprodField<Complex, Output, Input> staggeredOprod(arg, X, location);
      staggeredOprod.apply(0);
      if(location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
    }




  template<typename Float>
    void computeStaggeredOprod(cudaGaugeField& out, const cudaColorSpinorField &inA, const cudaColorSpinorField &in, 
        const double coeff, const unsigned int dir, const unsigned int parity){

      if(inA.SiteSubset() == QUDA_FULL_SITE_SUBSET){
        computeStaggeredOprod<Float>(out, inA.Even(), inB.Even(), dir, 0);
        computeStaggeredOprod<Float>(out, inA.Odd(), inB.Odd(), dir, 1);
      }else{
        typedef typename ComplexTypeId<Float>::Type Complex;
        Spinor<Complex,Complex,Complex,3,0,0> srcA(inA);
        Spinor<Complex,Complex,Complex,3,0,0> srcB(inB);

        computeStaggeredOprod(FloatNOrder<Float,18,2,18>(out), srcA, srcB, inA.X(), coeff, dir, parity, QUDA_CUDA_FIELD_LOCATION);
      }
    }


  void computeStaggeredOprod(GaugeField& out, const ColorSpinorField &inA, const ColorSpinorField &inB,
      const double coeff, const unsigned int dir, const unsigned int parity){

    if(out.Location() == QUDA_CPU_FIELD_LOCATION){
      errorQuda("Outer product is not supported on the CPU\n");
    }

    if(out.Precision() == QUDA_DOUBLE_PRECISION){
      computeStaggeredOprod<double>(out, inA, inB, coeff, dir, parity);
    }else if(out.Precision() == QUDA_SINGLE_PRECISION){
      computeStaggeredOprod<float>(out, inA, inB, coeff, dir, parity);
    }else{
      errorQuda("Precision %d not supported", out.Precision());
    }

  }



} // namespace quda
