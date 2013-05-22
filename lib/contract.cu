//
// double2 contractCuda(float2 *x, float2 *y, float2 *result) {}
//

namespace quda
{
	#include <gamma5.h>		// g5 kernel

	void	gamma5Cuda	(cudaColorSpinorField *out, const cudaColorSpinorField *in, const dim3 &block)
	{
		dslashParam.threads	 = in->Volume();

		if	(in->Precision() == QUDA_DOUBLE_PRECISION)
		{

		#if (__COMPUTE_CAPABILITY__ >= 130)
			dim3 blockDim (block.x, block.y, block.z);
			dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

			bindSpinorTex<double2>(in); //for multi-gpu usage
			gamma5Kernel<<<gridDim, blockDim, 0>>> ((double2*)out->V(), (float*)out->Norm(), dslashParam, in->Stride());
		#else
			errorQuda("Double precision not supported on this GPU");
		#endif
		}
		else if	(in->Precision() == QUDA_SINGLE_PRECISION)
		{
			dim3 blockDim (block.x, block.y, block.z);
			dim3 gridDim( (dslashParam.threads+blockDim.x-1) / blockDim.x, 1, 1);

			bindSpinorTex<float4>(in); //for multi-gpu usage
			gamma5Kernel<<<gridDim, blockDim, 0>>> ((float4*)out->V(), (float*)out->Norm(), dslashParam, in->Stride());
		}

		cudaDeviceSynchronize	();
		checkCudaError		();
	}


	#include "contract_core.h"
	#include "contract_core_plus.h"
	#include "contract_core_minus.h"

	#ifndef	_TWIST_QUDA_CONTRACT
	#error	"Contraction core undefined"
	#endif

	#ifndef	_TWIST_QUDA_CONTRACT_PLUS
	#error	"Contraction core (plus) undefined"
	#endif

	#ifndef	_TWIST_QUDA_CONTRACT_MINUS
	#error	"Contraction core (minus) undefined"
	#endif

	#define checkSpinor(a, b)									\
	{												\
		if	(a.Precision() != b.Precision())						\
			errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision());	\
		if	(a.Length() != b.Length())							\
			errorQuda("lengths do not match: %d %d", a.Length(), b.Length());		\
		if	(a.Stride() != b.Stride())							\
			errorQuda("strides do not match: %d %d", a.Stride(), b.Stride());		\
	}

	void	contractGamma5Cuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity)
	{
		dim3 gridDim((x.Volume() - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			bindSpinorTex<double2>(&x, &y);

			switch	(sign)
			{
				default:
				case	0:
				contractGamma5Kernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractGamma5PlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractGamma5MinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}
		}
		else if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			bindSpinorTex<float4>(&x, &y);

			switch	(sign)
			{
				default:
				case	0:
				contractGamma5Kernel     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractGamma5PlusKernel <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractGamma5MinusKernel<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}
		}

		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}

	void	contractTsliceCuda	(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Tslice, const int Parity)
	{
	        const int tVolume = x.X(0)*x.X(1)*x.X(2);//!half volume (check it!), i.e., x.Volume() / x.X(3);
		dim3 gridDim((tVolume - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(double);

				cudaBindTexture(0, spinorTexDouble, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexDouble,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractTsliceKernelD     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
		
				case	1:
				contractTslicePlusKernelD <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;

				case	2:
				contractTsliceMinusKernelD<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexDouble);
				cudaUnbindTexture(interTexDouble);
			#endif
		}

		if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(float);

				cudaBindTexture(0, spinorTexSingle, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexSingle,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractTsliceKernelS     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
		
				case	1:
				contractTslicePlusKernelS <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;

				case	2:
				contractTsliceMinusKernelS<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), tVolume, x.Stride(), XS[0], XS[1], XS[2], Tslice, Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexSingle);
				cudaUnbindTexture(interTexSingle);
			#endif
		}
		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}

	void	contractCuda		(const cudaColorSpinorField &x, const cudaColorSpinorField &y, void *result, const int sign, const dim3 &blockDim, int *XS, const int Parity)
	{
		dim3 gridDim((x.Volume() - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

		checkSpinor(x,y);

		if	(x.Precision() == QUDA_HALF_PRECISION)
			errorQuda("Error: Half precision not supported");

		if	(x.Precision() == QUDA_DOUBLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(double);

				cudaBindTexture(0, spinorTexDouble, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexDouble,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractKernelD     <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractPlusKernelD <<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractMinusKernelD<<<gridDim, blockDim, blockDim.x*32*sizeof(double)>>>((double2*)result, (double2*)x.V(), (double2*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexDouble);
				cudaUnbindTexture(interTexDouble);
			#endif
		}

		if	(x.Precision() == QUDA_SINGLE_PRECISION)
		{
			#ifndef USE_TEXTURE_OBJECTS
				int	spinor_bytes	= x.Length()*sizeof(float);

				cudaBindTexture(0, spinorTexSingle, x.V(), spinor_bytes);
				cudaBindTexture(0, interTexSingle,  y.V(), spinor_bytes);
			#else
				dslashParam.inTex	 = (&x)->Tex();
				dslashParam.inTexNorm	 = (&x)->TexNorm();
				dslashParam.xTex	 = (&y)->Tex();
				dslashParam.xTexNorm	 = (&y)->TexNorm();
			#endif

			switch	(sign)
			{
				default:
				case	0:
				contractKernelS     <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
	
				case	1:
				contractPlusKernelS <<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;

				case	2:
				contractMinusKernelS<<<gridDim, blockDim, blockDim.x*32*sizeof(float)>>>((float2*)result, (float4*)x.V(), (float4*)y.V(), x.Volume(), x.Stride(), XS[0], XS[1], XS[2], Parity, dslashParam);
				break;
			}

			#ifndef USE_TEXTURE_OBJECTS
				cudaUnbindTexture(spinorTexSingle);
				cudaUnbindTexture(interTexSingle);
			#endif
		}

		cudaDeviceSynchronize	();
		checkCudaError		();

		return;
	}

}

