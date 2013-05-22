//#include	<mpi.h>

namespace quda
{
	#include	"covDev.h"	//Covariant derivative definitions

	#ifdef MULTI_GPU
		#include	<mpi.h>
	#endif

	template<typename Float, typename Float2> 
	void	covDevQuda	(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, const int *commOverride)
	{
		bindSpinorTex<Float2>(in); //for multi-gpu usage

		dslashParam.parity	 = parity;
		dslashParam.kernel_type	 = INTERIOR_KERNEL;
		dslashParam.threads	 = in->Volume();

		#ifdef MULTI_GPU
	        	//Aux parameters for memory operations:
			const int	tghostVolume	 = in->X(0)*in->X(1)*in->X(2);		// / 2tslice size without padding and for parity spinor!

			const int	Nvec		 = (sizeof(Float2)/sizeof(Float));	// for any kind of Float2
			const int	Nint		 = in->Ncolor()*in->Nspin()*Nvec;	// degrees of freedom	
			const int	Npad		 = Nint / Nvec;				// number Nvec buffers we have
			const int	Nt_minus1_offset = (in->Volume() - tghostVolume);	// Vh-Vsh	

			//MPI parameters:
			const int	my_rank		 = comm_rank();

			//Neighbour  rank in t direction:
			Topology	*Topo		 = comm_default_topology();

			const int Cfwd[QUDA_MAX_DIM]	 = {0, 0, 0, +1};
			const int Cbck[QUDA_MAX_DIM]	 = {0, 0, 0, -1};

			const int	fwd_neigh_t_rank = comm_rank_displaced	(Topo, Cfwd);
			const int	bwd_neigh_t_rank = comm_rank_displaced	(Topo, Cbck);
		#endif

		void *gauge0, *gauge1;
		bindGaugeTex	(gauge, parity, &gauge0, &gauge1);

		if (in->Precision() != gauge.Precision())
			errorQuda	("Mixing gauge and spinor precision not supported");

		#if (__COMPUTE_CAPABILITY__ < 130)
			if	(in->Precision() == QUDA_DOUBLE_PRECISION)
				errorQuda	("Double precision not supported by hardware");
		#endif
		
		{
			dim3 gridBlock(64, 1, 1);
			dim3 gridDim( (dslashParam.threads+gridBlock.x-1) / gridBlock.x, 1, 1);

			switch	(mu)
			{
				case	0:
				covDevM012Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(), dslashParam);
				break;

				case	1:
				covDevM112Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(),  dslashParam);
				break;

				case	2:
				covDevM212Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(),  dslashParam);
				break;

				case	3:
				covDevM312Kernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(), dslashParam);
				break;

				case	4:
				covDevM012DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(), dslashParam);
				break;

				case	5:
				covDevM112DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(),  dslashParam);
				break;

				case	6:
				covDevM212DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(),  dslashParam);
				break;

				case	7:
				covDevM312DaggerKernel<INTERIOR_KERNEL><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) in->V(), dslashParam);
				break;
			}
		}
//	else
//		errorQuda("Single or half precision not supported");    
//#else
//	errorQuda("Double precision not supported on this GPU");
//#endif

		#ifdef MULTI_GPU	
			int		send_t_rank, recv_t_rank;
			int	rel, nDimComms = 4;


			if	(comm_size() > 1)
			{
				if	(mu == 3)
				{
					send_t_rank	 = bwd_neigh_t_rank;
					recv_t_rank	 = fwd_neigh_t_rank;
					rel		 = +1;  
				}	  
				else if	(mu == 7)
				{
					send_t_rank	 = fwd_neigh_t_rank;
					recv_t_rank	 = bwd_neigh_t_rank;	  
					rel		 = -1;  
				}
				else
				{
					unbindGaugeTex		(gauge);	  
					return;
				}
			}
			else
			{
				unbindGaugeTex		(gauge);	  
				return;
			}

			if	(cudaDeviceSynchronize() != cudaSuccess)
			{
				printf	("Something went wrong in rank %d", my_rank);
				exit	(-1);
			}

			//send buffers in t-dir:
			void	*send_t	 = 0;
			if	(cudaHostAlloc(&send_t, Nint * tghostVolume * sizeof(Float), 0) != cudaSuccess)
			{
				printf	("Error in rank %d: Unable to allocate %d bytes for MPI requests (send_t)\n", my_rank, Nint*tghostVolume*sizeof(Float));
				exit	(-1);
			}

			//recv buffers in t-dir:
			void	*recv_t	 = 0;
			if	(cudaHostAlloc(&recv_t, Nint * tghostVolume * sizeof(Float), 0) != cudaSuccess)
			{
				printf	("Error in rank %d: Unable to allocate %d bytes for MPI requests (recv_t)\n", my_rank, Nint*tghostVolume*sizeof(Float));
				exit	(-1);
			}

			//ghost buffer on gpu:
			void	*ghostTBuffer;
			cudaMalloc(&ghostTBuffer, Nint * tghostVolume * sizeof(Float));	

			//Collect t-slice faces on the host (befor MPI send):
			int	tface_offset	 = Nt_minus1_offset;

			//Recv buffers from neighbor:
			void	*sendTFacePtr	 = mu == 3 ? (char*)in->V() : mu == 7 ? (char*)in->V() + tface_offset*Nvec*sizeof(Float) : NULL;	//Front face -> 3, back face -> 7

			size_t	len		 = tghostVolume*Nvec*sizeof(Float);
			size_t	spitch		 = in->Stride()*Nvec*sizeof(Float);

			cudaMemcpy2DAsync(send_t, len, sendTFacePtr, spitch, len, Npad, cudaMemcpyDeviceToHost, streams[0]);
			cudaStreamSynchronize(streams[0]);

			for	(int i=0;i<4;i++)
			{
				dslashParam.ghostDim[i]		 = commDimPartitioned(i); // determines whether to use regular or ghost indexing at boundary
				dslashParam.ghostOffset[i]	 = 0;
				dslashParam.ghostNormOffset[i]	 = 0;
				dslashParam.commDim[i]		 = (!commOverride[i]) ? 0 : commDimPartitioned(i); // switch off comms if override = 0
			}	
	
			//Send buffers to neighbors:

			MsgHandle	*mh_send[4];
			MsgHandle	*mh_from[4];

			int		nbytes[4];

			nbytes[0]	 = 0;
			nbytes[1]	 = 0;
			nbytes[2]	 = 0;
			nbytes[3]	 = Nint * tghostVolume * sizeof(Float);

			for	(int i=3; i<nDimComms; i++)
			{
		/*		int dx[4] = {0};
				dx[3] = -1;
				dx[mu] = +1;
		*/
				mh_send[i]	= comm_declare_send_relative	(send_t, i, rel,      nbytes[i]);
		/*
				int dx[4] = {0};
				dx[nu] = -1;
				dx[mu] = +1;
		*/
				mh_from[i]	= comm_declare_receive_relative	(recv_t, i, rel*(-1), nbytes[i]);
			}

			for	(int i=3; i<nDimComms; i++)
			{
				comm_start	(mh_from[i]);
				comm_start	(mh_send[i]);
			}
	
			for	(int i=3; i<nDimComms; i++)
			{
				comm_wait	(mh_send[i]);
				comm_wait	(mh_from[i]);
			}
	
			for	(int i=3; i<nDimComms; i++)
			{
				comm_free	(mh_send[i]);
				comm_free	(mh_from[i]);
			}

			//Send buffers to GPU:
			cudaMemcpy(ghostTBuffer, recv_t, Nint * tghostVolume * sizeof(Float), cudaMemcpyHostToDevice);
	
			//start execution
			//define exec domain
			dslashParam.kernel_type	 = EXTERIOR_KERNEL_T;
			dslashParam.threads	 = tghostVolume;
	
			cudaBindTexture		(0, spinorTexDouble, (Float2*)ghostTBuffer, Nint*tghostVolume*sizeof(Float));

			dim3	gridBlock(64, 1, 1);
			dim3	gridDim((dslashParam.threads+gridBlock.x-1) / gridBlock.x, 1, 1);	

			switch	(mu)
			{
				case	3:
				covDevM312Kernel<EXTERIOR_KERNEL_T><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) ghostTBuffer, dslashParam);
				break;
				case	7:
				covDevM312DaggerKernel<EXTERIOR_KERNEL_T><<<gridDim, gridBlock, 0, streams[Nstream-1]>>>((Float2*) out->V(), (const Float2*) gauge0, (const Float2*) gauge1, (const Float2*) ghostTBuffer, dslashParam);
				break;
			}

			cudaFree(ghostTBuffer);
			cudaFreeHost(send_t);
			cudaFreeHost(recv_t);
		#endif

		cudaUnbindTexture	(spinorTexDouble);
		unbindGaugeTex		(gauge);

		cudaDeviceSynchronize	();
		checkCudaError		();
	}

	
	void	covDev		(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in, const int parity, const int mu, const int *commOverride)
	{
		dim3 gridDim((in->Volume() - 1) / blockDim.x + 1, 1, 1);	//CHANGE FOR MULTI_GPU

	//	checkSpinor(x,y);

		if	(in->Precision	() == QUDA_HALF_PRECISION)
			errorQuda	("Error: Half precision not supported");

		if	(in->Precision	() == QUDA_SINGLE_PRECISION)
			covDevQuda<float,float4>	(out, gauge, in, parity, mu, commOverride);
		else if	(in->Precision	() == QUDA_DOUBLE_PRECISION)
			#if (__COMPUTE_CAPABILITY__ >= 130)
				covDevQuda<double,double2>	(out, gauge, in, parity, mu, commOverride);
			#else
				errorQuda	("Error: Double precision not supported by hardware");
			#endif
	}

}
