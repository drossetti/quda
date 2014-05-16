#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#ifdef QMP_COMMS
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>
#include <gsl/gsl_rng.h>

#define MAX(a,b) ((a)>(b)?(a):(b))


//#define	TESTPOINT
//#define	RANDOM_CONF
//#define	CROSSCHECK

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <cufft.h>

//#include <randCuda.h>

extern bool			tune;
extern int			device;
extern QudaDslashType		dslash_type;
extern int			xdim;
extern int			ydim;
extern int			zdim;
extern int			tdim;
extern int			Lsdim;
extern int			numberHP;
extern int			nConf;
extern int			numberLP;
extern int			MaxP;
extern int			gridsize_from_cmdline[];
extern QudaReconstructType	link_recon;
extern QudaPrecision		prec;
extern QudaReconstructType	link_recon_sloppy;
extern QudaPrecision		prec_sloppy;
extern QudaInverterType		inv_type;
extern int			multishift;			// whether to test multi-shift or standard solver

extern char			latfile[];

extern void			usage(char**);

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

void	genRandomSource	(void *spinorIn, QudaInvertParam *inv_param, gsl_rng *rNum)
{
#ifdef	TESTPOINT
	if	(inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
			((float*) spinorIn)[18]		 = 1.;		//t-Component
	}
	else if	(inv_param->cpu_prec == QUDA_DOUBLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
			((double*) spinorIn)[18]	 = 1.;
	}
#else
	if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION) 
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((float*) spinorIn)[i*2]	= 1.;
				break;

				case 1:
	
				((float*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((float*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((float*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
	else
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((double*) spinorIn)[i*2]	= 1.;
				break;
	
				case 1:
	
				((double*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((double*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((double*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
#endif
}

void	dumpData	(int nSol, const char *Pref, void **cnRes_vv, void **cnRs2_vv, const int iDiv, QudaPrecision prec)
//void	dumpData	(int nSol, const char *Pref, void **cnRes_vv, void *cnRs2_vv, const int iDiv, QudaPrecision prec)
{
	FILE		*sfp;

	char		file_name[256];

	printfQuda	("\n\nNsol es %d\n\n", nSol);

	if	(prec == QUDA_DOUBLE_PRECISION)
	{
		sprintf(file_name, "Tune.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int it=0; it<iDiv; it++)	//Era 200
		{
			for	(int wt=0; wt<tdim; wt++)
			{
			 	int	rT	 = wt+comm_coord(3)*tdim;

				double	meanR	 = ((double2*)cnRes_vv[it])[wt].x/((double) nSol);
				double	meanI	 = ((double2*)cnRes_vv[it])[wt].y/((double) nSol);
				double	varR	 = ((double2*)cnRs2_vv[it])[wt].x/((double) nSol);
				double	varI	 = ((double2*)cnRs2_vv[it])[wt].y/((double) nSol);
				double	errR	 = sqrt((varR - meanR*meanR)/((double) nSol));
				double	errI	 = sqrt((varI - meanI*meanI)/((double) nSol));
	 			fprintf (sfp, "%03d %03d %02d %+.6le %+.6le %+.6le %+.6le %.6le %.6le\n", it, nSol, rT, meanR, meanI, errR, errI, varR, varI);
			}
		}

		fflush  (sfp);
/*		fclose(sfp);
		sprintf(file_name, "%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);
		int Vol = xdim*ydim*zdim;
for     (int wt=0; wt<tdim; wt++)
{
                int     rT       = wt+comm_coord(3)*tdim;

                for     (int gm=0; gm<16; gm++)
                {                                                                               // TEST
                        fprintf (sfp, "%03d %02d %02d %+.10le %+.10le\n", nSol, rT, gm,
                                ((double2*)cnRs2_vv)[Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2*)cnRs2_vv)[Vol*wt+Vol*tdim*gm].y/((double) nSol));

                        fflush  (sfp);
                }
}
*/
		fflush  (sfp);
	}
	else if	(prec == QUDA_SINGLE_PRECISION)
	{
		sprintf(file_name, "Tune.lpSg.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

		if	((sfp = fopen(file_name, "wb")) == NULL)
			printf("\nCannot open file %s\n", file_name),	exit(-1);

		for	(int it=0; it<iDiv; it++)
		{
			for	(int wt=0; wt<tdim; wt++)
			{
			 	int	rT	 = wt+comm_coord(3)*tdim;

				float	meanR	 = ((float2**)cnRes_vv)[it][wt].x/((float) nSol);
				float	meanI	 = ((float2**)cnRes_vv)[it][wt].y/((float) nSol);
				float	varR	 = ((float2**)cnRs2_vv)[it][wt].x/((float) nSol) - meanR*meanR;
				float	varI	 = ((float2**)cnRs2_vv)[it][wt].y/((float) nSol) - meanI*meanI;
	 			fprintf (sfp, "%03d %02d %+.6e %+.6e %+.6e %+.6e\n", nSol, rT, meanR, meanI, varR, varI);
			}
		}

		fflush  (sfp);
	}

	fclose(sfp);

	return;
}

int	main	(int argc, char **argv)
{
	int	i, k;
	double	precisionHP	 = 4e-10;

	char	name[16];

//	int	maxSources, flag;

//	curandStateMRG32k3a	*rngStat;

	for	(i =1;i < argc; i++)
	{
		if	(process_command_line_option(argc, argv, &i) == 0)
			continue;
    
		printf	("ERROR: Invalid option:%s\n", argv[i]);
		usage	(argv);
	}

	// initialize QMP or MPI
#if defined(QMP_COMMS)
	QMP_thread_level_t tl;
	QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
	MPI_Init(&argc, &argv);
#endif

	initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

	//	Initialize random number generator

	int	myRank	 = comm_rank();
	int	seed;

	if	(myRank == 0)
		seed	 = (int) clock();

	MPI_Bcast	(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

//	setUpRandomCuda	(state, seed, myRank, 256, 64);


	//	Starts Quda initialization

	if	(prec_sloppy == QUDA_INVALID_PRECISION)
		prec_sloppy		 = prec;

	if	(link_recon_sloppy == QUDA_RECONSTRUCT_INVALID)
		link_recon_sloppy	 = link_recon;


  // *** QUDA parameters begin here.


//	dslash_type				 = QUDA_TWISTED_MASS_DSLASH;
	dslash_type				 = QUDA_TWISTED_CLOVER_DSLASH;

	QudaPrecision cpu_prec			 = QUDA_DOUBLE_PRECISION;
	QudaPrecision cuda_prec			 = prec;
	QudaPrecision cuda_prec_sloppy		 = prec_sloppy;

	QudaGaugeParam gauge_param		 = newQudaGaugeParam();
	QudaInvertParam inv_param		 = newQudaInvertParam();

	gauge_param.X[0]			 = xdim;
	gauge_param.X[1]			 = ydim;
	gauge_param.X[2]			 = zdim;
	gauge_param.X[3]			 = tdim;

	gauge_param.anisotropy			 = 1.0;
	gauge_param.type			 = QUDA_WILSON_LINKS;
	gauge_param.gauge_order			 = QUDA_QDP_GAUGE_ORDER;
	gauge_param.t_boundary			 = QUDA_ANTI_PERIODIC_T;

	gauge_param.cpu_prec			 = cpu_prec;
	gauge_param.cuda_prec			 = cuda_prec;
	gauge_param.reconstruct			 = link_recon;
	gauge_param.cuda_prec_sloppy		 = cuda_prec_sloppy;
	gauge_param.reconstruct_sloppy		 = link_recon_sloppy;
	gauge_param.cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	gauge_param.reconstruct_precondition	 = link_recon_sloppy;
	gauge_param.gauge_fix			 = QUDA_GAUGE_FIXED_NO;

	inv_param.dslash_type = dslash_type;

	double mass = -2.;
	inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

	if	(dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
	{
		inv_param.mu = 0.003;
		inv_param.twist_flavor = QUDA_TWIST_MINUS;
	}

	inv_param.solution_type		 = QUDA_MAT_SOLUTION;
	inv_param.solve_type		 = QUDA_NORMOP_PC_SOLVE;
//	inv_param.solve_type		 = QUDA_DIRECT_PC_SOLVE;
	if	(inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH)
		inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
	else
		inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN;

	inv_param.dagger		 = QUDA_DAG_NO;
//	inv_param.mass_normalization	 = QUDA_MASS_NORMALIZATION;
	inv_param.mass_normalization	 = QUDA_KAPPA_NORMALIZATION;
	inv_param.solver_normalization	 = QUDA_DEFAULT_NORMALIZATION;

//	inv_param.inv_type		 = QUDA_BICGSTAB_INVERTER;
	inv_param.inv_type		 = QUDA_CG_INVERTER;

	inv_param.gcrNkrylov		 = 30;
	inv_param.tol			 = precisionHP;
	inv_param.maxiter		 = 40000;
	inv_param.reliable_delta	 = 1e-2; // ignored by multi-shift solver

#if __COMPUTE_CAPABILITY__ >= 200
	// require both L2 relative and heavy quark residual to determine convergence
//	inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL);
	inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
	inv_param.tol_hq = precisionHP;	// specify a tolerance for the residual for heavy quark residual
#else
	// Pre Fermi architecture only supports L2 relative residual norm
	inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
#endif

	// domain decomposition preconditioner parameters

	inv_param.inv_type_precondition	 = QUDA_INVALID_INVERTER;
	inv_param.schwarz_type		 = QUDA_ADDITIVE_SCHWARZ;
	inv_param.precondition_cycle	 = 1;
	inv_param.tol_precondition	 = 1e-1;
	inv_param.maxiter_precondition	 = 10;
	inv_param.verbosity_precondition = QUDA_SILENT;
	inv_param.omega			 = 1.0;


	inv_param.cpu_prec		 = cpu_prec;
	inv_param.cuda_prec		 = cuda_prec;
	inv_param.cuda_prec_sloppy	 = cuda_prec_sloppy;
	inv_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
	inv_param.preserve_source	 = QUDA_PRESERVE_SOURCE_NO;
	inv_param.gamma_basis		 = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
//	inv_param.gamma_basis		 = QUDA_UKQCD_GAMMA_BASIS;
	inv_param.dirac_order		 = QUDA_DIRAC_ORDER;

	inv_param.tune			 = QUDA_TUNE_YES;
//	inv_param.tune			 = QUDA_TUNE_NO;
//	inv_param.preserve_dirac	 = QUDA_PRESERVE_DIRAC_NO;

	inv_param.input_location	 = QUDA_CPU_FIELD_LOCATION;
	inv_param.output_location	 = QUDA_CPU_FIELD_LOCATION;

	gauge_param.ga_pad		 = 0; // 24*24*24/2;
	inv_param.sp_pad		 = 0; // 24*24*24/2;
	inv_param.cl_pad		 = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
	int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
	int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
	int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
	int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
	int pad_size =MAX(x_face_size, y_face_size);
	pad_size = MAX(pad_size, z_face_size);
	pad_size = MAX(pad_size, t_face_size);
	gauge_param.ga_pad = pad_size; 
//  inv_param.cl_pad = pad_size; 
//  inv_param.sp_pad = pad_size; 
#endif

	if	(dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
	{
		inv_param.clover_cpu_prec		 = cpu_prec;
		inv_param.clover_cuda_prec		 = cuda_prec;
		inv_param.clover_cuda_prec_sloppy	 = cuda_prec_sloppy;
		inv_param.clover_cuda_prec_precondition	 = QUDA_HALF_PRECISION;
		inv_param.clover_order			 = QUDA_PACKED_CLOVER_ORDER;
	}

	inv_param.verbosity = QUDA_VERBOSE;

	//set the T dimension partitioning flag
	//commDimPartitionedSet(3);

	// *** Everything between here and the call to initQuda() is
	// *** application-specific.

	// set parameters for the reference Dslash, and prepare fields to be loaded
	setDims			(gauge_param.X);

	setSpinorSiteSize	(24);

	size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
	size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

	void *gauge[4];

	for	(int dir = 0; dir < 4; dir++)
		if	((gauge[dir]	 = malloc(V*gaugeSiteSize*gSize)) == NULL)
		{
			printf	("Fatal Error; Couldn't allocate memory in host for gauge fields. Asked for %ld bytes.", V*gaugeSiteSize*gSize);
			exit	(1);
		}

//	totalMem	+= ((double) (V*gaugeSiteSize*gSize*4))/(1024.*1024.*1024.);

	if	(strcmp(latfile,""))			// load in the command line supplied gauge field
	{
		if	(read_custom_binary_gauge_field((double**)gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline))
		{
			printf	("Fatal Error; Couldn't read gauge conf %s\n", latfile);
			exit	(1);
		}
	}
	else
	{						// else generate a random SU(3) field
		construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
		inv_param.kappa	 = 0.12;
	}

	comm_barrier	();

	const int	Vol	 = xdim*ydim*zdim;

	// initialize the QUDA library
	initQuda(device);

	const int	nSteps	 = 400;

	void	**cnRes_vv;
	void	**cnRs2_vv;
	void	**cnCor_vv;
	void	**cnCr2_vv;
	void	***cnRes_gv;
	void	***cnRs2_gv;
	void	***cnCor_gv;
	void	***cnCr2_gv;

	cnRes_gv	 = (void***) malloc(sizeof(double2**)*3);
	cnRs2_gv	 = (void***) malloc(sizeof(double2**)*3);
	cnCor_gv	 = (void***) malloc(sizeof(double2**)*3);
	cnCr2_gv	 = (void***) malloc(sizeof(double2**)*3);

	if	(cnRes_gv == NULL) printf("Error allocating memory cnRes_gv\n"), exit(1);
	if	(cnRs2_gv == NULL) printf("Error allocating memory cnRs2_gv\n"), exit(1);
	if	(cnCor_gv == NULL) printf("Error allocating memory cnCor_gv\n"), exit(1);
	if	(cnCr2_gv == NULL) printf("Error allocating memory cnCr2_gv\n"), exit(1);

	for	(int dt = 0; dt < 3; dt++)
	{
		cnRes_gv[dt]	 = (void**) malloc(sizeof(double2*)*nSteps);
		cnRs2_gv[dt]	 = (void**) malloc(sizeof(double2*)*nSteps);
		cnCor_gv[dt]	 = (void**) malloc(sizeof(double2*)*nSteps);
		cnCr2_gv[dt]	 = (void**) malloc(sizeof(double2*)*nSteps);

		if	(cnRes_gv[dt] == NULL) printf("Error allocating memory cnRes_gv[%d]\n", dt), exit(1);
		if	(cnRs2_gv[dt] == NULL) printf("Error allocating memory cnRs2_gv[%d]\n", dt), exit(1);
		if	(cnCor_gv[dt] == NULL) printf("Error allocating memory cnCor_gv[%d]\n", dt), exit(1);
		if	(cnCr2_gv[dt] == NULL) printf("Error allocating memory cnCr2_gv[%d]\n", dt), exit(1);

		for	(int it = 0; it < nSteps; it++)
		{
			if	((cudaHostAlloc(&(cnRes_gv[dt][it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
				printf("Error allocating memory cnRes_gv[%d][%d]\n", dt, it), exit(1);

			cudaMemset	(cnRes_gv[dt][it], 0, sizeof(double2)*tdim);

			if	((cudaHostAlloc(&(cnRs2_gv[dt][it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
				printf("Error allocating memory cnRs2_gv[%d][%d]\n", dt, it), exit(1);

			cudaMemset	(cnRs2_gv[dt][it], 0, sizeof(double2)*tdim);

			if	((cudaHostAlloc(&(cnCor_gv[dt][it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
				printf("Error allocating memory cnCor_gv[%d][%d]\n", dt, it), exit(1);

			cudaMemset	(cnCor_gv[dt][it], 0, sizeof(double2)*tdim);

			if	((cudaHostAlloc(&(cnCr2_gv[dt][it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
				printf("Error allocating memory cnCr2_gv[%d][%d]\n", dt, it), exit(1);

			cudaMemset	(cnCr2_gv[dt][it], 0, sizeof(double2)*tdim);

			cudaDeviceSynchronize();
		}
	}

	cnRes_vv	 = (void**) malloc(sizeof(double2*)*nSteps);
	cnRs2_vv	 = (void**) malloc(sizeof(double2*)*nSteps);
	cnCor_vv	 = (void**) malloc(sizeof(double2*)*nSteps);
	cnCr2_vv	 = (void**) malloc(sizeof(double2*)*nSteps);

	if	(cnRes_vv == NULL) printf("Error allocating memory cnRes_vv_HP\n"), exit(1);
	if	(cnRs2_vv == NULL) printf("Error allocating memory cnRs2_vv_HP\n"), exit(1);
	if	(cnCor_vv == NULL) printf("Error allocating memory cnCor_vv_HP\n"), exit(1);
	if	(cnCr2_vv == NULL) printf("Error allocating memory cnCr2_vv_HP\n"), exit(1);

	for	(int it = 0; it < nSteps; it++)
	{
		if	((cudaHostAlloc(&(cnRes_vv[it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
			printf("Error allocating memory cnRes_vv[%d]\n", it), exit(1);

		cudaMemset	(cnRes_vv[it], 0, sizeof(double2)*tdim);

		if	((cudaHostAlloc(&(cnRs2_vv[it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
			printf("Error allocating memory cnRes_vv[%d]\n", it), exit(1);

		cudaMemset	(cnRs2_vv[it], 0, sizeof(double2)*tdim);

		if	((cudaHostAlloc(&(cnCor_vv[it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
			printf("Error allocating memory cnRes_vv[%d]\n", it), exit(1);

		cudaMemset	(cnCor_vv[it], 0, sizeof(double2)*tdim);

		if	((cudaHostAlloc(&(cnCr2_vv[it]), sizeof(double2)*tdim, cudaHostAllocMapped)) != cudaSuccess)
			printf("Error allocating memory cnRes_vv[%d]\n", it), exit(1);

		cudaMemset	(cnCr2_vv[it], 0, sizeof(double2)*tdim);

		cudaDeviceSynchronize();
	}

	//	load the gauge field
	loadGaugeQuda	((void*)gauge, &gauge_param);

	inv_param.clover_coeff = 1.57551;
	inv_param.clover_coeff *= inv_param.kappa;

	if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(NULL, NULL, &inv_param);

	void	*spinorIn	 = malloc(V*spinorSiteSize*sSize);
	void	*spinorOut	 = malloc(V*spinorSiteSize*sSize);

	gsl_rng	*rNum	 = gsl_rng_alloc(gsl_rng_ranlux);
	gsl_rng_set	(rNum, (int) clock());

	printfQuda	("Allocated memory for random number generator\n");

	// start the timer
	double time0 = -((double)clock());

	// perform the inversion

	printfQuda	("Starting inversions\n");

	inv_param.tol	 = precisionHP;
	inv_param.tol_hq = precisionHP;
	inv_param.maxiter = 25;

	for	(i=0; i<numberHP; i++)
	{
		genRandomSource	(spinorIn, &inv_param, rNum);
		tuneOneEndTrick	(spinorOut, spinorIn, &inv_param, cnRes_gv, cnRs2_gv, cnRes_vv, cnRs2_vv, nSteps, true, cnCor_gv, cnCr2_gv, cnCor_vv, cnCr2_vv);
	}

	for	(i=0; i<(numberLP - numberHP); i++)
	{
		genRandomSource	(spinorIn, &inv_param, rNum);
		tuneOneEndTrick	(spinorOut, spinorIn, &inv_param, cnRes_gv, cnRs2_gv, cnRes_vv, cnRs2_vv, nSteps, false, NULL, NULL, NULL, NULL);
	}

	sprintf		(name, "Sigma.%03d.S%03d.%02d", numberLP, nSteps, 5);
	dumpData	(numberLP, name, cnRes_vv, cnRs2_vv, nSteps, cuda_prec);
	sprintf		(name, "SigCr.%03d.S%03d.%02d", numberHP, nSteps, 5);
	dumpData	(numberHP, name, cnCor_vv, cnCr2_vv, nSteps, cuda_prec);
	sprintf		(name, "gALp.X.%03d.S%03d.%02d", numberLP, nSteps, 5);
	dumpData	(numberLP, name, cnRes_gv[0], cnRs2_gv[0], nSteps, cuda_prec);
	sprintf		(name, "gALp.Y.%03d.S%03d.%02d", numberLP, nSteps, 5);
	dumpData	(numberLP, name, cnRes_gv[1], cnRs2_gv[1], nSteps, cuda_prec);
	sprintf		(name, "gALp.Z.%03d.S%03d.%02d", numberLP, nSteps, 5);
	dumpData	(numberLP, name, cnRes_gv[2], cnRs2_gv[2], nSteps, cuda_prec);
	sprintf		(name, "gACr.X.%03d.S%03d.%02d", numberHP, nSteps, 5);
	dumpData	(numberHP, name, cnCor_gv[0], cnCr2_gv[0], nSteps, cuda_prec);
	sprintf		(name, "gACr.Y.%03d.S%03d.%02d", numberHP, nSteps, 5);
	dumpData	(numberHP, name, cnCor_gv[1], cnCr2_gv[1], nSteps, cuda_prec);
	sprintf		(name, "gACr.Z.%03d.S%03d.%02d", numberHP, nSteps, 5);
	dumpData	(numberHP, name, cnCor_gv[2], cnCr2_gv[2], nSteps, cuda_prec);

  // stop the timer
	double timeIO	 = -((double)clock());
	time0		+= clock();
	time0		/= CLOCKS_PER_SEC;
    
	printfQuda	("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", inv_param.spinorGiB, gauge_param.gaugeGiB);
	printfQuda	("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

	gsl_rng_free(rNum);

	for	(int it=0; it<200; it++)
	{
		cudaFreeHost	(cnRs2_vv[it]);
		cudaFreeHost	(cnRes_vv[it]);
	}

	free(cnRs2_vv);
	free(cnRes_vv);

	freeGaugeQuda	();

  // finalize the QUDA library
	endQuda		();

  // end if the communications layer
	MPI_Finalize	();


	free	(spinorIn);
	free	(spinorOut);

	timeIO		+= clock();
	timeIO		/= CLOCKS_PER_SEC;

	printf		("%g seconds spent on IO\n", timeIO);
	fflush		(stdout);

	return	0;
}

