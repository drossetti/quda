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
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

//#define TESTTMLQCD
//#define DISKSPINOR

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
extern QudaInverterType  inv_type;
extern int multishift; // whether to test multi-shift or standard solver

extern char latfile[];

extern void usage(char** );

extern double newMu;

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


/*	For Testing	*/

template<class Float>
void	reOrder	(Float *array1, Float *array2, const int arrayOffset)
{
	if	(array1 != array2)
	{
		for	(int i = 0; i<V*arrayOffset; i++)
			array2[i]	= 0.;
	}

	for	(int i = 0; i<V*arrayOffset; i++)
	{
		int	pointT		=	i/arrayOffset;
		int	offset		=	i%arrayOffset;
		int	oddBit		=	0;

		if	(pointT >= V/2)
		{
			pointT	-= V/2;
			oddBit	 = 1;
		}

		int za		 = pointT/(xdim/2);
		int x1h		 = pointT - za*(xdim/2);
		int zb		 = za/ydim;
		int x2		 = za - zb*ydim;
		int x4		 = zb/zdim;
		int x3		 = zb - x4*zdim;
		int x1odd	 = (x2 + x3 + x4 + oddBit) & 1;
		int x1		 = 2*x1h + x1odd;
		int X		 = x1 + xdim*(x2 + ydim*(x3 + zdim*x4));
		X		*= arrayOffset;
		X		+= offset;

		if	(array1 != array2)
			array2[X]	= array1[i];
		else
		{
			Float	temp	 = array2[X];
			array2[X]	 = array1[i];
			array1[i]	 = temp;
		}
	}

	return;
}

template<class Float>
void	dumpSpinor	(void *spinorData, const char *end, const double factor)
{
	FILE	*output;
	char	 name[256];
//	void	*spinorData, *spinorOrder;
	void	*spinorOrder;
	int	myRank, totalTdim;

	myRank	= comm_coord(3);//comm_rank();

	sprintf(name, "Spinor.%s.%d", end, myRank);

//	spinorData	= out->V();

	if	((output = (fopen(name, "w+"))) == NULL)
	{
		printfQuda	("Error opening file\n");
		return;
	}

	spinorOrder	= (void*)malloc(V*24*sizeof(Float));

	reOrder<Float>	((Float*)spinorData, (Float*)spinorOrder, 24);

	totalTdim	= comm_dim(3)*tdim;

	for	(int j=0; j<V; j++)
	{
		int	tC	= j/(xdim*ydim*zdim);
		int	zC	= (j - xdim*ydim*zdim*tC)/(xdim*ydim);
		int	yC	= (j - xdim*ydim*(zdim*tC + zC))/xdim;
		int	xC	= (j - xdim*(ydim*(zdim*tC + zC) + yC));

//		double	phase	= ((double) tC)/totalTdim*M_PI;
		double  phase   = (((double) (tC + myRank*tdim))/totalTdim)*M_PI;

		for(int dirac=0; dirac<4; dirac++)
			for(int col=0; col<3; col++)
			{
//				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+((dirac+3)%4)*3+col;
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((Float*)spinorOrder)[2*idx  ]*cos(phase) + ((Float*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((Float*)spinorOrder)[2*idx+1]*cos(phase) - ((Float*)spinorOrder)[2*idx  ]*sin(phase);
/*
				if	(fabs(rPart) < 1e-8)
					((Float*)spinorOrder)[2*idx]	=	0.;

				if	(fabs(iPart) < 1e-8)
					((Float*)spinorOrder)[2*idx+1]	=	0.;
*/
				fprintf	(output, "%02d %02d %02d %02d %d %d %+2.32le %+2.32le\n", xC, yC, zC, tC+tdim*myRank, col, dirac, rPart*factor, iPart*factor);
			}
	}

	fclose(output);

	free(spinorOrder);

	return;
}

/*	End testing	*/

int main(int argc, char **argv)
{

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }

  // initialize QMP or MPI
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

	printfQuda("Start\n");
	fflush(stdout);

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
//  QudaPrecision cpu_prec = prec;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
 
  double kappa5;

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  inv_param.Ls = 1;

//  gauge_param.anisotropy = 2.38;
  gauge_param.anisotropy = 1.;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  double mass = -3.5;//-0.585;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = newMu;
//    inv_param.epsilon = 0.1385;
//    inv_param.twist_flavor = QUDA_TWIST_NONDEG_DOUBLET;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    inv_param.mass = 0.02;
    inv_param.m5 = -1.8;
    kappa5 = 0.5/(5 + inv_param.m5);  
    inv_param.Ls = Lsdim;
  }

  // offsets used only by multi-shift solver
  inv_param.num_offset = 4;
  double offset[4] = {0.01, 0.02, 0.03, 0.04};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  inv_param.inv_type = inv_type;

  if (inv_param.dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = multishift ? QUDA_MATPCDAG_MATPC_SOLUTION : QUDA_MATPC_SOLUTION;
  }
    inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH 
      || multishift || inv_type == QUDA_CG_INVERTER) {
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  } else {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  }

  inv_param.pipeline = 0;

  inv_param.gcrNkrylov = 10;
  inv_param.tol = 4e-10;
#if __COMPUTE_CAPABILITY__ >= 200
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL);
  inv_param.tol_hq = 4e-10; // specify a tolerance for the residual for heavy quark residual
#else
  // Pre Fermi architecture only supports L2 relative residual norm
  inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
#endif
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = 40000;
  inv_param.reliable_delta = 1e-2;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = 
    inv_param.inv_type == QUDA_GCR_INVERTER ? QUDA_MR_INVERTER : QUDA_INVALID_INVERTER;
    
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

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
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.verbosity = QUDA_VERBOSE;

  // declare the dimensions of the communication grid
  initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);


  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv=0, *clover=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
	if	(read_custom_binary_gauge_field((double**)gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline))
	{
		printf	("Fatal Error; Couldn't read gauge conf %s\n", latfile);
		exit	(1);
	}
//    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
//    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    inv_param.kappa = 0.12;
  }
/*
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);

    // The uninverted clover term is only needed when solving the unpreconditioned
    // system or when using "asymmetric" even/odd preconditioning.
    int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			  inv_param.solve_type == QUDA_NORMOP_PC_SOLVE);
    int asymmetric = preconditioned &&
                         (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
                          inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
    if (!preconditioned) {
      clover = clover_inv;
      clover_inv = NULL;
    } else if (asymmetric) { // fake it by using the same random matrix
      clover = clover_inv;   // for both clover and clover_inv
    } else {
      clover = NULL;
    }
  }
*/
  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  void *spinorOut = NULL, **spinorOutMulti = NULL;
  if (multishift) {
    spinorOutMulti = (void**)malloc(inv_param.num_offset*sizeof(void *));
    for (int i=0; i<inv_param.num_offset; i++) {
      spinorOutMulti[i] = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
    }
  } else {
    spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  }

  memset(spinorIn, 0, inv_param.Ls*V*spinorSiteSize*sSize);
  memset(spinorCheck, 0, inv_param.Ls*V*spinorSiteSize*sSize);
  if (multishift) {
    for (int i=0; i<inv_param.num_offset; i++) memset(spinorOutMulti[i], 0, inv_param.Ls*V*spinorSiteSize*sSize);    
  } else {
    memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);
  }

#ifdef DISKSPINOR
  printfQuda("Reading Spinor from disk\n");
  fflush (stdout);

  FILE *Caca = fopen("SpinorTm.Out", "r+");

  int		Cx,Cy,Cz,Ct,Cidx,colIdx,diracIdx;
  double	reP, imP;

  int		myRank;
  myRank	= comm_coord(3);//comm_rank();

  do
  {
	fscanf(Caca, "%d %d %d %d %d %d %le %le\n", &Cx, &Cy, &Cz, &Ct, &colIdx, &diracIdx, &reP, &imP);

	if ((Ct >= tdim*(myRank+1)) || (Ct < tdim*myRank))
		continue;

	Ct -= tdim*myRank;

	int	oddbit = (Cx+Cy+Cz+Ct)&1;

	Cidx	= (Cx + Cy*xdim + Cz*xdim*ydim + Ct*xdim*ydim*zdim)/2;

	if	(oddbit)
	{
 		if (inv_param.solution_type == QUDA_MATPC_SOLUTION)
			continue;
		else
			Cidx += V/2;
	}

	double  phase   = (((double) (Ct + myRank*tdim))/(tdim*comm_dim(3)))*M_PI;

	unsigned long indexRe = ((Cidx*4+diracIdx)*3+colIdx)*2;
	unsigned long indexIm = ((Cidx*4+diracIdx)*3+colIdx)*2 + 1;

	switch	(cuda_prec)
	{
		case QUDA_DOUBLE_PRECISION:

		((double*)spinorIn)[indexRe] = cos(phase)*reP - sin(phase)*imP;
		((double*)spinorIn)[indexIm] = cos(phase)*imP + sin(phase)*reP;
		break;

		case QUDA_SINGLE_PRECISION:

		((float*)spinorIn)[indexRe] = (float) (cos(phase)*reP - sin(phase)*imP);
		((float*)spinorIn)[indexIm] = (float) (cos(phase)*imP + sin(phase)*reP);
		break;
	}
  }	while(!feof(Caca));

  fclose(Caca);
#else

  // create a point source at 0 (in each subvolume...  FIXME)
 
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION)
  {
    //((float*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)spinorIn)[i] = rand() / (float)RAND_MAX;
  }
  else
  {
    //((double*)spinorIn)[0] = 1.0;
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = rand() / (double)RAND_MAX;
  }
#endif
//  if  (cuda_prec == QUDA_DOUBLE_PRECISION)
    dumpSpinor<double>	(spinorIn,  "In",  1.);
//  else if (cuda_prec == QUDA_SINGLE_PRECISION)
//    dumpSpinor<float>	(spinorIn,  "In",  1.);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // load the clover term, if desired
  inv_param.kappa = 0.12;
  inv_param.clover_coeff = 1.57551;
//  inv_param.clover_coeff = 0.1;
  inv_param.clover_coeff *= inv_param.kappa;
//  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(NULL, NULL, &inv_param);

  // perform the inversion
  if (multishift) {
    invertMultiShiftQuda(spinorOutMulti, spinorIn, &inv_param);
  } else {
    invertQuda(spinorOut, spinorIn, &inv_param);
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  printfQuda("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) printfQuda("   Clover: %f GiB\n", inv_param.cloverGiB);
  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  if (multishift) {

    void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

    printfQuda("Host residuum checks: \n");
    for(int i=0; i < inv_param.num_offset; i++) {
      ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_MINUS && inv_param.twist_flavor != QUDA_TWIST_PLUS)
	  errorQuda("Twisted mass solution type not supported");
        tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
                 inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                  inv_param.cpu_prec, gauge_param);
      } else {
        printfQuda("Domain wall not supported for multi-shift\n");
        exit(-1);
      }

      axpy(inv_param.offset[i], spinorOutMulti[i], spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      mxpy(spinorIn, spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, Vh*spinorSiteSize, inv_param.cpu_prec);
      double l2r = sqrt(nrm2 / src2);

      printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
		 i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, 
		 inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i]);
    }
    free(spinorTmp);

  } else {
    
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)      
	  tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
	else
	{
          int tm_offset = V*spinorSiteSize; //12*spinorRef->Volume(); 	  
	  void *evenOut = spinorCheck;
	  void *oddOut  = cpu_prec == sizeof(double) ? (void*)((double*)evenOut + tm_offset): (void*)((float*)evenOut + tm_offset);
    
	  void *evenIn  = spinorOut;
	  void *oddIn   = cpu_prec == sizeof(double) ? (void*)((double*)evenIn + tm_offset): (void*)((float*)evenIn + tm_offset);
    
	  tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0, inv_param.cpu_prec, gauge_param);	
	}
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
        dw_mat(spinorCheck, gauge, spinorOut, kappa5, inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
          ax(0.5/kappa5, spinorCheck, V*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else {
          ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
        }
      }

    } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_MINUS && inv_param.twist_flavor != QUDA_TWIST_PLUS)
	  errorQuda("Twisted mass solution type not supported");
        tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, 
                  inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
	dw_matpc(spinorCheck, gauge, spinorOut, kappa5, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param, inv_param.mass);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
        if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
          ax(0.25/(kappa5*kappa5), spinorCheck, Vh*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
        } else {
          ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, Vh*spinorSiteSize, inv_param.cpu_prec);
      
	}
      }

    } else if (inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {

      void *spinorTmp = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

      ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	if (inv_param.twist_flavor != QUDA_TWIST_MINUS && inv_param.twist_flavor != QUDA_TWIST_PLUS)
	  errorQuda("Twisted mass solution type not supported");
        tm_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
        tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
                 inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else if (dslash_type == QUDA_WILSON_DSLASH || dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
        wil_matpc(spinorTmp, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
                  inv_param.cpu_prec, gauge_param);
      } else {
        printfQuda("Unsupported dslash_type\n");
        exit(-1);
      }

      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	errorQuda("Mass normalization not implemented");
      }

      free(spinorTmp);
    }


    int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
    mxpy(spinorIn, spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn, vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
    double l2r = sqrt(nrm2 / src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
	       inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);

//    if	(cuda_prec == QUDA_DOUBLE_PRECISION)
      dumpSpinor<double>(spinorOut, "Out", .5/inv_param.kappa);
//    else if (cuda_prec == QUDA_SINGLE_PRECISION)
//      dumpSpinor<float>	(spinorOut, "Out", .5/inv_param.kappa);
  }

#ifdef	TESTTMLQCD
  printfQuda("Integrity check, comparing with tmLQCD package...\n");

  FILE *Caca1 = fopen("SpinorTm.In", "r+");
  FILE *Caca2 = fopen("Spinor.Out.0", "r+");

  int		C1x,C1y,C1z,C1t,c1olIdx,d1iracIdx;
  int		C2x,C2y,C2z,C2t,c2olIdx,d2iracIdx;
  double	re1P, im1P;
  double	re2P, im2P;

  do
  {
	fscanf(Caca1, "%d %d %d %d %d %d %le %le\n", &C1x, &C1y, &C1z, &C1t, &c1olIdx, &d1iracIdx, &re1P, &im1P);

	if ((C1t >= tdim*(myRank+1)) || (C1t < tdim*myRank))
		continue;

	fscanf(Caca2, "%d %d %d %d %d %d %le %le\n", &C2x, &C2y, &C2z, &C2t, &c2olIdx, &d2iracIdx, &re2P, &im2P);

	if	((C1t != C2t)||(C1x != C2x)||(C1y != C2y)||(C1z != C2z)||(c1olIdx != c2olIdx)||(d1iracIdx != d2iracIdx))
	{
		printfQuda	("Error: Files don't match\n");
		printfQuda	("Offending point:\ntmLQCD: %03d %03d %03d %03d %d %d\tQUDA: %03d %03d %03d %03d %d %d\n",C1x,C1y,C1z,C1t,c1olIdx,d1iracIdx,C2x,C2y,C2z,C2t,c2olIdx,d2iracIdx);
		break;
	}

	if	((fabs((re1P - re2P)/re2P) > 2E-6)||(fabs((im1P - im2P)/im2P) > 2E-6))
		printf	("Mismatch: %03d %03d %03d %03d %d %d\nQUDA: %le %le\ntmLQCD %le %le\n",C1x,C1y,C1z,C1t,c1olIdx,d1iracIdx,re2P,im2P,re1P,im1P);

  }	while(!feof(Caca1));

  fclose(Caca1);
  fclose(Caca2);
#endif

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif

  return 0;
}
