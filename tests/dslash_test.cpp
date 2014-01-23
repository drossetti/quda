#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <dslash_util.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

using namespace quda;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const int transfer = 0; // include transfer time in the benchmark?

double kappa5;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *spinorTmp;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp1=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac;

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat, 3 = MatPCDagMatPC, 4 = MatDagMat)
extern int test_type;

// Dirac operator type
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
extern QudaDagType dagger;

extern int niter;
extern char latfile[];

/*	EMPIEZAN MIERDAS DE ALEX	*/

void	reOrder	(double *array1, double *array2, const int arrayOffset)
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
			double	temp	 = array2[X];
			array2[X]	 = array1[i];
			array1[i]	 = temp;
		}
	}

	return;
}

/*	FIN MIERDAS DE ALEX	*/

void init(int argc, char **argv) {

  cuda_prec = prec;

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    errorQuda("Asqtad not supported.  Please try staggered_dslash_test instead");
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    dw_setDims(gauge_param.X, Lsdim);
    setKernelPackT(true);
  } else {
    setDims(gauge_param.X);
    setKernelPackT(false);
    Ls = 1;
  }

  setSpinorSiteSize(24);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_coeff = 0.001;
    inv_param.mu = 0.5;
    inv_param.epsilon = 0.0; 
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
//!    inv_param.twist_flavor = QUDA_TWIST_NONDEG_DOUBLET;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    inv_param.mass = 0.01;
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
  }

  inv_param.Ls = (inv_param.twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) ? Ls : 1;
  
//  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  if (inv_param.cpu_prec != gauge_param.cpu_prec) {
    errorQuda("Gauge and spinor CPU precisions must match");
  }
  inv_param.cuda_prec = cuda_prec;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  //inv_param.sp_pad = xdim*ydim*zdim/2;
  //inv_param.cl_pad = 24*24*24;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
//  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  switch(test_type) {
  case 0:
  case 1:
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    break;
  case 2:
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_SOLVE;
    break;
  case 3:
    inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    break;
  case 4:
    inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
    inv_param.solve_type = QUDA_DIRECT_SOLVE;
    break;
  default:
    errorQuda("Test type %d not defined\n", test_type);
  }

  inv_param.dslash_type = dslash_type;

  if ((dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
    inv_param.clover_coeff = 0.001;
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_FLOAT2_CLOVER_ORDER;
    //if (test_type > 0) {
//      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
//      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
//      hostCloverInv = hostClover; // fake it
      /*} else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      }*/
  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {

  }

  setVerbosity(QUDA_VERBOSE);

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.nColor = 3;
  csParam.nSpin = 4;
  if ((dslash_type == QUDA_TWISTED_MASS_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
    csParam.twistFlavor = inv_param.twist_flavor;
  }
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    csParam.nDim = 5;
    csParam.x[4] = Ls;
  }

//ndeg_tm    
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
    csParam.nDim = (inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS) ? 4 : 5;
    csParam.x[4] = Ls;    
  }


  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  if (test_type < 2 || test_type ==3) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }    
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);		/*		QUITAR PARA MIERDAS DE ALEX		*/
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);
  spinorTmp = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
//    read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
//    construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
	if	(read_custom_binary_gauge_field((double**)hostGauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline))
	{
		printf	("Fatal Error; Couldn't read gauge conf %s\n", latfile);
		exit	(1);
	}
  } else { // else generate a random SU(3) field*/
    construct_gauge_field(hostGauge, 0, gauge_param.cpu_prec, &gauge_param);
}

  inv_param.kappa = 1.0;
//  spinor->Source(QUDA_RANDOM_SOURCE);

//  FILE *Caca = fopen("/home/avaquero/src/tmLQCD-master/SpinorTm.In", "r+");
  FILE *Caca = fopen("SpinorTm.In", "r+");

  int		Cx,Cy,Cz,Ct,Cidx,colIdx,diracIdx;
  double	reP, imP;

  int	myRank;
  myRank	= comm_rank();

  do
  {
	fscanf(Caca, "%d %d %d %d %d %d %le %le\n", &Cx, &Cy, &Cz, &Ct, &colIdx, &diracIdx, &reP, &imP);

	if ((Ct >= tdim*(myRank+1)) || (Ct < tdim*myRank))
		continue;

	Ct -= tdim*myRank;

	int	oddbit = (Cx+Cy+Cz+Ct)&1;

	Cidx	= (Cx + Cy*xdim + Cz*xdim*ydim + Ct*xdim*ydim*zdim)/2;

	if	(oddbit)
 		if (test_type < 2 || test_type ==3)
			continue;
		else
			Cidx += V/2;

	unsigned long indexRe = ((Cidx*spinor->Nspin()+diracIdx)*spinor->Ncolor()+colIdx)*2;
	unsigned long indexIm = ((Cidx*spinor->Nspin()+diracIdx)*spinor->Ncolor()+colIdx)*2 + 1;
	double *inputRe = ((double*)(spinor->V()) + indexRe);
	double *inputIm = ((double*)(spinor->V()) + indexIm);

	double	phase	= (((double) (Ct+myRank*tdim))/(tdim*comm_dim(3)))*M_PI;

	*inputRe = cos(phase)*reP - sin(phase)*imP;
	*inputIm = cos(phase)*imP + sin(phase)*reP;
  }	while(!feof(Caca));

  fclose(Caca);

//  spinor->Source(QUDA_POINT_SOURCE);
//  spinor->Source(QUDA_RANDOM_SOURCE);

/*	FIN MIERDAS DE ALEX	*/
/*
  if ((dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
	diag = 1.0; // constant added to the diagonal
        construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
	diag = 1.0; // constant added to the diagonal
        construct_clover_field(hostCloverInv, norm, diag, inv_param.clover_cpu_prec);
     } else {
      if (test_type == 2 || test_type == 4) {
        construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
      } else {
        construct_clover_field(hostCloverInv, norm, diag, inv_param.clover_cpu_prec);
      }
    }
  }
  printfQuda("done.\n"); fflush(stdout);
  */
  initQuda(device);

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(hostGauge, &gauge_param);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    loadCloverQuda(NULL, NULL, &inv_param);
  }

  if (!transfer) {
//    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2 || test_type == 3) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    tmp1 = new cudaColorSpinorField(csParam);

    if (test_type == 2 || test_type == 4) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU %p %p\n", cudaSpinor, spinor);
    *cudaSpinor = *spinor;
    
    double cpu_norm = norm2(*spinor);
    double cuda_norm = norm2(*cudaSpinor);
    printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

    bool pc = (test_type != 2 && test_type != 4);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
    
    dirac = Dirac::create(diracParam);
  } else {
    double cpu_norm = norm2(*spinor);
    printfQuda("Source: CPU = %e\n", cpu_norm);
  }
}

void end() {
  if (!transfer) {
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp1;
    delete tmp2;
  }

  // release memory
//printf("FEOTON %p\n", spinor);

//  if (comm_rank() == 0)
	  delete spinor;

  delete spinorOut;
  delete spinorRef;
  delete spinorTmp;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if ((dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
//    if (hostClover != hostCloverInv && hostClover) free(hostClover);
//    free(hostCloverInv);
  }
  endQuda();

}

// execute kernel
double dslashCUDA(int niter) {

	printfQuda("Prolog\n");
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  for (int i = 0; i < niter; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 1:
    case 2:
      if (transfer) {
	MatQuda(spinorOut->V(), spinor->V(), &inv_param);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
      break;
    case 3:
    case 4:
      if (transfer) {
	MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
      } else {
	dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
      }
      break;
    }
  }
    
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000; //stopwatchReadSeconds();

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printfQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  return secs;
}

void dslashRef() {

  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  fflush(stdout);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_WILSON_DSLASH) {
    switch (test_type) {
    case 0:
      wil_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case 1:    
      wil_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param);
      break;
    case 2:
      wil_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case 3:
      wil_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, QUDA_DAG_NO, 
		inv_param.cpu_prec, gauge_param);
      wil_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.matpc_type, QUDA_DAG_YES, 
		inv_param.cpu_prec, gauge_param);
      break;
    case 4:
      wil_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, QUDA_DAG_NO, inv_param.cpu_prec, gauge_param);
      wil_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, QUDA_DAG_YES, inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if ((dslash_type == QUDA_TWISTED_MASS_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)) {
    switch (test_type) {
    case 0:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)
	tm_dslash(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, parity, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + tm_offset): (void*)((float*)ref1 + tm_offset);
    
	void *flv1 = spinor->V();
	void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + tm_offset): (void*)((float*)flv1 + tm_offset);
    
	tm_ndeg_dslash(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, 
	               parity, dagger, inv_param.matpc_type, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case 1:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)      
	tm_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + tm_offset): (void*)((float*)ref1 + tm_offset);
    
	void *flv1 = spinor->V();
	void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + tm_offset): (void*)((float*)flv1 + tm_offset);
    
	tm_ndeg_matpc(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);	
      }	
      break;
    case 2:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)      
	tm_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *evenOut = spinorRef->V();
	void *oddOut  = cpu_prec == sizeof(double) ? (void*)((double*)evenOut + tm_offset): (void*)((float*)evenOut + tm_offset);
    
	void *evenIn = spinor->V();
	void *oddIn  = cpu_prec == sizeof(double) ? (void*)((double*)evenIn + tm_offset): (void*)((float*)evenIn + tm_offset);
    
	tm_ndeg_mat(evenOut, oddOut, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, dagger, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case 3:    
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS){      
	tm_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, QUDA_DAG_NO, inv_param.cpu_prec, gauge_param);
	tm_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, QUDA_DAG_YES, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	errorQuda("Twisted mass solution type not supported");
      }
      break;
    case 4:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS){      
	tm_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     QUDA_DAG_NO, inv_param.cpu_prec, gauge_param);
	tm_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	errorQuda("Twisted mass solution type not supported");
      }
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    switch (test_type) {
    case 0:
      dw_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 1:    
      dw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 2:
      dw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 3:    
      dw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, QUDA_DAG_NO, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, QUDA_DAG_YES, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 4:
      dw_matdagmat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
    break; 
    default:
      printf("Test type not supported for domain wall\n");
      exit(-1);
    }
  } else {
    printfQuda("Unsupported dslash_type\n");
    exit(-1);
  }

  printfQuda("done.\n");
}


void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension   Ls_dimension dslash_type niter\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d             %d        %s   %d\n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     test_type, dagger, xdim, ydim, zdim, tdim, Lsdim,
	     get_dslash_type_str(dslash_type), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));

  return ;
    
}

/*	For testing	*/

void	dumpSpinor	(cpuColorSpinorField *out, const char *end, const double factor)
{
	FILE	*output;
	char	 name[256];
	void	*spinorData, *spinorOrder;
	int	myRank;

	myRank	= comm_rank();

	sprintf(name, "Spinor.%s.%d", end, myRank);

	spinorData	= out->V();

	if	((output = (fopen(name, "w+"))) == NULL)
	{
		printfQuda	("Error opening file\n");
		return;
	}

	spinorOrder	= (void*)malloc(V*24*sizeof(double));

	reOrder	((double*)spinorData, (double*)spinorOrder, 24);

	for	(int j=0; j<V; j++)
	{
		int	tC	= j/(xdim*ydim*zdim);
		int	zC	= (j - xdim*ydim*zdim*tC)/(xdim*ydim);
		int	yC	= (j - xdim*ydim*(zdim*tC + zC))/xdim;
		int	xC	= (j - xdim*(ydim*(zdim*tC + zC) + yC));

		double	phase	= ((double) tC)/tdim*M_PI;

		if	(tC/tdim != myRank)
			continue;

		for(int dirac=0; dirac<4; dirac++)
			for(int col=0; col<3; col++)
			{
//				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+((dirac+3)%4)*3+col;
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((double*)spinorOrder)[2*idx]*cos(phase) + ((double*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((double*)spinorOrder)[2*idx+1]*cos(phase) - ((double*)spinorOrder)[2*idx]*sin(phase);

				if	(fabs(rPart) < 1e-8)
					((double*)spinorOrder)[2*idx]	=	0.;

				if	(fabs(iPart) < 1e-8)
					((double*)spinorOrder)[2*idx+1]	=	0.;

				fprintf	(output, "%02d %02d %02d %02d %d %d %+2.6le %+2.6le\n", xC, yC, zC, tC+tdim*myRank, col, dirac, rPart*factor, iPart*factor);
			}
	}

	fclose(output);

	free(spinorOrder);

	return;
}

void	dumpContract	(cpuColorSpinorField *out, const char *end, double factor)
{
	FILE	*output;
	char	 name[256];
	void	*spinorData, *spinorOrder;
	int	myRank;

	myRank	= comm_rank();

	sprintf(name, "SpinorC.%s.%d", end, myRank);

	spinorData	= out->V();

	if	((output = (fopen(name, "w+"))) == NULL)
	{
		printfQuda	("Error opening file\n");
		return;
	}

	spinorOrder	= (void*)malloc(V*24*sizeof(double));

	reOrder	((double*)spinorData, (double*)spinorOrder, 24);

	for	(int j=0; j<V; j++)
	{
		int	tC	= j/(xdim*ydim*zdim);
		int	zC	= (j - xdim*ydim*zdim*tC)/(xdim*ydim);
		int	yC	= (j - xdim*ydim*(zdim*tC + zC))/xdim;
		int	xC	= (j - xdim*(ydim*(zdim*tC + zC) + yC));

		if	(tC/tdim != myRank)
			continue;

		double	phase	= ((double) tC)/tdim*M_PI;

		for(int col=0; col<3; col++)
		{
			double	rE	= 0.;
			double	iM	= 0.;
/*
			int	idx0 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+0*3+col;
			int	idx1 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+1*3+col;
			int	idx2 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+2*3+col;
			int	idx3 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+3*3+col;

			rE	+= 2.*(((double*)spinorOrder)[2*idx1]*((double*)spinorOrder)[2*idx3] + ((double*)spinorOrder)[2*idx1+1]*((double*)spinorOrder)[2*idx3+1]);
			rE	+= 2.*(((double*)spinorOrder)[2*idx0]*((double*)spinorOrder)[2*idx2] + ((double*)spinorOrder)[2*idx0+1]*((double*)spinorOrder)[2*idx2+1]);
*/
			for(int dirac=0; dirac<2; dirac++)
			{
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((double*)spinorOrder)[2*idx]*cos(phase) + ((double*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((double*)spinorOrder)[2*idx+1]*cos(phase) - ((double*)spinorOrder)[2*idx]*sin(phase);

				rE	+= rPart*rPart + iPart*iPart;
			}

			for(int dirac=2; dirac<4; dirac++)
			{
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((double*)spinorOrder)[2*idx]*cos(phase) + ((double*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((double*)spinorOrder)[2*idx+1]*cos(phase) - ((double*)spinorOrder)[2*idx]*sin(phase);

				rE	-= rPart*rPart + iPart*iPart;
			}

			if	(fabs(rE) < 1e-8)
				rE	= 0.;

			fprintf	(output, "%02d %02d %02d %02d %d %+2.5le %+2.5le\n", xC, yC, zC, tC+tdim*myRank, col, rE*factor, iM*factor);
		}
	}

	fclose(output);

	free(spinorOrder);

	return;
}

void	dumpVolume	(cpuColorSpinorField *out, const char *end, double factor)
{
	FILE	*output;
	char	 name[256];
	void	*spinorData, *spinorOrder;
	double	*outCont;
	int	myRank;

	myRank	= comm_rank();

	sprintf(name, "SpinorV.%s.%d", end, myRank);

	spinorData	= out->V();

	if	((output = (fopen(name, "w+"))) == NULL)
	{
		printfQuda	("Error opening file\n");
		return;
	}

	spinorOrder	= (void*)malloc(V*24*sizeof(double));

	reOrder	((double*)spinorData, (double*)spinorOrder, 24);

	if	((outCont = ((double *) malloc(sizeof(double)*tdim))) == NULL)
	{
		printfQuda	("Error allocating memory for contraction\n");
		return;
	}

	for	(int t=0; t<tdim; t++)
		outCont[t]	= 0.;

	for	(int j=0; j<V; j++)
	{
		int	tC	= j/(xdim*ydim*zdim);
		int	zC	= (j - xdim*ydim*zdim*tC)/(xdim*ydim);
		int	yC	= (j - xdim*ydim*(zdim*tC + zC))/xdim;
		int	xC	= (j - xdim*(ydim*(zdim*tC + zC) + yC));

		if	(tC/tdim != myRank)
			continue;

		double	phase	= ((double) tC)/tdim*M_PI;

		double	rC	= 0.;

		for(int col=0; col<3; col++)
		{
			double	rE	= 0.;
			double	iM	= 0.;
/*
			int	idx0 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+0*3+col;
			int	idx1 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+1*3+col;
			int	idx2 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+2*3+col;
			int	idx3 = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+3*3+col;

			rE	+= ((double*)spinorOrder)[2*idx1]*((double*)spinorOrder)[2*idx3] + ((double*)spinorOrder)[2*idx1+1]*((double*)spinorOrder)[2*idx3+1];
			rE	+= ((double*)spinorOrder)[2*idx0]*((double*)spinorOrder)[2*idx2] + ((double*)spinorOrder)[2*idx0+1]*((double*)spinorOrder)[2*idx2+1];
*/
			for(int dirac=0; dirac<2; dirac++)
			{
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((double*)spinorOrder)[2*idx]*cos(phase) + ((double*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((double*)spinorOrder)[2*idx+1]*cos(phase) - ((double*)spinorOrder)[2*idx]*sin(phase);

				rE	+= rPart*rPart + iPart*iPart;
			}

			for(int dirac=2; dirac<4; dirac++)
			{
				int	idx = (xC+(xdim*(yC+ydim*(zC+zdim*tC))))*12+dirac*3+col;

				double	rPart	=	((double*)spinorOrder)[2*idx]*cos(phase) + ((double*)spinorOrder)[2*idx+1]*sin(phase);
				double	iPart	=	((double*)spinorOrder)[2*idx+1]*cos(phase) - ((double*)spinorOrder)[2*idx]*sin(phase);

				rE	-= rPart*rPart + iPart*iPart;
			}

			rC	+= rE;
		}

		outCont[tC]	+= rC;
	}

	for	(int t=0; t<tdim; t++)
	{
		if	(fabs(outCont[t]) < 1e-8)
			outCont[t]	= 0.;

		fprintf	(output, "%02d %+2.5le %+2.5le\n", t+tdim*myRank, outCont[t]*factor, 0.0);
	}

	fclose(output);

	free(spinorOrder);

	return;
}


/*	End testing	*/



extern void usage(char**);


int main(int argc, char **argv)
{

  for (int i =1;i < argc; i++){    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  init(argc, argv);

  float spinorGiB = (float)Vh*spinorSiteSize*inv_param.cuda_prec / (1 << 30);
  printfQuda("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printfQuda("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  
  int attempts = 1;
  dslashRef();
    
  for (int i=0; i<attempts; i++) {

    if (tune) { // warm-up run
      printfQuda("Tuning...\n");
      setTuning(QUDA_TUNE_YES);
      dslashCUDA(1);
    }
    printfQuda("Executing %d kernel loops...\n", niter);
    if (!transfer) dirac->Flops();
    double secs = dslashCUDA(niter);
    printfQuda("done.\n");

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printfQuda("%fus per kernel call\n", 1e6*secs / niter);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int spinor_floats = test_type ? 2*(7*24+24)+24 : 7*24+24;
    if (inv_param.cuda_prec == QUDA_HALF_PRECISION) 
      spinor_floats += test_type ? 2*(7*2 + 2) + 2 : 7*2 + 2; // relative size of norm is twice a short
    int gauge_floats = (test_type ? 2 : 1) * (gauge_param.gauge_fix ? 6 : 8) * gauge_param.reconstruct;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      gauge_floats += test_type ? 72*2 : 72;
    }
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printfQuda("GB/s = %f\n\n", 
	       (double)Vh*(Ls*spinor_floats+gauge_floats)*inv_param.cuda_prec/((secs/niter)*1e+9));
    
    if (!transfer) {
      double norm2_cpu = norm2(*spinorRef);
      double norm2_cuda= norm2(*cudaSpinorOut);
      double norm2_cpu_cuda= norm2(*spinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      double norm2_cpu = norm2(*spinorRef);
      double norm2_cpu_cuda= norm2(*spinorOut);
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n",  norm2_cpu, norm2_cpu_cuda);
    }
    
    cpuColorSpinorField::Compare(*spinorRef, *spinorOut);
  }    

  const double	factor	= 2.*inv_param.kappa;

  dumpVolume(spinor, "In", 1.);
  dumpSpinor(spinor, "In", 1.);
  dumpSpinor(spinorOut, "Out", factor);
  dumpContract(spinor, "In", 1.);
  dumpContract(spinorOut, "Out", factor*factor);
  dumpVolume(spinorOut, "Out", factor*factor);

  end();

  finalizeComms();
}
