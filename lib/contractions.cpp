#include	<contractQuda.h>
/*#include	<cufft.h>

void	doCudaFFT	(void *cnRes_vv, int xdim, int ydim, int zdim, int tdim);
*/
void	setDiracPreParamCG	(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
{
	setDiracParam(diracParam, inv_param, pc);

	diracParam.gauge = gaugePrecondition;
	diracParam.fatGauge = gaugeFatPrecondition;
	diracParam.longGauge = gaugeLongPrecondition;    
	diracParam.clover = cloverPrecondition;
	diracParam.cloverInv = cloverInvPrecondition;

	for (int i=0; i<4; i++)
		diracParam.commDim[i] = 1;
}

void	createDiracCG		(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
{
	DiracParam diracParam;
	DiracParam diracSloppyParam;
	DiracParam diracPreParam;

	setDiracParam(diracParam, &param, pc_solve);
	setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
	setDiracPreParamCG(diracPreParam, &param, pc_solve);

	d = Dirac::create(diracParam);
	dSloppy = Dirac::create(diracSloppyParam);
	dPre = Dirac::create(diracPreParam);
}

/*	TODO Arregla la corriente conservada!!!	*/

void	loopPlainCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes, void **cnD, void **cnC)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");

	if	(param->cuda_prec == QUDA_DOUBLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(float)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
		errorQuda	("Error: Contraction not supported in half precision.\n");
	if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
        	printfQuda	("Error allocating memory for contraction results in CPU.\n");
	        exit		(0);
	}

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
	static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
	static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if (nb==0.0) errorQuda("Source has zero norm");

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	// rescale the source and solution vectors to help prevent the onset of underflow
	if (param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
	{
		axCuda(1.0/sqrt(nb), *b);
		axCuda(1.0/sqrt(nb), *x);
	}

	setTuning(param->tune);

	dirac.prepare	(in, out, *x, *b, param->solution_type);
	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	// solution_type specifies *what* system is to be solved.
	// solve_type specifies *how* the system is to be solved.
	//
	// We have the following four cases (plus preconditioned variants):
	//
	// solution_type    solve_type    Effect
	// -------------    ----------    ------
	// MAT              DIRECT        Solve Ax=b
	// MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
	// MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
	// MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
	//
	// We generally require that the solution_type and solve_type
	// preconditioning match.  As an exception, the unpreconditioned MAT
	// solution_type may be used with any solve_type, including
	// DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
	// preconditioned source and reconstruction of the full solution are
	// taken care of by Dirac::prepare() and Dirac::reconstruct(),
	// respectively.

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}
	else if	(!mat_solution && direct_solve)
	{						// perform the first of two solves: A^dag y = b
		DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		copyCuda(*in, *out);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if (direct_solve)
	{
		DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}
	else
	{
		DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(sqrt(nb), *x);		// rescale the solution

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};
	int	cDim[4]	 = {   1,    1,    1,    1};

	contractCuda	(b->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractCuda	(b->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes)[ix]	+= ((double*)h_ctrn)[ix];

	printfQuda	("Locals contracted\n");
	fflush		(stdout);

	for     (int mu=0; mu<4; mu++)
	{
		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();
		
		contractCuda	(b->Even(), tmp2->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(b->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		cudaMemcpy	(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);
		
		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(b->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(b->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(b->Even(), tmp2->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(b->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(b->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(b->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD[mu])[ix]	-= ((double*)h_ctrn)[ix];

		cudaMemcpy	(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC[mu])[ix]	-= ((double*)h_ctrn)[ix];
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;       
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete	b;
	delete	h_b;

	delete  tmp2;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	loopHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes, void **cnD, void **cnC)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");
	fflush	(stdout);

	if	(param->cuda_prec == QUDA_DOUBLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(float)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
		errorQuda	("Error: Contraction not supported in half precision.\n");

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if (nb==0.0) errorQuda("Source has zero norm");

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
	{
		axCuda(1.0/sqrt(nb), *b);
		axCuda(1.0/sqrt(nb), *x);
	}

	setTuning(param->tune);

	dirac.prepare	(in, out, *x, *b, param->solution_type);
	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	// solution_type specifies *what* system is to be solved.
	// solve_type specifies *how* the system is to be solved.
	//
	// We have the following four cases (plus preconditioned variants):
	//
	// solution_type    solve_type    Effect
	// -------------    ----------    ------
	// MAT              DIRECT        Solve Ax=b
	// MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
	// MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
	// MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
	//
	// We generally require that the solution_type and solve_type
	// preconditioning match.  As an exception, the unpreconditioned MAT
	// solution_type may be used with any solve_type, including
	// DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
	// preconditioned source and reconstruction of the full solution are
	// taken care of by Dirac::prepare() and Dirac::reconstruct(),
	// respectively.

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}
	else if	(!mat_solution && direct_solve)
	{						// perform the first of two solves: A^dag y = b
		DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		copyCuda(*in, *out);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if (direct_solve)
	{
		DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}
	else
	{
		DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(sqrt(nb), *x);		// rescale the solution

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};
	int	cDim[4]	 = {   1,    1,    1,    1};

	gamma5Cuda	(&(tmp2->Even()), &(b->Even()), blockTwost);
	gamma5Cuda	(&(tmp2->Odd()),  &(b->Odd()),  blockTwost);

	delete  h_b;
	delete  b;

	tmp3		 = new cudaColorSpinorField(cudaParam);

	printfQuda	("Synchronizing\n");

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	for	(int i = 0; i<4; i++)
	{
		wilsonDslashCuda	(&(tmp3->Even()), *gaugePrecise, &(tmp2->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, cDim, profileContract);
		wilsonDslashCuda	(&(tmp3->Odd()),  *gaugePrecise, &(tmp2->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, cDim, profileContract);

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return;
		}

		double	mu_flavour	 = x->TwistFlavor()*param->mu;

		twistGamma5Cuda		(&(tmp2->Even()), &(tmp3->Even()), 1, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
		twistGamma5Cuda		(&(tmp2->Odd()),  &(tmp3->Odd()),  1, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	gamma5Cuda	(&(tmp3->Even()), &(tmp2->Even()), blockTwost);
	gamma5Cuda	(&(tmp3->Odd()),  &(tmp2->Odd()),  blockTwost);

	contractCuda	(tmp3->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractCuda	(tmp3->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes)[ix]	+= ((double*)h_ctrn)[ix];

	printfQuda	("Locals contracted\n");
	fflush		(stdout);

	for     (int mu=0; mu<4; mu++)
	{
		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp3->Even(), tmp2->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp3->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		cudaMemcpy	(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);
		
		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp3->Even(), tmp2->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp3->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDev		(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDev		(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD[mu])[ix]	-= ((double*)h_ctrn)[ix];

		cudaMemcpy	(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC[mu])[ix]	-= ((double*)h_ctrn)[ix];
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;        
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete  tmp2;
	delete  tmp3;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	oneEndTrickCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes_gv, void *cnRes_vv, void **cnD_gv, void **cnD_vv, void **cnC_gv, void **cnC_vv)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

//	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);
/*
	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
	}

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		cudaCloverField *tempC	 = cloverSloppy;
		cloverSloppy		 = cloverPrecondition;
		cloverPrecondition	 = tempC;
		tempC			 = cloverInvSloppy;
		cloverInvSloppy		 = cloverInvPrecondition;
		cloverInvPrecondition	 = tempC;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;

		tempPrec		 		= param->clover_cuda_prec_sloppy;
		param->clover_cuda_prec_sloppy		= param->clover_cuda_prec_precondition;
		param->clover_cuda_prec_precondition	= tempPrec;
	}
*/
	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;
	cudaColorSpinorField *tmp4	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");
	fflush	(stdout);

	if	(param->cuda_prec == QUDA_DOUBLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnC, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset(ctrnC, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(float)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
		errorQuda	("Error: Contraction not supported in half precision.\n");

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if	(nb==0.0) errorQuda("Source has zero norm");

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
	{
		axCuda(1.0/sqrt(nb), *b);
		axCuda(1.0/sqrt(nb), *x);
	}

	setTuning(param->tune);
	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	// solution_type specifies *what* system is to be solved.
	// solve_type specifies *how* the system is to be solved.
	//
	// We have the following four cases (plus preconditioned variants):
	//
	// solution_type    solve_type    Effect
	// -------------    ----------    ------
	// MAT              DIRECT        Solve Ax=b
	// MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
	// MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
	// MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
	//
	// We generally require that the solution_type and solve_type
	// preconditioning match.  As an exception, the unpreconditioned MAT
	// solution_type may be used with any solve_type, including
	// DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
	// preconditioned source and reconstruction of the full solution are
	// taken care of by Dirac::prepare() and Dirac::reconstruct(),
	// respectively.

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}
	else if	(!mat_solution && direct_solve)
	{						// perform the first of two solves: A^dag y = b
		DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		copyCuda(*in, *out);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if (direct_solve)
	{
		DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}
	else
	{
		DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(sqrt(nb), *x);		// rescale the solution

	delete  h_b;
	delete  b;

	tmp3		 = new cudaColorSpinorField(cudaParam);
	tmp4		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);

	DiracParam	dWParam;

	dWParam.type		 = QUDA_WILSON_DIRAC;
	dWParam.matpcType	 = QUDA_MATPC_EVEN_EVEN;
	dWParam.dagger		 = QUDA_DAG_NO;
	dWParam.gauge		 = gaugePrecise;
	dWParam.kappa		 = param->kappa;
	dWParam.mass		 = 1./(2.*param->kappa) - 4.;
	dWParam.m5		 = 0.;
	dWParam.mu		 = 0.;
//	dWParam.verbose		 = param->verbosity;

	for	(int i=0; i<4; i++)
        	dWParam.commDim[i]	 = 1;   // comms are always on

	DiracWilson	*dW	 = new DiracWilson(dWParam);

	dW->M(*tmp4,*x);

	delete	dW;

	gamma5Cuda	(&(tmp3->Even()), &(tmp4->Even()), blockTwost);
	gamma5Cuda	(&(tmp3->Odd()),  &(tmp4->Odd()),  blockTwost);

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};

	long int	sizeBuffer;

	if	(x->Precision() == QUDA_SINGLE_PRECISION)
	{
		sizeBuffer	= sizeof(float)*32*X[0]*X[1]*X[2]*X[3];

		contractGamma5Cuda	(x->Even(), tmp3->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  tmp3->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((float*) cnRes_gv)[ix]	+= ((float*)h_ctrn)[ix];

		contractGamma5Cuda	(x->Even(), x->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  x->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((float *) cnRes_vv)[ix]       -= ((float*)h_ctrn)[ix];

		printfQuda	("Locals contracted\n");
		fflush		(stdout);

		for	(int mu=0; mu<4; mu++)	//Hasta 4
		{
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);		//Term0
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
	                cudaDeviceSynchronize	();

			cudaMemcpy		(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);

			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

			contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((float2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (C Sum)
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((float2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);

	                contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((float2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (D Dif)
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((float2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			covDev		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((float2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term1
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((float2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);

			contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((float2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term3
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((float2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
			cudaDeviceSynchronize	();

			cudaMemcpy		(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((float *) cnD_gv[mu])[ix]	+= ((float*)h_ctrn)[ix];

			cudaMemcpy		(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

			for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((float *) cnC_gv[mu])[ix]	+= ((float*)h_ctrn)[ix];
		}

		for     (int mu=0; mu<4; mu++)
		{
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);
      	
      			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
      			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
      	
		      	contractGamma5Cuda	(tmp4->Even(), x->Even(), ((float2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
      			contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((float2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		      	cudaDeviceSynchronize	();

	      		cudaMemcpy	(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);

			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((float2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((float2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			contractGamma5Cuda	(tmp4->Even(), x->Even(), ((float2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((float2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
			cudaDeviceSynchronize	();

			cudaMemcpy	(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((float *) cnD_vv[mu])[ix]	-= ((float*)h_ctrn)[ix];

			cudaMemcpy	(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((float *) cnC_vv[mu])[ix]	-= ((float*)h_ctrn)[ix];
		}
	}
	else if	(x->Precision() == QUDA_DOUBLE_PRECISION)
	{
		sizeBuffer	= sizeof(double)*32*X[0]*X[1]*X[2]*X[3];

		contractGamma5Cuda	(x->Even(), tmp3->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double*) cnRes_gv)[ix]	+= ((double*)h_ctrn)[ix];

		contractGamma5Cuda	(x->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnRes_vv)[ix]       -= ((double*)h_ctrn)[ix];

		printfQuda	("Locals contracted\n");
		fflush		(stdout);

		for	(int mu=0; mu<4; mu++)	//Hasta 4
		{
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);		//Term0
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
	                cudaDeviceSynchronize	();

			cudaMemcpy		(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);

			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

			contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (C Sum)
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);

	                contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (D Dif)
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term1
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);

			contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term3
			contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
			cudaDeviceSynchronize	();

			cudaMemcpy		(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((double *) cnD_gv[mu])[ix]	+= ((double*)h_ctrn)[ix];

			cudaMemcpy		(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

			for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((double *) cnC_gv[mu])[ix]	+= ((double*)h_ctrn)[ix];
		}

		for     (int mu=0; mu<4; mu++)
		{
			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);
      	
      			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
      			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		      	cudaDeviceSynchronize	();

	      		cudaMemcpy	(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
      	
		      	contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);
      			contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);

		      	contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
      			contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
//		      	cudaDeviceSynchronize	();

//	      		cudaMemcpy	(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);

			covDev		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);
			covDev		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);

			contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

			contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
			cudaDeviceSynchronize	();

			cudaMemcpy	(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((double *) cnD_vv[mu])[ix]	-= ((double*)h_ctrn)[ix];

			cudaMemcpy	(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((double *) cnC_vv[mu])[ix]	-= ((double*)h_ctrn)[ix];
		}
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;                        
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete  tmp3;
	delete  tmp4;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());
/*
	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		cudaCloverField *tempC		 = cloverSloppy;
		cloverSloppy			 = cloverPrecondition;
		cloverPrecondition		 = tempC;

		tempC				 = cloverInvSloppy;
		cloverInvSloppy			 = cloverInvPrecondition;
		cloverInvPrecondition		 = tempC;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;

		tempPrec				 = param->clover_cuda_prec_sloppy;
		param->clover_cuda_prec_sloppy		 = param->clover_cuda_prec_precondition;
		param->clover_cuda_prec_precondition	 = tempPrec;
	}
*/
	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	tDilHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;

	const int *X = cudaGauge->X();

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if	(nb==0.0) errorQuda("Source has zero norm");

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	// rescale the source and solution vectors to help prevent the onset of underflow
	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
	{
		axCuda(1.0/sqrt(nb), *b);
		axCuda(1.0/sqrt(nb), *x);
	}

	setTuning(param->tune);

	dirac.prepare	(in, out, *x, *b, param->solution_type);
	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	// solution_type specifies *what* system is to be solved.
	// solve_type specifies *how* the system is to be solved.
	//
	// We have the following four cases (plus preconditioned variants):
	//
	// solution_type    solve_type    Effect
	// -------------    ----------    ------
	// MAT              DIRECT        Solve Ax=b
	// MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
	// MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
	// MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
	//
	// We generally require that the solution_type and solve_type
	// preconditioning match.  As an exception, the unpreconditioned MAT
	// solution_type may be used with any solve_type, including
	// DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
	// preconditioned source and reconstruction of the full solution are
	// taken care of by Dirac::prepare() and Dirac::reconstruct(),
	// respectively.

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}
	else if	(!mat_solution && direct_solve)
	{						// perform the first of two solves: A^dag y = b
		DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		copyCuda(*in, *out);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if (direct_solve)
	{
		DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}
	else
	{
		DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(sqrt(nb), *x);		// rescale the solution

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);
	int		LX[4]		 = { X[0], X[1], X[2], X[3] };
	int		commDim[4]	 = { 1, 1, 1, 1 };

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	tmp3		 = new cudaColorSpinorField(cudaParam);

	wilsonDslashCuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, commDim, profileContract);
	wilsonDslashCuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, commDim, profileContract);

	double	mu_flavour	 = x->TwistFlavor()*param->mu;

	for	(int i = 0; i<3; i++)
	{
		twistGamma5Cuda		(&(tmp3->Even()), &(tmp2->Even()), 0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
		twistGamma5Cuda		(&(tmp3->Odd()),  &(tmp2->Odd()),  0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);

		wilsonDslashCuda	(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, commDim, profileContract);
		wilsonDslashCuda	(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, commDim, profileContract);

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return;
		}
	}

	twistGamma5Cuda		(&(tmp3->Even()), &(tmp2->Even()), 0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
	twistGamma5Cuda		(&(tmp3->Odd()),  &(tmp2->Odd()),  0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	delete	tmp2;

	const int	Lt	 = X[3]*comm_dim(3);
	int		cT	 = 0;

	for	(int time = tSlice; time < (tSlice+Lt); time += (Lt/nCoh))
	{
		int	tempT	 = time%Lt;

		if	((tempT/X[3]) == comm_coord(3))
		{
			int	tC	 = tempT - comm_coord(3)*X[3];

			contractTsliceCuda	(b->Even(), tmp3->Even(), ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_EVEN_PARITY);
			contractTsliceCuda	(b->Odd(),  tmp3->Odd(),  ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_ODD_PARITY);
		}

		cT++;
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_TOTAL);
	profileContract.Stop(QUDA_PROFILE_COMPUTE);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;        
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	h_x;
	delete	x;
	delete	h_b;
	delete	b;

	delete  tmp3;

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	tDilutionCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;

	const int *X = cudaGauge->X();

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if	(nb==0.0) errorQuda("Source has zero norm");

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	// rescale the source and solution vectors to help prevent the onset of underflow
	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
	{
		axCuda(1.0/sqrt(nb), *b);
		axCuda(1.0/sqrt(nb), *x);
	}

	setTuning(param->tune);

	dirac.prepare	(in, out, *x, *b, param->solution_type);
	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	// solution_type specifies *what* system is to be solved.
	// solve_type specifies *how* the system is to be solved.
	//
	// We have the following four cases (plus preconditioned variants):
	//
	// solution_type    solve_type    Effect
	// -------------    ----------    ------
	// MAT              DIRECT        Solve Ax=b
	// MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
	// MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
	// MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
	//
	// We generally require that the solution_type and solve_type
	// preconditioning match.  As an exception, the unpreconditioned MAT
	// solution_type may be used with any solve_type, including
	// DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
	// preconditioned source and reconstruction of the full solution are
	// taken care of by Dirac::prepare() and Dirac::reconstruct(),
	// respectively.

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}
	else if	(!mat_solution && direct_solve)
	{						// perform the first of two solves: A^dag y = b
		DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		copyCuda(*in, *out);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if (direct_solve)
	{
		DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}
	else
	{
		DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
		SolverParam solverParam(*param);
		Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
		(*solve)(*out, *in);
		solverParam.updateInvertParam(*param);
		delete solve;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(sqrt(nb), *x);		// rescale the solution

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);
	int		LX[4]		 = { X[0], X[1], X[2], X[3] };
	int		commDim[4]	 = { 1, 1, 1, 1 };

	printfQuda	("Synchronizing\n");

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	const int	Lt	 = X[3]*comm_dim(3);
	int		cT	 = 0;

	for	(int time = tSlice; time < (tSlice+Lt); time += (Lt/nCoh))
	{
		int	tempT	 = time%Lt;

		if	((tempT/X[3]) == comm_coord(3))
		{
			int	tC	 = tempT - comm_coord(3)*X[3];

			contractTsliceCuda	(b->Even(), x->Even(), ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_EVEN_PARITY);
			contractTsliceCuda	(b->Odd(),  x->Odd(),  ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_ODD_PARITY);
		}

		cT++;
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_TOTAL);
	profileContract.Stop(QUDA_PROFILE_COMPUTE);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	h_x;
	delete	x;

	delete  h_b;
	delete  b;

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	tuneOneEndTrick	(void *hp_x, void *hp_b, QudaInvertParam *param, void ***cnRes_gv, void ***cnRs2_gv, void **cnRes_vv, void **cnRs2_vv,
					const int nSteps, const bool Cr, void ***cnCor_gv, void ***cnCr2_gv, void **cnCor_vv, void **cnCr2_vv)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

//	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	pushVerbosity(param->verbosity);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);
/*
	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
	}

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		cudaCloverField *tempC	 = cloverSloppy;
		cloverSloppy		 = cloverPrecondition;
		cloverPrecondition	 = tempC;
		tempC			 = cloverInvSloppy;
		cloverInvSloppy		 = cloverInvPrecondition;
		cloverInvPrecondition	 = tempC;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;

		tempPrec		 		= param->clover_cuda_prec_sloppy;
		param->clover_cuda_prec_sloppy		= param->clover_cuda_prec_precondition;
		param->clover_cuda_prec_precondition	= tempPrec;
	}
*/
	checkInvertParam(param);

	// It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
	// solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
	// for now, though, so here we factorize everything for convenience.

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
			   (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
	bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
			(param->solve_type == QUDA_NORMOP_PC_SOLVE);
	bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
			    (param->solution_type ==  QUDA_MATPC_SOLUTION);
	bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
			    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if (!pc_solve) param->spinorGiB *= 2;
	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	} else {
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
	}

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *bb	 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *xx	 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *inT	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *outT	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;
	cudaColorSpinorField *tmp4	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");
	fflush	(stdout);

	if	(param->cuda_prec == QUDA_DOUBLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
	{
		if	((cudaMallocHost(&h_ctrn, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	        	errorQuda	("Error allocating memory for contraction results in CPU.\n");

		cudaMemset(h_ctrn, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		if	((cudaMalloc(&ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
			errorQuda	("Error allocating memory for contraction results in GPU.\n");

		cudaMemset	(ctrnS, 0, sizeof(float)*32*X[0]*X[1]*X[2]*X[3]);

		printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(float)*64*X[0]*X[1]*X[2]*X[3]);
	}
	else if	(param->cuda_prec == QUDA_SINGLE_PRECISION)
		errorQuda	("Error: Contraction not supported in half precision.\n");

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);
	bb			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE))
			errorQuda("Initial guess not supported for two-pass solver");

		x			 = new cudaColorSpinorField(*h_x, cudaParam); // solution
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		xx			 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
		xx			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	double nb	 = norm2(*b);
	if	(nb==0.0) errorQuda("Source has zero norm");

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(1.0/sqrt(nb), *b);

	setTuning(param->tune);

	double		tempData[2*X[3]][nSteps];
	double		tempVecD[2*X[3]][3][nSteps];
	const int	sV = X[0]*X[1]*X[2];
	const int	tV = sV*X[3];

	if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
		axCuda(1.0/sqrt(nb), *x);

	dirac.prepare	(inT, outT, *xx, *bb, param->solution_type);
	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	if	(pc_solution && !pc_solve)
		errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

	if	(!mat_solution && !pc_solution && pc_solve)
		errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

	if	(mat_solution && !direct_solve)
	{						// prepare source: b' = A^dag b
		cudaColorSpinorField tmp(*in);
		dirac.Mdag(*in, tmp);
	}

	DiracMdagM mat(dirac), matSloppy(diracSloppy), matPrecon(diracPre);
	SolverParam solverParam(*param);

	profileInvert.Start(QUDA_PROFILE_INIT);

	// Check to see that we're not trying to invert on a zero-field source    
	const double b2 = norm2(*in);

	if(b2 == 0)
	{
		profileInvert.Stop(QUDA_PROFILE_INIT);
		printfQuda("Warning: inverting on zero-field source\n");
		out=in;
		solverParam.true_res = 0.0;
		solverParam.true_res_hq = 0.0;
		return;
	}

	cudaColorSpinorField r(*in);

	ColorSpinorParam csParam(*out);
	csParam.create = QUDA_ZERO_FIELD_CREATE;
	cudaColorSpinorField y(*in, csParam);
	cudaColorSpinorField yT(*inT, csParam);


	mat(r, *out, y);
//	zeroCuda(y);

	double r2 = xmyNormCuda(*in, r);

	csParam.setPrecision(solverParam.precision_sloppy);
	cudaColorSpinorField Ap(*out, csParam);
	cudaColorSpinorField tmp(*out, csParam);

	cudaColorSpinorField *tmp2_p = &tmp;
	//tmp only needed for multi-gpu Wilson-like kernels
	if	(mat.Type() != typeid(DiracStaggeredPC).name() && mat.Type() != typeid(DiracStaggered).name())
	{
		tmp2_p = new cudaColorSpinorField(*out, csParam);
	}

	cudaColorSpinorField &tmp2 = *tmp2_p;

	cudaColorSpinorField *r_sloppy;
	if	(solverParam.precision_sloppy == out->Precision())
	{
		csParam.create = QUDA_REFERENCE_FIELD_CREATE;
		r_sloppy = &r;
	} else {
		csParam.create = QUDA_COPY_FIELD_CREATE;
		r_sloppy = new cudaColorSpinorField(r, csParam);
	}

	cudaColorSpinorField *x_sloppy;
	if	(solverParam.precision_sloppy == out->Precision() || !solverParam.use_sloppy_partial_accumulator)
	{
		csParam.create = QUDA_REFERENCE_FIELD_CREATE;
		x_sloppy = out;
	} else {
		csParam.create = QUDA_COPY_FIELD_CREATE;
		x_sloppy = new cudaColorSpinorField(*out, csParam);
	}

	cudaColorSpinorField &xSloppy = *x_sloppy;
	cudaColorSpinorField &rSloppy = *r_sloppy;
	cudaColorSpinorField p(rSloppy);

	if	(out != &xSloppy) {
		copyCuda(y,*out);
		zeroCuda(xSloppy);
	} else {
		zeroCuda(y);
	}

	const bool use_heavy_quark_res = (solverParam.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

	profileInvert.Stop(QUDA_PROFILE_INIT);
	profileInvert.Start(QUDA_PROFILE_PREAMBLE);

	double r2_old;

	double stop = b2*solverParam.tol*solverParam.tol; // stopping condition of solver

	double heavy_quark_res = 0.0; // heavy quark residual
	if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(*out,r).z);
	int heavy_quark_check = 10; // how often to check the heavy quark residual

	double alpha=0.0, beta=0.0;
	double pAp;
	int rUpdate = 0;

	double rNorm = sqrt(r2);
	double r0Norm = rNorm;
	double maxrx = rNorm;
	double maxrr = rNorm;
	double delta = solverParam.delta;

	// this solverParameter determines how many consective reliable update
	// resiudal increases we tolerate before terminating the solver,
	// i.e., how long do we want to keep trying to converge
	int maxResIncrease = 0; // 0 means we have no tolerance 

	profileInvert.Stop(QUDA_PROFILE_PREAMBLE);
	blas_flops = 0;

	int steps_since_reliable = 1;

	tmp3		 = new cudaColorSpinorField(cudaParam);
	tmp4		 = new cudaColorSpinorField(cudaParam);

	//EMPIEZA
	for	(int it=0; it<nSteps;it++)
	{
/*
		else if	(!mat_solution && direct_solve)
		{						// perform the first of two solves: A^dag y = b
			DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
			SolverPara solverParam(*param);
			Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
			(*solve)(*out, *in);
			copyCuda(*in, *out);
			solverParam.updateInvertParam(*param);
			delete solve;
		}
*/
		if (direct_solve)
		{
/*
			DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
			SolverParam solverParam(*param);
			Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
			(*solve)(*out, *in);
			solverParam.updateInvertParam(*param);
			delete solve;
*/
		}
		else
		{
    profileInvert.Start(QUDA_PROFILE_COMPUTE);
    int k=it*solverParam.maxiter;

    printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));

    while ( (r2 > stop) &&
            k < solverParam.maxiter*(it + 1)) {
      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp

      double sigma;

      bool breakdown = false;

      if (solverParam.pipeline) {
        double3 triplet = tripleCGReductionCuda(rSloppy, Ap, p);
        r2 = triplet.x; double Ap2 = triplet.y; pAp = triplet.z;
        r2_old = r2;

        alpha = r2 / pAp;
        sigma = alpha*(alpha * Ap2 - pAp);
        if (sigma < 0.0 || steps_since_reliable==0) { // sigma condition has broken down
          r2 = axpyNormCuda(-alpha, Ap, rSloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;
        pAp = reDotProductCuda(p, Ap);
        alpha = r2 / pAp;

        // here we are deploying the alternative beta computation 
        Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy);
        r2 = real(cg_norm); // (r_new, r_new)
        sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( (r2 < stop) && delta >= solverParam.tol) updateX = 1;

      if ( !(updateR || updateX)) {                     //Fix Update reliable
        //beta = r2 / r2_old;
        beta = sigma / r2_old; // use the alternative beta computation

        if (solverParam.pipeline && !breakdown) tripleCGUpdateCuda(alpha, beta, Ap, xSloppy, rSloppy, p);
        else axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
        if (use_heavy_quark_res && k%heavy_quark_check==0) {
          copyCuda(tmp,y);
          heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
        }

        steps_since_reliable++;
      } else {
        axpyCuda(alpha, p, xSloppy);
        copyCuda(*out, xSloppy); // nop when these pointers alias

        xpyCuda(*out, y); // swap these around?
        mat(r, y, *out); // here we can use x as tmp
        r2 = xmyNormCuda(*in, r);

        copyCuda(rSloppy, r); //nop when these pointers alias
        zeroCuda(xSloppy);

        // break-out check if we have reached the limit of the precision
        static int resIncrease = 0;
        if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
          k++;
          rUpdate++;
          if (++resIncrease > maxResIncrease) break;
        } else {
          resIncrease = 0;
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        rUpdate++;

        // explicitly restore the orthogonality of the gradient vector
        double rp = reDotProductCuda(rSloppy, p) / (r2);
        axpyCuda(-rp, rSloppy, p);

        beta = r2 / r2_old;
        xpayCuda(rSloppy, beta, p);

        if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(y,r).z);

        steps_since_reliable = 0;
      }

      breakdown = false;
      k++;

      printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
    }
    copyCuda(*xx, *x);		// nop when these pointers alias	//T
    copyCuda(*outT, xSloppy);	// nop when these pointers alias	//T
    copyCuda(yT, y);							//T
    xpyCuda(yT, *outT);							//T

    profileInvert.Stop(QUDA_PROFILE_COMPUTE);
    profileInvert.Start(QUDA_PROFILE_EPILOGUE);

    solverParam.secs = profileInvert.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
    solverParam.gflops += gflops;
    solverParam.iter += k - (it*solverParam.maxiter);

    if (k==solverParam.maxiter*(it+1))
      warningQuda("Exceeded maximum iterations %d", solverParam.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, *outT, yT);	//T
    solverParam.true_res = sqrt(xmyNormCuda(*inT, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    solverParam.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(*outT,r).z);
#else
    solverParam.true_res_hq = 0.0;
#endif

    printfQuda("CG: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e\n", k, sqrt(r2/b2), solverParam.true_res);

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profileInvert.Stop(QUDA_PROFILE_EPILOGUE);

			solverParam.updateInvertParam(*param);
		}

		if	(getVerbosity() >= QUDA_VERBOSE)
		{
			double nx	 = norm2(*outT);
			printfQuda	("Solution = %f\n",nx);
		}

		printfQuda	("Reconstruyo\n");		//DELETE ME!!
		fflush		(stdout);

		dirac.reconstruct(*xx, *bb, param->solution_type);

		if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
			axCuda(sqrt(nb), *xx);		// rescale the solution

		printfQuda	("Contraigo\n");		//DELETE ME!!
		fflush		(stdout);

		if	(getVerbosity() >= QUDA_VERBOSE)
        		printfQuda	("Contracting source\n");

		profileContract.Start(QUDA_PROFILE_TOTAL);
		profileContract.Start(QUDA_PROFILE_COMPUTE);

		dim3		blockTwust(32, 1, 1);
		dim3		blockTwost(512, 1, 1);

		DiracParam	dWParam;

		dWParam.type		 = QUDA_WILSON_DIRAC;
		dWParam.matpcType	 = QUDA_MATPC_EVEN_EVEN;
		dWParam.dagger		 = QUDA_DAG_NO;
		dWParam.gauge		 = gaugePrecise;
		dWParam.kappa		 = param->kappa;
		dWParam.mass		 = 1./(2.*param->kappa) - 4.;
		dWParam.m5		 = 0.;
		dWParam.mu		 = 0.;
//		dWParam.verbose		 = param->verbosity;

		for	(int i=0; i<4; i++)
        		dWParam.commDim[i]	 = 1;   // comms are always on

		DiracWilson	*dW	 = new DiracWilson(dWParam);

		dW->M(*tmp4, *xx);

		delete	dW;

		gamma5Cuda	(&(tmp3->Even()), &(tmp4->Even()), blockTwost);
		gamma5Cuda	(&(tmp3->Odd()),  &(tmp4->Odd()),  blockTwost);

		int	LX[4]	 = {X[0], X[1], X[2], X[3]};

		long int	sizeBuffer;

		if	(x->Precision() == QUDA_SINGLE_PRECISION)
		{
			sizeBuffer		= sizeof(float)*32*X[0]*X[1]*X[2]*X[3];

			contractGamma5Cuda	(xx->Even(), tmp3->Even(), ((float2 *) ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  tmp3->Odd(),  ((float2 *) ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

			cudaMemcpy	(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				tempVecD[2*ix][0][it]	= 0.;
				tempVecD[2*ix+1][0][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][0][it]	+=  ((float*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  + ((float*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
						     		    ((float*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] - ((float*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
					tempVecD[2*ix+1][0][it]	+= -((float*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((float*)h_ctrn)[8*tV+2*ix*sV+2*iv] +
								    ((float*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((float*)h_ctrn)[28*tV+2*ix*sV+2*iv];
				}

				((float*) cnRes_gv[0][it])[2*ix]	+=  tempVecD[2*ix][0][it];
				((float*) cnRes_gv[0][it])[2*ix+1]	+=  tempVecD[2*ix+1][0][it];
				((float*) cnRs2_gv[0][it])[2*ix]	+=  tempVecD[2*ix][0][it]   * tempVecD[2*ix][0][it];
				((float*) cnRs2_gv[0][it])[2*ix+1]	+=  tempVecD[2*ix+1][0][it] * tempVecD[2*ix+1][0][it];

				tempVecD[2*ix][1][it]	= 0.;
				tempVecD[2*ix+1][1][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][1][it]	+=  ((float*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((float*)h_ctrn)[8*tV+2*ix*sV+2*iv] -
							     	    ((float*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((float*)h_ctrn)[28*tV+2*ix*sV+2*iv];
					tempVecD[2*ix+1][1][it]	+=  ((float*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  - ((float*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
								    ((float*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] + ((float*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
				}

				((float*) cnRes_gv[1][it])[2*ix]	+=  tempVecD[2*ix][1][it];
				((float*) cnRes_gv[1][it])[2*ix+1]	+=  tempVecD[2*ix+1][1][it];
				((float*) cnRs2_gv[1][it])[2*ix]	+=  tempVecD[2*ix][1][it]   * tempVecD[2*ix][1][it];
				((float*) cnRs2_gv[1][it])[2*ix+1]	+=  tempVecD[2*ix+1][1][it] * tempVecD[2*ix+1][1][it];

				tempVecD[2*ix][2][it]	= 0.;
				tempVecD[2*ix+1][2][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][2][it]	+=  ((float*)h_ctrn)[2*ix*sV+2*iv+1]	    - ((float*)h_ctrn)[10*tV+2*ix*sV+2*iv+1] -
						     		    ((float*)h_ctrn)[20*tV+2*ix*sV+2*iv+1] + ((float*)h_ctrn)[30*tV+2*ix*sV+2*iv+1];
					tempVecD[2*ix+1][2][it]	+= -((float*)h_ctrn)[2*ix*sV+2*iv]	    + ((float*)h_ctrn)[10*tV+2*ix*sV+2*iv] +
								    ((float*)h_ctrn)[20*tV+2*ix*sV+2*iv]   - ((float*)h_ctrn)[30*tV+2*ix*sV+2*iv];
				}

				((float*) cnRes_gv[2][it])[2*ix]	+=  tempVecD[2*ix][2][it];
				((float*) cnRes_gv[2][it])[2*ix+1]	+=  tempVecD[2*ix+1][2][it];
				((float*) cnRs2_gv[2][it])[2*ix]	+=  tempVecD[2*ix][2][it]   * tempVecD[2*ix][2][it];
				((float*) cnRs2_gv[2][it])[2*ix+1]	+=  tempVecD[2*ix+1][2][it] * tempVecD[2*ix+1][2][it];
			}

			contractGamma5Cuda	(xx->Even(), xx->Even(), ((float2 *) ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  xx->Odd(),  ((float2 *) ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

			cudaMemcpy	(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				tempData[2*ix][it]	= 0.;
				tempData[2*ix+1][it]	= 0.;

				for	(int iv=0; iv<sV;iv++)//sV; iv++)
				{
					tempData[2*ix][it]	+=  ((float*)h_ctrn)[4*tV+2*ix*sV+2*iv]    + ((float*)h_ctrn)[14*tV+2*ix*sV+2*iv] +
								    ((float*)h_ctrn)[16*tV+2*ix*sV+2*iv]   + ((float*)h_ctrn)[26*tV+2*ix*sV+2*iv];
					tempData[2*ix+1][it]	+=  ((float*)h_ctrn)[4*tV+2*ix*sV+2*iv+1]  + ((float*)h_ctrn)[14*tV+2*ix*sV+2*iv+1] +
								    ((float*)h_ctrn)[16*tV+2*ix*sV+2*iv+1] + ((float*)h_ctrn)[26*tV+2*ix*sV+2*iv+1];
				}

				((float*) cnRes_vv[it])[2*ix]	+=  tempData[2*ix][it];
				((float*) cnRes_vv[it])[2*ix+1]	+=  tempData[2*ix+1][it];
				((float*) cnRs2_vv[it])[2*ix]	+=  tempData[2*ix][it]   * tempData[2*ix][it];
				((float*) cnRs2_vv[it])[2*ix+1]	+=  tempData[2*ix+1][it] * tempData[2*ix+1][it];
			}
		}	else if	(x->Precision() == QUDA_DOUBLE_PRECISION) {
			sizeBuffer		= sizeof(double)*32*X[0]*X[1]*X[2]*X[3];

			contractGamma5Cuda	(xx->Even(), tmp3->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				tempVecD[2*ix][0][it]	= 0.;
				tempVecD[2*ix+1][0][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][0][it]	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  + ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
						     		    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] - ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
					tempVecD[2*ix+1][0][it]	+= -((double*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv] +
								    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv];
				}

				((double*) cnRes_gv[0][it])[2*ix]	+=  tempVecD[2*ix][0][it];
				((double*) cnRes_gv[0][it])[2*ix+1]	+=  tempVecD[2*ix+1][0][it];
				((double*) cnRs2_gv[0][it])[2*ix]	+=  tempVecD[2*ix][0][it]   * tempVecD[2*ix][0][it];
				((double*) cnRs2_gv[0][it])[2*ix+1]	+=  tempVecD[2*ix+1][0][it] * tempVecD[2*ix+1][0][it];

				tempVecD[2*ix][1][it]	= 0.;
				tempVecD[2*ix+1][1][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][1][it]	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv] -
							     	    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv];
					tempVecD[2*ix+1][1][it]	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
								    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
				}

				((double*) cnRes_gv[1][it])[2*ix]	+=  tempVecD[2*ix][1][it];
				((double*) cnRes_gv[1][it])[2*ix+1]	+=  tempVecD[2*ix+1][1][it];
				((double*) cnRs2_gv[1][it])[2*ix]	+=  tempVecD[2*ix][1][it]   * tempVecD[2*ix][1][it];
				((double*) cnRs2_gv[1][it])[2*ix+1]	+=  tempVecD[2*ix+1][1][it] * tempVecD[2*ix+1][1][it];

				tempVecD[2*ix][2][it]	= 0.;
				tempVecD[2*ix+1][2][it]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempVecD[2*ix][2][it]	+=  ((double*)h_ctrn)[2*ix*sV+2*iv+1]	    - ((double*)h_ctrn)[10*tV+2*ix*sV+2*iv+1] -
						     		    ((double*)h_ctrn)[20*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[30*tV+2*ix*sV+2*iv+1];
					tempVecD[2*ix+1][2][it]	+= -((double*)h_ctrn)[2*ix*sV+2*iv]	    + ((double*)h_ctrn)[10*tV+2*ix*sV+2*iv] +
								    ((double*)h_ctrn)[20*tV+2*ix*sV+2*iv]   - ((double*)h_ctrn)[30*tV+2*ix*sV+2*iv];
				}

				((double*) cnRes_gv[2][it])[2*ix]	+=  tempVecD[2*ix][2][it];
				((double*) cnRes_gv[2][it])[2*ix+1]	+=  tempVecD[2*ix+1][2][it];
				((double*) cnRs2_gv[2][it])[2*ix]	+=  tempVecD[2*ix][2][it]   * tempVecD[2*ix][2][it];
				((double*) cnRs2_gv[2][it])[2*ix+1]	+=  tempVecD[2*ix+1][2][it] * tempVecD[2*ix+1][2][it];
			}

			contractGamma5Cuda	(xx->Even(), xx->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  xx->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				tempData[2*ix][it]	= 0.;
				tempData[2*ix+1][it]	= 0.;

				for	(int iv=0; iv<sV;iv++)//sV; iv++)
				{
					tempData[2*ix][it]	+=  ((double*)h_ctrn)[4*tV+2*ix*sV+2*iv]    + ((double*)h_ctrn)[14*tV+2*ix*sV+2*iv] +
								    ((double*)h_ctrn)[16*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[26*tV+2*ix*sV+2*iv];
					tempData[2*ix+1][it]	+=  ((double*)h_ctrn)[4*tV+2*ix*sV+2*iv+1]  + ((double*)h_ctrn)[14*tV+2*ix*sV+2*iv+1] +
								    ((double*)h_ctrn)[16*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[26*tV+2*ix*sV+2*iv+1];
				}

				((double*) cnRes_vv[it])[2*ix]		+=  tempData[2*ix][it];
				((double*) cnRes_vv[it])[2*ix+1]	+=  tempData[2*ix+1][it];
				((double*) cnRs2_vv[it])[2*ix]		+=  tempData[2*ix][it]   * tempData[2*ix][it];
				((double*) cnRs2_vv[it])[2*ix+1]	+=  tempData[2*ix+1][it] * tempData[2*ix+1][it];
			}

			printfQuda	("Locals contracted\n");
			fflush		(stdout);
		}

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printfQuda	("Error synchronizing!!!\n");
       			return;
		}
	//TERMINA

		profileContract.Stop(QUDA_PROFILE_COMPUTE);
		profileContract.Stop(QUDA_PROFILE_TOTAL);
	}

	if	(Cr == true)
	{
		int	oldMaxIter	= param->maxiter;
		param->maxiter		= 40000;

    profileInvert.Start(QUDA_PROFILE_COMPUTE);
    int k=nSteps*oldMaxIter;

    printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));

    while ( (r2 > stop) &&
            k < solverParam.maxiter) {
      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp

      double sigma;

      bool breakdown = false;

      if (solverParam.pipeline) {
        double3 triplet = tripleCGReductionCuda(rSloppy, Ap, p);
        r2 = triplet.x; double Ap2 = triplet.y; pAp = triplet.z;
        r2_old = r2;

        alpha = r2 / pAp;
        sigma = alpha*(alpha * Ap2 - pAp);
        if (sigma < 0.0 || steps_since_reliable==0) { // sigma condition has broken down
          r2 = axpyNormCuda(-alpha, Ap, rSloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;
        pAp = reDotProductCuda(p, Ap);
        alpha = r2 / pAp;

        // here we are deploying the alternative beta computation 
        Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy);
        r2 = real(cg_norm); // (r_new, r_new)
        sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( (r2 < stop) && delta >= solverParam.tol) updateX = 1;

      if ( !(updateR || updateX)) {                     //Fix Update reliable
        //beta = r2 / r2_old;
        beta = sigma / r2_old; // use the alternative beta computation

        if (solverParam.pipeline && !breakdown) tripleCGUpdateCuda(alpha, beta, Ap, xSloppy, rSloppy, p);
        else axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);
        if (use_heavy_quark_res && k%heavy_quark_check==0) {
          copyCuda(tmp,y);
          heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
        }

        steps_since_reliable++;
      } else {
        axpyCuda(alpha, p, xSloppy);
        copyCuda(*out, xSloppy); // nop when these pointers alias

        xpyCuda(*out, y); // swap these around?
        mat(r, y, *out); // here we can use x as tmp
        r2 = xmyNormCuda(*in, r);

        copyCuda(rSloppy, r); //nop when these pointers alias
        zeroCuda(xSloppy);

        // break-out check if we have reached the limit of the precision
        static int resIncrease = 0;
        if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
          k++;
          rUpdate++;
          if (++resIncrease > maxResIncrease) break;
        } else {
          resIncrease = 0;
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        rUpdate++;

        // explicitly restore the orthogonality of the gradient vector
        double rp = reDotProductCuda(rSloppy, p) / (r2);
        axpyCuda(-rp, rSloppy, p);

        beta = r2 / r2_old;
        xpayCuda(rSloppy, beta, p);

        if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(y,r).z);

        steps_since_reliable = 0;
      }

      breakdown = false;
      k++;

      printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
    }
    copyCuda(*xx, *x);		// nop when these pointers alias	//T
    copyCuda(*outT, xSloppy);	// nop when these pointers alias	//T
    copyCuda(yT, y);							//T
    xpyCuda(yT, *outT);							//T

    profileInvert.Stop(QUDA_PROFILE_COMPUTE);
    profileInvert.Start(QUDA_PROFILE_EPILOGUE);

    solverParam.secs = profileInvert.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
    solverParam.gflops += gflops;
    solverParam.iter += k - oldMaxIter*nSteps;

    if (k==solverParam.maxiter)
      warningQuda("Exceeded maximum iterations %d", solverParam.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, *outT, yT);
    solverParam.true_res = sqrt(xmyNormCuda(*inT, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    solverParam.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(*outT,r).z);
#else
    solverParam.true_res_hq = 0.0;
#endif

    printfQuda("CG: Convergence at %d iterations, L2 relative residual: iterated = %e, true = %e\n", k, sqrt(r2/b2), solverParam.true_res);

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profileInvert.Stop(QUDA_PROFILE_EPILOGUE);

/*
		if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
			axCuda(1.0/sqrt(nb), *x);

		dirac.prepare	(in, out, *x, *b, param->solution_type);

		if	(getVerbosity() >= QUDA_VERBOSE)
		{
			double nin	 = norm2(*in);
			double nout	 = norm2(*out);
			printfQuda	("Prepared source   = %f\n", nin);
			printfQuda	("Prepared solution = %f\n", nout);
		}

		massRescale	(param->dslash_type, param->kappa, param->mass, param->solution_type, param->mass_normalization, *in);

		if	(getVerbosity() >= QUDA_VERBOSE)
		{
			double nin	 = norm2(*in);
			printfQuda	("Prepared source post mass rescale = %f\n", nin);
		}

		if	(pc_solution && !pc_solve)
			errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");

		if	(!mat_solution && !pc_solution && pc_solve)
			errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");

		if	(mat_solution && !direct_solve)
		{						// prepare source: b' = A^dag b
			cudaColorSpinorField tmp(*in);
			dirac.Mdag(*in, tmp);
		}
		else if	(!mat_solution && direct_solve)
		{						// perform the first of two solves: A^dag y = b
			DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
			SolverParam solverParam(*param);
			Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
			(*solve)(*out, *in);
			copyCuda(*in, *out);
			solverParam.updateInvertParam(*param);
			delete solve;
		}

		if (direct_solve)
		{
			DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
			SolverParam solverParam(*param);
			Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
			(*solve)(*out, *in);
			solverParam.updateInvertParam(*param);
			delete solve;
		}
		else
		{
			DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
			SolverParam solverParam(*param);
			Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
			(*solve)(*out, *in);
			solverParam.updateInvertParam(*param);
			delete solve;
		}

		if	(getVerbosity() >= QUDA_VERBOSE)
		{
			double nx	 = norm2(*x);
			printfQuda	("Solution = %f\n",nx);
		}
*/
		solverParam.updateInvertParam(*param);

		if	(getVerbosity() >= QUDA_VERBOSE)
		{
			double nx	= norm2(*outT);
			printfQuda	("Solution = %f\n",nx);
		}

		dirac.reconstruct(*xx, *bb, param->solution_type);

		if	(param->solver_normalization == QUDA_SOURCE_NORMALIZATION)
			axCuda(sqrt(nb), *xx);		// rescale the solution

//		tmp3		 = new cudaColorSpinorField(cudaParam);
//		tmp4		 = new cudaColorSpinorField(cudaParam);

		if	(getVerbosity() >= QUDA_VERBOSE)
        		printfQuda	("Contracting source\n");

		profileContract.Start(QUDA_PROFILE_TOTAL);
		profileContract.Start(QUDA_PROFILE_COMPUTE);

		dim3		blockTwust(32, 1, 1);
		dim3		blockTwost(512, 1, 1);

		DiracParam	dWParam;

		dWParam.type		 = QUDA_WILSON_DIRAC;
		dWParam.matpcType	 = QUDA_MATPC_EVEN_EVEN;
		dWParam.dagger		 = QUDA_DAG_NO;
		dWParam.gauge		 = gaugePrecise;
		dWParam.kappa		 = param->kappa;
		dWParam.mass		 = 1./(2.*param->kappa) - 4.;
		dWParam.m5		 = 0.;
		dWParam.mu		 = 0.;
//		dWParam.verbose		 = param->verbosity;

		for	(int i=0; i<4; i++)
        		dWParam.commDim[i]	 = 1;   // comms are always on

		DiracWilson	*dW	 = new DiracWilson(dWParam);

		dW->M(*tmp4, *xx);

		delete	dW;

		gamma5Cuda	(&(tmp3->Even()), &(tmp4->Even()), blockTwost);
		gamma5Cuda	(&(tmp3->Odd()),  &(tmp4->Odd()),  blockTwost);

		int	LX[4]	 = {X[0], X[1], X[2], X[3]};

		long int	sizeBuffer;

		if	(x->Precision() == QUDA_SINGLE_PRECISION)
		{
			sizeBuffer	= sizeof(float)*32*X[0]*X[1]*X[2]*X[3];
/*
			contractGamma5Cuda	(x->Even(), tmp3->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(x->Odd(),  tmp3->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((float*) cnRes_gv)[ix]	+= ((float*)h_ctrn)[ix];

			contractGamma5Cuda	(x->Even(), x->Even(), ((float2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(x->Odd(),  x->Odd(),  ((float2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(float)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			float		tempData[32*X[3]];
			const int	sV = X[0]*X[1]*X[2];

			for	(int ix=0; ix<16*X[3]; ix++)
			{
				tempData[2*ix]		= 0.;
				tempData[2*ix+1]	= 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					tempData[2*ix]		+= ((float*)h_ctrn)[2*ix*sV + 2*iv];
					tempData[2*ix + 1]	+= ((float*)h_ctrn)[2*ix*sV + 2*iv + 1];
				}

				((float *) cnRes_vv[it])[2*ix]		+=  tempData[2*ix];
				((float *) cnRes_vv[it])[2*ix+1]	+=  tempData[2*ix+1];
//				((float *) cnRs2_vv[it])[2*ix]		+=  tempData[2*ix]  *tempData[2*ix];
//				((float *) cnRs2_vv[it])[2*ix+1]	+=  tempData[2*ix+1]*tempData[2*ix+1];
			}

			printfQuda	("Locals contracted\n");
			fflush		(stdout);*/
		}
		else if	(x->Precision() == QUDA_DOUBLE_PRECISION)
		{
			sizeBuffer	= sizeof(double)*32*X[0]*X[1]*X[2]*X[3];

			contractGamma5Cuda	(xx->Even(), tmp3->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				double	crR = 0., crI = 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					crR	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  + ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
						    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] - ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
					crI	+= -((double*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv] +
						    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv];
				}

				for	(int it=0; it<nSteps; it++)				
				{
					((double*) cnCor_gv[0][it])[2*ix]	+=  (crR - tempVecD[2*ix][0][it]);
					((double*) cnCor_gv[0][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][0][it]);
					((double*) cnCr2_gv[0][it])[2*ix]	+=  (crR - tempVecD[2*ix][0][it])   * (crR - tempVecD[2*ix][0][it]);
					((double*) cnCr2_gv[0][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][0][it]) * (crI - tempVecD[2*ix+1][0][it]);
				}

				crR = crI = 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					crR	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv]    - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv] -
					     	    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv];
					crI	+=  ((double*)h_ctrn)[2*tV+2*ix*sV+2*iv+1]  - ((double*)h_ctrn)[8*tV+2*ix*sV+2*iv+1] -
						    ((double*)h_ctrn)[22*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[28*tV+2*ix*sV+2*iv+1];
				}

				for	(int it=0; it<nSteps; it++)				
				{
					((double*) cnCor_gv[1][it])[2*ix]	+=  (crR - tempVecD[2*ix][1][it]);
					((double*) cnCor_gv[1][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][1][it]);
					((double*) cnCr2_gv[1][it])[2*ix]	+=  (crR - tempVecD[2*ix][1][it])   * (crR - tempVecD[2*ix][1][it]);
					((double*) cnCr2_gv[1][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][1][it]) * (crI - tempVecD[2*ix+1][1][it]);
				}

				crR = crI = 0.;

				for	(int iv=0; iv<sV; iv++)
				{
					crR	+=  ((double*)h_ctrn)[2*ix*sV+2*iv+1]	    - ((double*)h_ctrn)[10*tV+2*ix*sV+2*iv+1] -
						    ((double*)h_ctrn)[20*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[30*tV+2*ix*sV+2*iv+1];
					crI	+= -((double*)h_ctrn)[2*ix*sV+2*iv]	    + ((double*)h_ctrn)[10*tV+2*ix*sV+2*iv] +
						    ((double*)h_ctrn)[20*tV+2*ix*sV+2*iv]   - ((double*)h_ctrn)[30*tV+2*ix*sV+2*iv];
				}

				for	(int it=0; it<nSteps; it++)				
				{
					((double*) cnCor_gv[2][it])[2*ix]	+=  (crR - tempVecD[2*ix][2][it]);
					((double*) cnCor_gv[2][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][2][it]);
					((double*) cnCr2_gv[2][it])[2*ix]	+=  (crR - tempVecD[2*ix][2][it])   * (crR - tempVecD[2*ix][2][it]);
					((double*) cnCr2_gv[2][it])[2*ix+1]	+=  (crI - tempVecD[2*ix+1][2][it]) * (crI - tempVecD[2*ix+1][2][it]);
				}
			}

			contractGamma5Cuda	(xx->Even(), xx->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
			contractGamma5Cuda	(xx->Odd(),  xx->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

			cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<X[3]; ix++)
			{
				double	crR = 0., crI = 0.;

				for	(int iv=0; iv<sV;iv++)
				{
					crR	+=  ((double*)h_ctrn)[4*tV+2*ix*sV+2*iv]    + ((double*)h_ctrn)[14*tV+2*ix*sV+2*iv] +
						    ((double*)h_ctrn)[16*tV+2*ix*sV+2*iv]   + ((double*)h_ctrn)[26*tV+2*ix*sV+2*iv];
					crI	+=  ((double*)h_ctrn)[4*tV+2*ix*sV+2*iv+1]  + ((double*)h_ctrn)[14*tV+2*ix*sV+2*iv+1] +
						    ((double*)h_ctrn)[16*tV+2*ix*sV+2*iv+1] + ((double*)h_ctrn)[26*tV+2*ix*sV+2*iv+1];
				}

				for	(int it=0; it<nSteps; it++)				
				{
					((double*) cnCor_vv[it])[2*ix]		+=  (crR - tempData[2*ix][it]);
					((double*) cnCor_vv[it])[2*ix+1]	+=  (crI - tempData[2*ix+1][it]);
					((double*) cnCr2_vv[it])[2*ix]		+=  (crR - tempData[2*ix][it])   * (crR - tempData[2*ix][it]);
					((double*) cnCr2_vv[it])[2*ix+1]	+=  (crI - tempData[2*ix+1][it]) * (crI - tempData[2*ix+1][it]);
				}
			}
/*
			cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

			for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
				((double *) cnRs2_vv)[ix]       += ((double*)h_ctrn)[ix];

			doCudaFFT(cnRs2_vv, X[0], X[1], X[2], X[3]);

*/
			printfQuda	("Locals contracted\n");
			fflush		(stdout);
		}

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printfQuda	("Error synchronizing!!!\n");
       			return;
		}
	//TERMINA

		param->maxiter		= oldMaxIter;

		profileContract.Stop(QUDA_PROFILE_COMPUTE);
		profileContract.Stop(QUDA_PROFILE_TOTAL);
	}

	delete  tmp3;
	delete  tmp4;

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *xx;
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*xx);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	if (&tmp2 != &tmp) delete tmp2_p;

	if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
	if (xSloppy.Precision() != out->Precision()) delete x_sloppy;

	delete	x;
	delete	xx;
	delete	h_x;

	delete  h_b;
	delete  bb;
	delete  b;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	popVerbosity();

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());
/*
	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		cudaCloverField *tempC		 = cloverSloppy;
		cloverSloppy			 = cloverPrecondition;
		cloverPrecondition		 = tempC;

		tempC				 = cloverInvSloppy;
		cloverInvSloppy			 = cloverInvPrecondition;
		cloverInvPrecondition		 = tempC;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;

		tempPrec				 = param->clover_cuda_prec_sloppy;
		param->clover_cuda_prec_sloppy		 = param->clover_cuda_prec_precondition;
		param->clover_cuda_prec_precondition	 = tempPrec;
	}
*/
	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

/*
void	doCudaFFT	(void *cnRes_vv, int xdim, int ydim, int zdim, int tdim)
{
        static cufftHandle      fftPlan;
        static int              init = 0;
        int                     nRank[3]         = {xdim, ydim, zdim};
        int               Vol              = xdim*ydim*zdim;

        static cudaStream_t     streamCuFFT;

        cudaStreamCreate        (&streamCuFFT);

        if      (cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*tdim) != CUFFT_SUCCESS)
        {
                printf  ("Error in the FFT!!!\n");
        }

        cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_NATIVE);
        cufftSetStream                  (fftPlan, streamCuFFT);

        printfQuda      ("Synchronizing\n");

        if      (cudaDeviceSynchronize() != cudaSuccess)
        {
                printf  ("Error synchronizing!!!\n");
        }

        printfQuda      ("CuFFT plan already initialized\n");

        void    *ctrnS;

        if      ((cudaMalloc(&ctrnS, sizeof(double)*32*Vol*tdim)) == cudaErrorMemoryAllocation)
        {
                printf  ("Error allocating memory for contraction results in GPU.\n");
                exit    (0);
        }
        cudaMemcpy      (ctrnS, cnRes_vv, sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
   
        if      (cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
        {
                printf  ("Error executing FFT!!!\n");
        }
   
        if      (cudaDeviceSynchronize() != cudaSuccess)
        {
                printf  ("Error synchronizing!!!\n");
        }
   
        cudaMemcpy      (cnRes_vv, ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	cufftDestroy            (fftPlan);
	cudaStreamDestroy       (streamCuFFT);
}
*/

