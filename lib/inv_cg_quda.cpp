#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>

#include <iostream>

#ifdef BLOCKSOLVER
#include <Eigen/Dense>
#include "nvToolsExt.h"

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();

#else

#define PUSH_RANGE(name,cid)
#define PIP_RANFE

#endif


namespace quda {
  CG::CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), init(false) {
  }

  CG::~CG() {
    if ( init ) {
      delete rp;
      delete yp;
      delete App;
      delete tmpp;
      init = false;
    }
  }

  void CG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = blas::norm2(b);
    if (b2 == 0) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;

    mat(r, x, y);
    double r2 = blas::xmyNorm(b, r);
    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
    }

    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
        !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p =
      (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
      ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField* pp = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &p = *pp;

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    bool heavy_quark_restart = false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double r2_old;

    double stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

    double heavy_quark_res = 0.0;  // heavy quark res idual
    double heavy_quark_res_old = 0.0;  // heavy quark residual

    if (use_heavy_quark_res) {
      heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
      heavy_quark_res_old = heavy_quark_res;   // heavy quark residual
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double alpha = 0.0;
    double beta = 0.0;
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    // 0 means we have no tolerance
    // maybe we should expose this as a parameter
    const int hqmaxresIncrease = maxResIncrease + 1;

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;

    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    int k = 0;

    PrintStats("CG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;
    bool converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

    while ( !converged && k < param.maxiter ) {
      PUSH_RANGE("Dslash",1)
      matSloppy(Ap, p, tmp, tmp2);  // tmp as tmp
POP_RANGE
PUSH_RANGE("LinearAlgebra",2)
      double sigma;

      bool breakdown = false;
      if (param.pipeline) {
        double3 triplet = blas::tripleCGReduction(rSloppy, Ap, p);
        r2 = triplet.x; double Ap2 = triplet.y; pAp = triplet.z;
        r2_old = r2;
        alpha = r2 / pAp;
        sigma = alpha*(alpha * Ap2 - pAp);
        if (sigma < 0.0 || steps_since_reliable == 0) { // sigma condition has broken down
          r2 = blas::axpyNorm(-alpha, Ap, rSloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;
        pAp = blas::reDotProduct(p, Ap);
        alpha = r2 / pAp;

        // here we are deploying the alternative beta computation
        Complex cg_norm = blas::axpyCGNorm(-alpha, Ap, rSloppy);
        r2 = real(cg_norm);  // (r_new, r_new)
        sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2;  // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol ) updateX = 1;

      // For heavy-quark inversion force a reliable update if we continue after
      if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq) and param.delta >= param.tol ) {
        updateX = 1;
      }

      if ( !(updateR || updateX )) {
        beta = sigma / r2_old;  // use the alternative beta computation

        if (param.pipeline && !breakdown)
          blas::tripleCGUpdate(alpha, beta, Ap, rSloppy, xSloppy, p);
        else
          blas::axpyZpbx(alpha, p, xSloppy, rSloppy, beta);


        if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
          if (&x != &xSloppy) {
            blas::copy(tmp, y);
            heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
          } else {
            blas::copy(r, rSloppy);
            heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
          }
        }

        steps_since_reliable++;
      } else {

        printfQuda("\t * Reliable update \n");
        blas::axpy(alpha, p, xSloppy);
        blas::copy(x, xSloppy); // nop when these pointers alias

        blas::xpy(x, y); // swap these around?
        mat(r, y, x, tmp3); //  here we can use x as tmp
        r2 = blas::xmyNorm(b, r);

        blas::copy(rSloppy, r); //nop when these pointers alias
        blas::zero(xSloppy);

        // calculate new reliable HQ resididual
        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2), r0Norm, resIncreaseTotal);
          if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("CG: solver exiting due to too many true residual norm increases");
              break;
            }
          }
        } else {
          resIncrease = 0;
        }
        // if L2 broke down already we turn off reliable updates and restart the CG
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res > heavy_quark_res_old) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res, heavy_quark_res_old);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }

        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;
        rUpdate++;

        if (use_heavy_quark_res and heavy_quark_restart) {
          // perform a restart
          blas::copy(p, rSloppy);
          heavy_quark_restart = false;
        } else {
          // explicitly restore the orthogonality of the gradient vector
          double rp = blas::reDotProduct(rSloppy, p) / (r2);
          blas::axpy(-rp, rSloppy, p);

          beta = r2 / r2_old;
          blas::xpay(rSloppy, beta, p);
        }


        steps_since_reliable = 0;
        heavy_quark_res_old = heavy_quark_res;
      }

      breakdown = false;
      k++;

      PrintStats("CG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        // L2 is concverged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2, heavy_quark_res, stop, param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq);
        converged = L2done and HQdone;
      }
POP_RANGE
    }

    blas::copy(x, xSloppy);
    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, x, y, tmp3);
    param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    PrintSummary("CG", k, r2, b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
    if (xSloppy.Precision() != x.Precision()) delete x_sloppy;

    delete pp;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

  // use BlockCGrQ algortithm or BlockCG (with / without GS, see BLOCKCG_GS option)
  #define BCGRQ 1
//#define MWVERBOSE
  #if BCGRQ
  void CG::solve(ColorSpinorField& x, ColorSpinorField& b) {
    #ifndef BLOCKSOLVER
    errorQuda("QUDA_BLOCKSOLVER not built.");
    #else

    printfQuda("BCGrQ Solver\n");

    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
    errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    using Eigen::MatrixXcd;

    // Check to see that we're not trying to invert on a zero-field source
    //MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_MULTI_SHIFT];
    double b2avg=0;
    for(int i=0; i< param.num_src; i++){
      b2[i]=blas::norm2(b.Component(i));
      b2avg += b2[i];
      if(b2[i] == 0){
        profile.TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source\n");
        x=b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    b2avg = b2avg / param.num_src;

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;


    for(int i=0; i<param.num_src; i++){
      mat(r.Component(i), x.Component(i), y.Component(i));
    }

    double r2avg=0;
    MatrixXcd r2(param.num_src, param.num_src);
    for(int i=0; i<param.num_src; i++){
      for(int j=i; j < param.num_src; j++){
        r2(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
        if (i!=j) r2(j,i) = std::conj(r2(i,j));
        if (i==j) {
          r2avg += r2(i,i).real();
          printfQuda("r2[%i] %e\n", i, r2(i,i).real());
        }
      }
    }


    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      // will that work ?
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
      for(int i=0; i<param.num_src; i++){
        blas::copy(r_sloppy->Component(i), r.Component(i)); //nop when these pointers alias
      }
    }


    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
    !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p =
    (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField* pp = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &p = *pp;
    ColorSpinorField* rpnew = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &rnew = *rpnew;

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res =
    (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;


    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    MatrixXcd r2_old(param.num_src, param.num_src);
    double stop[QUDA_MAX_MULTI_SHIFT];


    double heavy_quark_res[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark res idual
    double heavy_quark_res_old[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark residual



    for(int i = 0; i < param.num_src; i++){
      stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
      if (use_heavy_quark_res) {
        heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
        heavy_quark_res_old[i] = heavy_quark_res[i];   // heavy quark residual
      }
    }

// FIXME heavy quark
//    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual
//     bool heavy_quark_restart = false;

    MatrixXcd alpha = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd beta = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd gamma = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd C = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd S = MatrixXcd::Identity(param.num_src,param.num_src);
    MatrixXcd pTp =  MatrixXcd::Identity(param.num_src,param.num_src);
    MatrixXcd pAp = MatrixXcd::Identity(param.num_src,param.num_src);
    //  gamma = gamma * 2.0;


    int rUpdate = 0;

    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }


//FIXME:reliable updates currently not implemented
/**
    double delta = param.delta;//MW: hack no reliable updates param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    // 0 means we have no tolerance
    // maybe we should expose this as a parameter
    const int hqmaxresIncrease = maxResIncrease + 1;

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;
*/
    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    int k = 0;



    PrintStats("CG", k, r2avg / param.num_src, b2avg, heavy_quark_res[0]);
    int steps_since_reliable = 1;
    bool allconverged = true;
    bool converged[QUDA_MAX_MULTI_SHIFT];
    for(int i=0; i<param.num_src; i++){
      converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }
    // MatrixXcd sigma(param.num_src,param.num_src);


    C = MatrixXcd::Zero(param.num_src,param.num_src);

    for(int i = 0; i<param.num_src; i++){
      for(int j = i; j < param.num_src; j++){
        r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
        if (i !=j ) r2(j,i) = std::conj(r2(i,j));
      }
    }

    // Eigen::LLT<MatrixXcd> lltOfA(r2);  // compute the Cholesky decomposition of A
    MatrixXcd L = r2.llt().matrixL();  // retrieve factor L  in the decomposition
    C = L.adjoint();
    MatrixXcd Linv = C.inverse();
// r2.llt().matrixL() should do as well
#ifdef MWVERBOSE
    std::cout << "r2\n " << r2 << std::endl;
    std::cout << "L\n " << L.adjoint() << std::endl;
#endif
    for(int i=0; i<param.num_src; i++){
      blas::zero(p.Component(i));
      // blas::cax(Linv(i,0),p.Component(i));

      for(int j=0;j<param.num_src; j++){
        // nsrc * nsrc (write rnew) + nsrc * nsrc (read rnew) + nsrc * nsrc (read p)
        blas::caxpy(Linv(j,i),r.Component(j),p.Component(i));
        // if we can do something like rnew_i(x) += sum_j S(j,i)^h p_j(x) we get away with
        // nsrc * read rnew + nsrc * write rnew + nsrc*nsrc read p_j = (nsrc+nsrc + nsrc^2)
        // maybe we can also use i as blockIdx.y and by that explit caching of the p_j(x) Components
      }
    }
    // L.adjoint() is equal to C.
    // C is <pnew, pold>
    //std::cout << "L\n " << L.inverse() << std::endl;

    // QR decomposition -- in the first iteration we can do the QR decomposition on P
    // for(int i=0; i < param.num_src; i++){
    //   n = blas::norm2(p.Component(i));
    //   blas::ax(1/sqrt(n),p.Component(i));
    //   for(int j=i+1; j < param.num_src; j++) {
    //     ri=blas::cDotProduct(p.Component(i),p.Component(j));
    //     blas::caxpy(-ri,p.Component(i),p.Component(j));
    //   }
    // }


    // C = MatrixXcd::Zero(param.num_src,param.num_src);
    // for ( int i = 0; i < param.num_src; i++){
    //   for (int j=i; j < param.num_src; j++){
    //     C(i,j) = blas::cDotProduct(p.Component(i),r.Component(j));
    //   }
    // }


    // set r to QR decompoistion of r
    for(int i=0; i< param.num_src; i++){
      blas::copy(rSloppy.Component(i), p.Component(i));
    }

    #ifdef MWVERBOSE
    for(int i=0; i<param.num_src; i++){
      for(int j=0; j<param.num_src; j++){
        pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
      }
    }
    std::cout << " pTp  " << std::endl << pTp << std::endl;
    std::cout << " L " << std::endl << L.adjoint() << std::endl;
    std::cout << " C " << std::endl << C << std::endl;
    #endif

    while ( !allconverged && k < param.maxiter ) {
      PUSH_RANGE("Dslash",1)
      for(int i=0; i<param.num_src; i++){
        matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
      }
      POP_RANGE

      PUSH_RANGE("LinearAlgebra",2)
      bool breakdown = false;
      // FIXME: need to check breakdown
      for(int i=0; i<param.num_src; i++){
        for(int j=i; j < param.num_src; j++){
          pAp(i,j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
          if (i!=j) pAp(j,i) = std::conj(pAp(i,j));
        }
      }

      alpha = pAp.inverse() * C;
      #ifdef MWVERBOSE
      std::cout << "pAp\n" << pAp << std::endl;
      std::cout << "alpha\n" << alpha << std::endl;
      #endif
      // update X

      for(int i = 0; i < param.num_src; i++){
        // for(int j = 0; j < param.num_src; j++){
        //   blas::caxpy(alpha(j,i),  p.Component(j),xSloppy.Component(i));
        // }
        // this is just a workaround to reduce LA overhead somewhat
        const int j3 = param.num_src/3;
        const int j2 = ((param.num_src%3)/2);
        const int j1 = ((param.num_src%3)%2);
        for (int j=0;j<j3;j++){
          blas::caxpbypczpw(alpha(3*j,i),p.Component(3*j),alpha(3*j+1,i),p.Component(3*j+1),alpha(3*j+2,i),p.Component(3*j+2),xSloppy.Component(i));
        }
        for (int j=0;j<j2;j++){
          blas::caxpbypz(alpha(3*j3+2*j,i),p.Component(3*j3+2*j),alpha(3*j3+2*j+1,i),p.Component(3*j3+2*j+1),xSloppy.Component(i));
        }
        for (int j=0; j<j1;j++){
          blas::caxpy(alpha(3*j3+2*j2+j,i),p.Component(3*j3+2*j2+j),xSloppy.Component(i));
        }
      }

      beta = pAp.inverse();
      // here we are deploying the alternative beta computation
      for(int i=0; i<param.num_src; i++){
        // for(int j=0; j < param.num_src; j++){
        //   blas::caxpy(-beta(j,i), Ap.Component(j), rSloppy.Component(i));
        // }
        const int j3 = param.num_src/3;
        const int j2 = ((param.num_src%3)/2);
        const int j1 = ((param.num_src%3)%2);
        for (int j=0;j<j3;j++){
          blas::caxpbypczpw(-beta(3*j,i),Ap.Component(3*j),-beta(3*j+1,i),Ap.Component(3*j+1),-beta(3*j+2,i),Ap.Component(3*j+2),rSloppy.Component(i));
        }
        for (int j=0;j<j2;j++){
          blas::caxpbypz(-beta(3*j3+2*j,i),Ap.Component(3*j3+2*j),-beta(3*j3+2*j+1,i),Ap.Component(3*j3+2*j+1),rSloppy.Component(i));
        }
        for (int j=0; j<j1;j++){
          blas::caxpy(-beta(3*j3+2*j2+j,i),Ap.Component(3*j3+2*j2+j),rSloppy.Component(i));
        }
      }


      POP_RANGE
      PUSH_RANGE("GramSchmidt",3)
      for(int i=0; i< param.num_src; i++){
        blas::copy(rnew.Component(i), rSloppy.Component(i));
      }

      for(int i=0; i<param.num_src; i++){
        for(int j=i; j < param.num_src; j++){
          r2(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
          if (i!=j) r2(j,i) = std::conj(r2(i,j));
        }
      }


      Eigen::LLT<MatrixXcd> lltOfr2(r2); // compute the Cholesky decomposition of A
      L = lltOfr2.matrixL(); // retrieve factor L  in the decomposition
      S = L.adjoint();
      MatrixXcd Linv = S.inverse();

      for(int i=0; i<param.num_src; i++){
        blas::zero(rSloppy.Component(i));


        for(int j=0;j<param.num_src; j++){
          // nsrc * nsrc (write rnew) + nsrc * nsrc (read rnew) + nsrc * nsrc (read p)
          blas::caxpy(Linv(j,i),rnew.Component(j),rSloppy.Component(i));
          // if we can do something like rnew_i(x) += sum_j S(j,i)^h p_j(x) we get away with
          // nsrc * read rnew + nsrc * write rnew + nsrc*nsrc read p_j = (nsrc+nsrc + nsrc^2)
          // maybe we can also use i as blockIdx.y and by that explit caching of the p_j(x) Components
        }
      }

    for(int i=0; i<param.num_src; i++){
      for(int j=0;j<param.num_src; j++){
          blas::caxpy(Linv(j,i),rnew.Component(j),rSloppy.Component(i));
        }
      }

      // for(int i=0; i < param.num_src; i++){
      //   // scales with nsrc
      //   n = blas::norm2(rSloppy.Component(i));
      //   blas::ax(1/sqrt(n),rSloppy.Component(i));
      //   // (nsrc * (nsrc-1))/2) * nsrc
      //   for(int j=i+1; j < param.num_src; j++) {
      //     ri=blas::cDotProduct(rSloppy.Component(i),rSloppy.Component(j));
      //     blas::caxpy(-ri,rSloppy.Component(i),rSloppy.Component(j));
      //
      //   }
      // }
      //
      //
      // S = MatrixXcd::Zero(param.num_src,param.num_src);
      // for (int i = 0; i < param.num_src; i++){
      //   for (int j=i; j < param.num_src; j++){
      //     S(i,j) = blas::cDotProduct(rSloppy.Component(i), rnew.Component(j));
      //   }
      // }
      POP_RANGE
      #ifdef MWVERBOSE
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j<param.num_src; j++){
          pTp(i,j) = blas::cDotProduct(rSloppy.Component(i), rSloppy.Component(j));
        }
      }
      std::cout << " rTr " << std::endl << pTp << std::endl;

      // gamma = qr.householderQ();
      // gamma = gamma.transpose().eval();
      std::cout <<  "QR" << S<<  std::endl << "QP " << S.inverse()*S << std::endl;;
      #endif

      // update p
      PUSH_RANGE("LinearAlgebra",2)
      // mw this needs a lot of cleanup
      // for(int i=0; i < param.num_src; i++){
      //   // blas::ax(0,rnew.Component(i)); // do we need components here?
      // }
      // add r
      // scales with nsrc
      for(int i=0; i<param.num_src; i++){
        blas::copy(rnew.Component(i),rSloppy.Component(i));
      }

      for(int i=0; i<param.num_src; i++){
        // for(int j=0;j<param.num_src; j++){
        //   // nsrc * nsrc (write rnew) + nsrc * nsrc (read rnew) + nsrc * nsrc (read p)
        //   blas::caxpy(std::conj(S(i,j)),p.Component(j),rnew.Component(i));
        //   // if we can do something like rnew_i(x) += sum_j S(j,i)^h p_j(x) we get away with
        //   // nsrc * read rnew + nsrc * write rnew + nsrc*nsrc read p_j = (nsrc+nsrc + nsrc^2)
        //   // maybe we can also use i as blockIdx.y and by that explit caching of the p_j(x) Components
        // }
        const int j3 = param.num_src/3;
        const int j2 = ((param.num_src%3)/2);
        const int j1 = ((param.num_src%3)%2);
        for (int j=0;j<j3;j++){
          blas::caxpbypczpw(std::conj(S(i,3*j)),p.Component(3*j),std::conj(S(i,3*j+1)),p.Component(3*j+1),std::conj(S(i,3*j+2)),p.Component(3*j+2),rnew.Component(i));
        }
        for (int j=0;j<j2;j++){
          blas::caxpbypz(std::conj(S(i,3*j3+2*j)),p.Component(3*j3+2*j),std::conj(S(i,3*j3+2*j+1)),p.Component(3*j3+2*j+1),rnew.Component(i));
        }
        for (int j=0; j<j1;j++){
          blas::caxpy(std::conj(S(i,3*j3+2*j2+j)),p.Component(3*j3+2*j2+j),rnew.Component(i));
        }

      }
      // copy sclae with nsrc
      for(int i=0; i < param.num_src; i++){
        blas::copy(p.Component(i),rnew.Component(i)); // do we need components here?
      }

      C = S * C;
      POP_RANGE
      #ifdef MWVERBOSE
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j<param.num_src; j++){
          pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
        }
      }

      std::cout << " pTp " << std::endl << pTp << std::endl;
      std::cout <<  "S " << S<<  std::endl << "C " << C << std::endl;
      #endif

      // calculate the residuals for all shifts
      r2avg=0;
      for (int j=0; j<param.num_src; j++ ){
        r2(j,j) = C(0,j)*conj(C(0,j));
        for(int i=1; i < param.num_src; i++)
        r2(j,j) += C(i,j) * conj(C(i,j));
        r2avg += r2(j,j).real();
      }

      //       bool updateX=false;
      //       bool updateR=false;
      // //      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? true : false;
      // //      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? true : false;
      // //
      // // printfQuda("Checking reliable update %i %i\n",updateX,updateR);
      //       // reliable update conditions
      //       // for(int i=0; i<param.num_src; i++){
      //       //   rNorm[i] = sqrt(r2(i,i));
      //       //   if (rNorm[i] > maxrx[i]) maxrx[i] = rNorm[i];
      //       //   if (rNorm[i] > maxrr[i]) maxrr[i] = rNorm[i];
      //       //   updateX = (rNorm[i] < delta * r0Norm[i] && r0Norm[i] <= maxrx[i]) ? true : false;
      //       //   updateR = ((rNorm[i] < delta * maxrr[i] && r0Norm[i] <= maxrr[i]) || updateX) ? true : false;
      //       //   // printfQuda("Checking reliable update %i %i %i\n",i, updateX,updateR);
      //       //
      //       //   // // force a reliable update if we are within target tolerance (only if doing reliable updates)
      //       //   // if ( convergence(r2(i,i), heavy_quark_res[i], stop[i], param.tol_hq) && param.delta >= param.tol ) updateX = true;
      //       //   //
      //       //   // // For heavy-quark inversion force a reliable update if we continue after
      //       //   // if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2(i,i), heavy_quark_res[i], stop[i], param.tol_hq) and param.delta >= param.tol ) {
      //       //   //   updateX = true;
      //       //   // }
      //       // }
      //       if ( (updateR || updateX )) {
      //         printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
      //         updateX=false;
      //         updateR=false;
      //         printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
      //       }
      //
      //       if ( !(updateR || updateX )) {
      //         // for(int i=0; i<param.num_src; i++){
      //         //   beta(i,i) = sigma(i,i) / r2_old(i,i);  // use the alternative beta computation
      //         // }
      //         beta = gamma * r2_old.inverse() * sigma;
      //
      //         std::cout << "beta\n" << beta << std::endl;
      //
      //         if (param.pipeline && !breakdown)
      //         for(int i=0; i<param.num_src; i++){
      //           blas::tripleCGUpdate(alpha(i,i), beta(i,i), Ap.Component(i), rSloppy.Component(i), xSloppy.Component(i), p.Component(i));
      //         }
      //         else{
      //           for(int i=0; i<param.num_src; i++){
      //             for(int j=0; j<param.num_src; j++){
      //               //MW: probably need to split this into to separate updates here. For now better save, optimize later
      //               // x[i] = alpha*p[i] + x[i]; p[i] = r[i] + beta*p[i]
      //               // x = p, y = x, z =r . We can only update X after p update has been completed
      //
      //               //blas::axpyZpbx(alpha(i,i), p.Component(i), xSloppy.Component(i), rSloppy.Component(i), beta(i,i));
      //               blas::axpy(alpha(j,i),p.Component(j),xSloppy.Component(i));
      //             }
      //           }
      //         }
      // gamma = MatrixXd::Identity(param.num_src,param.num_src) * (k+1);
      // gamma(0,1) = 1.;

      //   // set to zero
      //   for(int i=0; i < param.num_src; i++){
      //     blas::ax(0,pnew.Component(i)); // do we need components here?
      //   }
      //   // add r
      //   for(int i=0; i<param.num_src; i++){
      //     // for(int j=0;j<param.num_src; j++){
      //       // order of updating p might be relevant here
      //       blas::axpy(1.0,r.Component(i),pnew.Component(i));
      //       // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
      //     // }
      //   }
      //   // beta = beta * gamma.inverse();
      //   for(int i=0; i<param.num_src; i++){
      //     for(int j=0;j<param.num_src; j++){
      //       double rcoeff= (j==0?1.0:0.0);
      //       // order of updating p might be relevant hereq
      //       blas::axpy(beta(j,i),p.Component(j),pnew.Component(i));
      //       // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
      //     }
      //   }
      //   // now need to do something with the p's
      //
      //   for(int i=0; i< param.num_src; i++){
      //     blas::copy(p.Component(i), pnew.Component(i));
      //   }
      //
      //
      //
      //   for(int i=0; i < param.num_src; i++){
      //     double n = blas::norm2(p.Component(i));
      //     blas::ax(1/sqrt(n),p.Component(i));
      //     for(int j=i+1; j < param.num_src; j++) {
      //       double ri=blas::reDotProduct(p.Component(i),p.Component(j));
      //       blas::axpy(-ri,p.Component(i),p.Component(j));
      //
      //     }
      //   }
      //
      //
      //   gamma = MatrixXd::Zero(param.num_src,param.num_src);
      //   for ( int i = 0; i < param.num_src; i++){
      //     for (int j=i; j < param.num_src; j++){
      //       gamma(i,j) = blas::reDotProduct(p.Component(i),pnew.Component(j));
      //     }
      //   }
      //
      //   for(int i=0; i<param.num_src; i++){
      //     for(int j=0; j<param.num_src; j++){
      //       pTp(i,j) = blas::reDotProduct(p.Component(i), p.Component(j));
      //     }
      //   }
      //   std::cout << " pTp " << std::endl << pTp << std::endl;
      //
      //   // gamma = qr.householderQ();
      //   // gamma = gamma.transpose().eval();
      //   std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
      //
      // }


      //   if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
      //     if (&x != &xSloppy) {
      //       blas::copy(tmp, y);   //  FIXME: check whether copy works here
      //       for(int i=0; i<param.num_src; i++){
      //         heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy.Component(i), tmp.Component(i), rSloppy.Component(i)).z);
      //       }
      //     } else {
      //       blas::copy(r, rSloppy);  //  FIXME: check whether copy works here
      //       for(int i=0; i<param.num_src; i++){
      //         heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(x.Component(i), y.Component(i), r.Component(i)).z);
      //       }
      //     }
      //   }
      //
      //   steps_since_reliable++;
      // } else {
      //   printfQuda("reliable update\n");
      //   for(int i=0; i<param.num_src; i++){
      //     blas::axpy(alpha(i,i), p.Component(i), xSloppy.Component(i));
      //   }
      //   blas::copy(x, xSloppy); // nop when these pointers alias
      //
      //   for(int i=0; i<param.num_src; i++){
      //     blas::xpy(x.Component(i), y.Component(i)); // swap these around?
      //   }
      //   for(int i=0; i<param.num_src; i++){
      //     mat(r.Component(i), y.Component(i), x.Component(i), tmp3.Component(i)); //  here we can use x as tmp
      //   }
      //   for(int i=0; i<param.num_src; i++){
      //     r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
      //   }
      //
      //   for(int i=0; i<param.num_src; i++){
      //     blas::copy(rSloppy.Component(i), r.Component(i)); //nop when these pointers alias
      //     blas::zero(xSloppy.Component(i));
      //   }
      //
      //   // calculate new reliable HQ resididual
      //   if (use_heavy_quark_res){
      //     for(int i=0; i<param.num_src; i++){
      //       heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(y.Component(i), r.Component(i)).z);
      //     }
      //   }
      //
      //   // MW: FIXME as this probably goes terribly wrong right now
      //   for(int i = 0; i<param.num_src; i++){
      //     // break-out check if we have reached the limit of the precision
      //     if (sqrt(r2(i,i)) > r0Norm[i] && updateX) { // reuse r0Norm for this
      //       resIncrease++;
      //       resIncreaseTotal++;
      //       warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
      //       sqrt(r2(i,i)), r0Norm[i], resIncreaseTotal);
      //       if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
      //         if (use_heavy_quark_res) {
      //           L2breakdown = true;
      //         } else {
      //           warningQuda("CG: solver exiting due to too many true residual norm increases");
      //           break;
      //         }
      //       }
      //     } else {
      //       resIncrease = 0;
      //     }
      //   }
      //   // if L2 broke down already we turn off reliable updates and restart the CG
      //   for(int i = 0; i<param.num_src; i++){
      //     if (use_heavy_quark_res and L2breakdown) {
      //       delta = 0;
      //       warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
      //       heavy_quark_restart = true;
      //       if (heavy_quark_res[i] > heavy_quark_res_old[i]) {
      //         hqresIncrease++;
      //         warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res[i], heavy_quark_res_old[i]);
      //         // break out if we do not improve here anymore
      //         if (hqresIncrease > hqmaxresIncrease) {
      //           warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
      //           break;
      //         }
      //       }
      //     }
      //   }
      //
      //   for(int i=0; i<param.num_src; i++){
      //     rNorm[i] = sqrt(r2(i,i));
      //     maxrr[i] = rNorm[i];
      //     maxrx[i] = rNorm[i];
      //     r0Norm[i] = rNorm[i];
      //     heavy_quark_res_old[i] = heavy_quark_res[i];
      //   }
      //   rUpdate++;
      //
      //   if (use_heavy_quark_res and heavy_quark_restart) {
      //     // perform a restart
      //     blas::copy(p, rSloppy);
      //     heavy_quark_restart = false;
      //   } else {
      //     // explicitly restore the orthogonality of the gradient vector
      //     for(int i=0; i<param.num_src; i++){
      //       double rp = blas::reDotProduct(rSloppy.Component(i), p.Component(i)) / (r2(i,i));
      //       blas::axpy(-rp, rSloppy.Component(i), p.Component(i));
      //
      //       beta(i,i) = r2(i,i) / r2_old(i,i);
      //       blas::xpay(rSloppy.Component(i), beta(i,i), p.Component(i));
      //     }
      //   }
      //
      //   steps_since_reliable = 0;
      // }

      breakdown = false;
      k++;

      allconverged = true;


      PrintStats("CG", k, r2avg / param.num_src, b2avg, heavy_quark_res[0]);
      for(int i=0; i<param.num_src; i++){
        // PrintStats("CG", k, r2(i,i).real(), b2[i], heavy_quark_res[i]);
        // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
        converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
      }

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        for(int i=0; i<param.num_src; i++){
          // L2 is concverged or precision maxed out for L2
          bool L2done = L2breakdown or convergenceL2(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
          // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
          bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
          converged[i] = L2done and HQdone;
        }
      }

    }

    blas::copy(x, xSloppy);
    for(int i=0; i<param.num_src; i++){
      blas::xpy(y.Component(i), x.Component(i));
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
    warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    for(int i=0; i<param.num_src; i++){
      mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
      param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;

      PrintSummary("CG", k, r2(i,i).real(), b2[i]);
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
    if (xSloppy.Precision() != x.Precision()) delete x_sloppy;

    delete rpnew;

    delete pp;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;

    #endif

  }

  #else
  #define BLOCKCG_GS 1
  void CG::solve(ColorSpinorField& x, ColorSpinorField& b) {
    #ifndef BLOCKSOLVER
    errorQuda("QUDA_BLOCKSOLVER not built.");
    #else
    #ifdef BLOCKCG_GS
    printfQuda("BCGdQ Solver\n");
    #else
    printfQuda("BCQ Solver\n");
    #endif
    const bool use_block = true;
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
    errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    using Eigen::MatrixXcd;



    MatrixXcd mPAP(param.num_src,param.num_src);
    MatrixXcd mRR(param.num_src,param.num_src);


    // Check to see that we're not trying to invert on a zero-field source
    //MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_MULTI_SHIFT];
    double b2avg=0;
    double r2avg=0;
    for(int i=0; i< param.num_src; i++){
      b2[i]=blas::norm2(b.Component(i));
      b2avg += b2[i];
      if(b2[i] == 0){
        profile.TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source\n");
        x=b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    #ifdef MWVERBOSE
    MatrixXcd b2m(param.num_src,param.num_src);
    // just to check details of b
    for(int i=0; i<param.num_src; i++){
      for(int j=0; j<param.num_src; j++){
        b2m(i,j) = blas::cDotProduct(b.Component(i), b.Component(j));
      }
    }

    std::cout << "b2m\n" <<  b2m << std::endl;
    #endif

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;


    //  const int i = 0;  // MW: hack to be able to write Component(i) instead and try with i=0 for now

    for(int i=0; i<param.num_src; i++){
      mat(r.Component(i), x.Component(i), y.Component(i));
    }

    // double r2[QUDA_MAX_MULTI_SHIFT];
    MatrixXcd r2(param.num_src,param.num_src);
    for(int i=0; i<param.num_src; i++){
      r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
      printfQuda("r2[%i] %e\n", i, r2(i,i).real());
    }
    if(use_block){
      // MW need to initalize the full r2 matrix here
      for(int i=0; i<param.num_src; i++){
        for(int j=i+1; j<param.num_src; j++){
          r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
          r2(j,i) = std::conj(r2(i,j));
        }
      }
    }

    csParam.setPrecision(param.precision_sloppy);
    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      // will that work ?
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
      for(int i=0; i<param.num_src; i++){
        blas::copy(r_sloppy->Component(i), r.Component(i)); //nop when these pointers alias
      }
    }


    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
    !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p =
    (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
    ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField* pp = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &p = *pp;
    ColorSpinorField* ppnew = ColorSpinorField::Create(rSloppy, csParam);
    ColorSpinorField &pnew = *ppnew;

    if (&x != &xSloppy) {
      blas::copy(y, x);
      blas::zero(xSloppy);
    } else {
      blas::zero(y);
    }

    const bool use_heavy_quark_res =
    (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    bool heavy_quark_restart = false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    MatrixXcd r2_old(param.num_src, param.num_src);
    double heavy_quark_res[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark res idual
    double heavy_quark_res_old[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark residual
    double stop[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
      if (use_heavy_quark_res) {
        heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
        heavy_quark_res_old[i] = heavy_quark_res[i];   // heavy quark residual
      }
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    MatrixXcd alpha = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd beta = MatrixXcd::Zero(param.num_src,param.num_src);
    MatrixXcd gamma = MatrixXcd::Identity(param.num_src,param.num_src);
    //  gamma = gamma * 2.0;

    MatrixXcd pAp(param.num_src, param.num_src);
    MatrixXcd pTp(param.num_src, param.num_src);
    int rUpdate = 0;

    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }

    double delta = param.delta;//MW: hack no reliable updates param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    // 0 means we have no tolerance
    // maybe we should expose this as a parameter
    const int hqmaxresIncrease = maxResIncrease + 1;

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;

    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    int k = 0;

    for(int i=0; i<param.num_src; i++){
      r2avg+=r2(i,i).real();
    }
    PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);
    int steps_since_reliable = 1;
    bool allconverged = true;
    bool converged[QUDA_MAX_MULTI_SHIFT];
    for(int i=0; i<param.num_src; i++){
      converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }
    MatrixXcd sigma(param.num_src,param.num_src);

    #ifdef BLOCKCG_GS
    // begin ignore Gram-Schmidt for now

    for(int i=0; i < param.num_src; i++){
      double n = blas::norm2(p.Component(i));
      blas::ax(1/sqrt(n),p.Component(i));
      for(int j=i+1; j < param.num_src; j++) {
        std::complex<double> ri=blas::cDotProduct(p.Component(i),p.Component(j));
        blas::caxpy(-ri,p.Component(i),p.Component(j));
      }
    }

    gamma = MatrixXcd::Zero(param.num_src,param.num_src);
    for ( int i = 0; i < param.num_src; i++){
      for (int j=i; j < param.num_src; j++){
        gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
      }
    }
    #endif
    // end ignore Gram-Schmidt for now

    #ifdef MWVERBOSE
    for(int i=0; i<param.num_src; i++){
      for(int j=0; j<param.num_src; j++){
        pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
      }
    }

    std::cout << " pTp " << std::endl << pTp << std::endl;
    std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
    #endif
    while ( !allconverged && k < param.maxiter ) {
      for(int i=0; i<param.num_src; i++){
        matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
      }


      bool breakdown = false;
      // FIXME: need to check breakdown
      // current implementation sets breakdown to true for pipelined CG if one rhs triggers breakdown
      // this is probably ok


      if (param.pipeline) {
        errorQuda("pipeline not implemented");
        // for(int i=0; i<param.num_src; i++){
        //   //MW: ignore this for now
        //   double3 triplet = blas::tripleCGReduction(rSloppy.Component(i), Ap.Component(i), p.Component(i));
        //   r2(i,i) = triplet.x; double Ap2 = triplet.y; pAp(i,i) = triplet.z;
        //   r2_old(i,i) = r2(i,i);
        //   alpha(i,i) = r2(i,i) / pAp(i,i);
        //   sigma(i,i) = alpha(i,i)*(alpha(i,i) * Ap2 - pAp(i,i));
        //   if (sigma(i,i) < 0.0 || steps_since_reliable == 0) { // sigma condition has broken down
        //     r2(i,i) = blas::axpyNorm(-alpha(i,i), Ap.Component(i), rSloppy.Component(i));
        //     sigma(i,i) = r2(i,i);
        //     breakdown = true;
        //   }
        //
        //   r2(i,i) = sigma(i,i);
        // }
      } else {
        r2_old = r2;
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j < param.num_src; j++){
            if(use_block or i==j)
            pAp(i,j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
            else
            pAp(i,j) = 0.;
          }
        }

        alpha = pAp.inverse() * gamma.adjoint().inverse() * r2;
        #ifdef MWVERBOSE
        std::cout << "alpha\n" << alpha << std::endl;

        if(k==1){
          std::cout << "pAp " << std::endl <<pAp << std::endl;
          std::cout << "pAp^-1 " << std::endl <<pAp.inverse() << std::endl;
          std::cout << "r2 " << std::endl <<r2 << std::endl;
          std::cout << "alpha " << std::endl <<alpha << std::endl;
          std::cout << "pAp^-1r2" << std::endl << pAp.inverse()*r2 << std::endl;
        }
        #endif
        // here we are deploying the alternative beta computation
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j < param.num_src; j++){

            blas::caxpy(-alpha(j,i), Ap.Component(j), rSloppy.Component(i));
          }
        }
        // MW need to calculate the full r2 matrix here, after update. Not sure how to do alternative sigma yet ...
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j<param.num_src; j++){
            if(use_block or i==j)
            r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
            else
            r2(i,j) = 0.;
          }
        }
        sigma = r2;
      }


      bool updateX=false;
      bool updateR=false;
      //      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? true : false;
      //      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? true : false;
      //
      // printfQuda("Checking reliable update %i %i\n",updateX,updateR);
      // reliable update conditions
      for(int i=0; i<param.num_src; i++){
        rNorm[i] = sqrt(r2(i,i).real());
        if (rNorm[i] > maxrx[i]) maxrx[i] = rNorm[i];
        if (rNorm[i] > maxrr[i]) maxrr[i] = rNorm[i];
        updateX = (rNorm[i] < delta * r0Norm[i] && r0Norm[i] <= maxrx[i]) ? true : false;
        updateR = ((rNorm[i] < delta * maxrr[i] && r0Norm[i] <= maxrr[i]) || updateX) ? true : false;
        // printfQuda("Checking reliable update %i %i %i\n",i, updateX,updateR);

        // // force a reliable update if we are within target tolerance (only if doing reliable updates)
        // if ( convergence(r2(i,i), heavy_quark_res[i], stop[i], param.tol_hq) && param.delta >= param.tol ) updateX = true;
        //
        // // For heavy-quark inversion force a reliable update if we continue after
        // if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2(i,i), heavy_quark_res[i], stop[i], param.tol_hq) and param.delta >= param.tol ) {
        //   updateX = true;
        // }
      }
      if ( (updateR || updateX )) {
        // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
        updateX=false;
        updateR=false;
        // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
      }

      if ( !(updateR || updateX )) {
        // for(int i=0; i<param.num_src; i++){
        //   beta(i,i) = sigma(i,i) / r2_old(i,i);  // use the alternative beta computation
        // }
        beta = gamma * r2_old.inverse() * sigma;
        #ifdef MWVERBOSE
        std::cout << "beta\n" << beta << std::endl;
        #endif
        if (param.pipeline && !breakdown)
        errorQuda("pipeline not implemented");
        //for(int i=0; i<param.num_src; i++){
        //   blas::tripleCGUpdate(alpha(i,i), beta(i,i), Ap.Component(i), rSloppy.Component(i), xSloppy.Component(i), p.Component(i));
        // }
        else{
          for(int i=0; i<param.num_src; i++){
            for(int j=0; j<param.num_src; j++){
              //MW: probably need to split this into to separate updates here. For now better save, optimize later
              // x[i] = alpha*p[i] + x[i]; p[i] = r[i] + beta*p[i]
              // x = p, y = x, z =r . We can only update X after p update has been completed

              //blas::axpyZpbx(alpha(i,i), p.Component(i), xSloppy.Component(i), rSloppy.Component(i), beta(i,i));
              blas::caxpy(alpha(j,i),p.Component(j),xSloppy.Component(i));
            }
          }
          // gamma = MatrixXd::Identity(param.num_src,param.num_src) * (k+1);
          // gamma(0,1) = 1.;

          // set to zero
          for(int i=0; i < param.num_src; i++){
            blas::ax(0,pnew.Component(i)); // do we need components here?
          }
          // add r
          for(int i=0; i<param.num_src; i++){
            // for(int j=0;j<param.num_src; j++){
            // order of updating p might be relevant here
            blas::axpy(1.0,r.Component(i),pnew.Component(i));
            // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
            // }
          }
          // beta = beta * gamma.inverse();
          for(int i=0; i<param.num_src; i++){
            for(int j=0;j<param.num_src; j++){
              double rcoeff= (j==0?1.0:0.0);
              // order of updating p might be relevant hereq
              blas::caxpy(beta(j,i),p.Component(j),pnew.Component(i));
              // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
            }
          }
          // now need to do something with the p's

          for(int i=0; i< param.num_src; i++){
            blas::copy(p.Component(i), pnew.Component(i));
          }


          #ifdef BLOCKCG_GS
          for(int i=0; i < param.num_src; i++){
            double n = blas::norm2(p.Component(i));
            blas::ax(1/sqrt(n),p.Component(i));
            for(int j=i+1; j < param.num_src; j++) {
              std::complex<double> ri=blas::cDotProduct(p.Component(i),p.Component(j));
              blas::caxpy(-ri,p.Component(i),p.Component(j));

            }
          }


          gamma = MatrixXcd::Zero(param.num_src,param.num_src);
          for ( int i = 0; i < param.num_src; i++){
            for (int j=i; j < param.num_src; j++){
              gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
            }
          }
          #endif

          #ifdef MWVERBOSE
          for(int i=0; i<param.num_src; i++){
            for(int j=0; j<param.num_src; j++){
              pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
            }
          }
          std::cout << " pTp " << std::endl << pTp << std::endl;

          // gamma = qr.householderQ();
          // gamma = gamma.transpose().eval();
          std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
          #endif
        }


        if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
          if (&x != &xSloppy) {
            blas::copy(tmp, y);   //  FIXME: check whether copy works here
            for(int i=0; i<param.num_src; i++){
              heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy.Component(i), tmp.Component(i), rSloppy.Component(i)).z);
            }
          } else {
            blas::copy(r, rSloppy);  //  FIXME: check whether copy works here
            for(int i=0; i<param.num_src; i++){
              heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(x.Component(i), y.Component(i), r.Component(i)).z);
            }
          }
        }

        steps_since_reliable++;
      } else {
        printfQuda("reliable update\n");
        for(int i=0; i<param.num_src; i++){
          blas::axpy(alpha(i,i).real(), p.Component(i), xSloppy.Component(i));
        }
        blas::copy(x, xSloppy); // nop when these pointers alias

        for(int i=0; i<param.num_src; i++){
          blas::xpy(x.Component(i), y.Component(i)); // swap these around?
        }
        for(int i=0; i<param.num_src; i++){
          mat(r.Component(i), y.Component(i), x.Component(i), tmp3.Component(i)); //  here we can use x as tmp
        }
        for(int i=0; i<param.num_src; i++){
          r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
        }

        for(int i=0; i<param.num_src; i++){
          blas::copy(rSloppy.Component(i), r.Component(i)); //nop when these pointers alias
          blas::zero(xSloppy.Component(i));
        }

        // calculate new reliable HQ resididual
        if (use_heavy_quark_res){
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(y.Component(i), r.Component(i)).z);
          }
        }

        // MW: FIXME as this probably goes terribly wrong right now
        for(int i = 0; i<param.num_src; i++){
          // break-out check if we have reached the limit of the precision
          if (sqrt(r2(i,i).real()) > r0Norm[i] && updateX) { // reuse r0Norm for this
            resIncrease++;
            resIncreaseTotal++;
            warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2(i,i).real()), r0Norm[i], resIncreaseTotal);
            if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
              if (use_heavy_quark_res) {
                L2breakdown = true;
              } else {
                warningQuda("CG: solver exiting due to too many true residual norm increases");
                break;
              }
            }
          } else {
            resIncrease = 0;
          }
        }
        // if L2 broke down already we turn off reliable updates and restart the CG
        for(int i = 0; i<param.num_src; i++){
          if (use_heavy_quark_res and L2breakdown) {
            delta = 0;
            warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
            heavy_quark_restart = true;
            if (heavy_quark_res[i] > heavy_quark_res_old[i]) {
              hqresIncrease++;
              warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res[i], heavy_quark_res_old[i]);
              // break out if we do not improve here anymore
              if (hqresIncrease > hqmaxresIncrease) {
                warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
                break;
              }
            }
          }
        }

        for(int i=0; i<param.num_src; i++){
          rNorm[i] = sqrt(r2(i,i).real());
          maxrr[i] = rNorm[i];
          maxrx[i] = rNorm[i];
          r0Norm[i] = rNorm[i];
          heavy_quark_res_old[i] = heavy_quark_res[i];
        }
        rUpdate++;

        if (use_heavy_quark_res and heavy_quark_restart) {
          // perform a restart
          blas::copy(p, rSloppy);
          heavy_quark_restart = false;
        } else {
          // explicitly restore the orthogonality of the gradient vector
          for(int i=0; i<param.num_src; i++){
            double rp = blas::reDotProduct(rSloppy.Component(i), p.Component(i)) / (r2(i,i).real());
            blas::axpy(-rp, rSloppy.Component(i), p.Component(i));

            beta(i,i) = r2(i,i) / r2_old(i,i);
            blas::xpay(rSloppy.Component(i), beta(i,i).real(), p.Component(i));
          }
        }

        steps_since_reliable = 0;
      }

      breakdown = false;
      k++;

      allconverged = true;
      r2avg=0;
      for(int i=0; i<param.num_src; i++){
        r2avg+= r2(i,i).real();
        // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
        converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
      }
      PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        for(int i=0; i<param.num_src; i++){
          // L2 is concverged or precision maxed out for L2
          bool L2done = L2breakdown or convergenceL2(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
          // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
          bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
          converged[i] = L2done and HQdone;
        }
      }

    }

    blas::copy(x, xSloppy);
    for(int i=0; i<param.num_src; i++){
      blas::xpy(y.Component(i), x.Component(i));
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
    warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
    printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    for(int i=0; i<param.num_src; i++){
      mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
      param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      param.true_res_offset[i] = param.true_res;
      param.true_res_hq_offset[i] = param.true_res_hq;

      PrintSummary("CG", k, r2(i,i).real(), b2[i]);
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (rSloppy.Precision() != r.Precision()) delete r_sloppy;
    if (xSloppy.Precision() != x.Precision()) delete x_sloppy;

    delete pp;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;

    #endif

  }
  #endif


}  // namespace quda
