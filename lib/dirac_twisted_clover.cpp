#include <dirac_quda.h>
#include <blas_quda.h>
#include <iostream>

namespace quda {

	int DiracTwistedClover::initTMCFlag = 0;//set to 1 for parity spinors, and 2 for full spinors 

	DiracTwistedClover::DiracTwistedClover(const DiracTwistedClover &dirac) : DiracWilson(dirac), mu(dirac.mu), epsilon(dirac.epsilon), clover(dirac.clover), cloverInv(dirac.cloverInv)
	{
		initCloverConstants(clover, profile);
		initCloverConstants(cloverInv, profile);
	}

	DiracTwistedClover::DiracTwistedClover(const DiracParam &param, const int nDim) : DiracWilson(param, nDim), mu(param.mu), epsilon(param.epsilon), clover(*(param.clover)), cloverInv(*(param.cloverInv))
	{
		initCloverConstants(clover, profile);
		initCloverConstants(cloverInv, profile);
	}

	DiracTwistedClover::~DiracTwistedClover() { }

	DiracTwistedClover& DiracTwistedClover::operator=(const DiracTwistedClover &dirac)
	{
		if (&dirac != this)
		{
			DiracWilson::operator=(dirac);
			clover = dirac.clover;
		}

		return *this;
	}

	void DiracTwistedClover::initConstants(const cudaColorSpinorField &a) const
	{
		if (a.SiteSubset() == QUDA_PARITY_SITE_SUBSET && initTMCFlag != 1)
		{
			int flavor_stride = (a.TwistFlavor() != QUDA_TWIST_PLUS || a.TwistFlavor() != QUDA_TWIST_MINUS) ? a.VolumeCB()/2 : a.VolumeCB();
			initSpinorConstants(a, profile);
			initTwistedMassConstants(flavor_stride, profile);
			initTMCFlag = 1;
		}
		else if (a.SiteSubset() == QUDA_FULL_SITE_SUBSET && initTMCFlag != 2)
		{
			int flavor_stride = (a.TwistFlavor() != QUDA_TWIST_PLUS || a.TwistFlavor() != QUDA_TWIST_MINUS) ? a.VolumeCB()/4 : a.VolumeCB()/2;
			initSpinorConstants(a, profile);
			initTwistedMassConstants(flavor_stride, profile);
			initTMCFlag = 2;
		}
	}

	void DiracTwistedClover::checkParitySpinor(const cudaColorSpinorField &out, const cudaColorSpinorField &in) const
	{
		Dirac::checkParitySpinor(out, in);

		if (out.Volume() != clover.VolumeCB())
			errorQuda("Parity spinor volume %d doesn't match clover checkboard volume %d", out.Volume(), clover.VolumeCB());
	}

	// Protected method for applying twist
/*
	void DiracTwistedClover::twistedApply(cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaTwistGamma5Type twistType) const
	{
		checkParitySpinor(out, in);
		initConstants(in);

		if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
			errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

		if (in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS)
		{
			double flavor_mu = in.TwistFlavor() * mu;
			twistGamma5Cuda(&out, &in, dagger, kappa, flavor_mu, 0.0, twistType);
			flops += 24ll*in.Volume();
		}
		else
      			errorQuda("DiracTwistedClover::twistedApply method for flavor doublet is not implemented..\n");  
	}


  // Public method to apply the twist
  void DiracTwistedClover::Twist(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    twistedApply(out, in, QUDA_TWIST_GAMMA5_DIRECT);
  }
*/
  void DiracTwistedClover::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());

    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID) {
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());
    }

    // We can eliminate this temporary at the expense of more kernels (like clover)
    cudaColorSpinorField *tmp=0; // this hack allows for tmp2 to be full or parity field
    if (tmp2) {
      if (tmp2->SiteSubset() == QUDA_FULL_SITE_SUBSET) tmp = &(tmp2->Even());
      else tmp = tmp2;
    }
    bool reset = newTmp(&tmp, in.Even());

    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda      
    initConstants(in);
  
    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = 2.0 * kappa * in.TwistFlavor() * mu;//for direct twist (must be daggered separately)  
      twistedCloverDslashCuda(&out.Odd(), gauge, clover, cloverInv, &in.Even(), QUDA_ODD_PARITY, dagger, &in.Odd(), QUDA_DEG_DSLASH_TWIST_CLOVER_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      twistedCloverDslashCuda(&out.Even(), gauge, clover, cloverInv, &in.Odd(), QUDA_EVEN_PARITY, dagger, &in.Even(), QUDA_DEG_DSLASH_TWIST_CLOVER_XPAY, a, -kappa, 0.0, 0.0, commDim, profile);
      flops += (1320ll+72ll)*in.Volume();
    } else {
      errorQuda("Non-deg twisted clover not implemented yet");
    }
    deleteTmp(&tmp, reset);
  }

  void DiracTwistedClover::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    checkFullSpinor(out, in);
    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedClover::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
				 cudaColorSpinorField &x, cudaColorSpinorField &b, 
				 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    src = &b;
    sol = &x;
  }

  void DiracTwistedClover::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				     const QudaSolutionType solType) const
  {
    // do nothing
  }


  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracTwistedCloverPC &dirac) : DiracTwistedClover(dirac) { }

  DiracTwistedCloverPC::DiracTwistedCloverPC(const DiracParam &param, const int nDim) : DiracTwistedClover(param, nDim){ }

  DiracTwistedCloverPC::~DiracTwistedCloverPC()
  {

  }

  DiracTwistedCloverPC& DiracTwistedCloverPC::operator=(const DiracTwistedCloverPC &dirac)
  {
    if (&dirac != this) {
      DiracTwistedClover::operator=(dirac);
    }
    return *this;
  }

  // Public method to apply the inverse twist
  void DiracTwistedCloverPC::TwistCloverInv(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
//    twistedApply(out, in, QUDA_TWIST_GAMMA5_INVERSE);
  }

  // apply hopping term, then inverse twist: (A_ee^-1 D_eo) or (A_oo^-1 D_oe),
  // and likewise for dagger: (D^dagger_eo D_ee^-1) or (D^dagger_oe A_oo^-1)
  void DiracTwistedCloverPC::Dslash
  (cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

    setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda
    initConstants(in);
  
    if (in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = -2.0 * kappa * in.TwistFlavor() * mu;  //for invert twist (not daggered)
      double b = 1.0 / (1.0 + a*a);                     //for invert twist
      if (!dagger || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	twistedCloverDslashCuda(&out, gauge, clover, cloverInv, &in, parity, dagger, 0, QUDA_DEG_DSLASH_TWIST_CLOVER_INV, a, b, 0.0, 0.0, commDim, profile);
	flops += 1392ll*in.Volume();                          
      } else {                                                
	twistedCloverDslashCuda(&out, gauge, clover, cloverInv, &in, parity, dagger, 0, QUDA_DEG_TWIST_CLOVER_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);	
        flops += 1392ll*in.Volume();
      }
    } else {//TWIST doublet :
        errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }
  }

  // xpay version of the above
  void DiracTwistedCloverPC::DslashXpay
  (cudaColorSpinorField &out, const cudaColorSpinorField &in, const QudaParity parity, const cudaColorSpinorField &x, const double &k) const
  {
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);
    if (in.TwistFlavor() != out.TwistFlavor()) 
      errorQuda("Twist flavors %d %d don't match", in.TwistFlavor(), out.TwistFlavor());
    if (in.TwistFlavor() == QUDA_TWIST_NO || in.TwistFlavor() == QUDA_TWIST_INVALID)
      errorQuda("Twist flavor not set %d\n", in.TwistFlavor());

   setFace(face); // FIXME: temporary hack maintain C linkage for dslashCuda
   initConstants(in);  
  
    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      double a = -2.0 * kappa * in.TwistFlavor() * mu;  //for invert twist
      double b = k / (1.0 + a*a);                     //for invert twist 
      if (!dagger) {
        twistedCloverDslashCuda(&out, gauge, clover, cloverInv, &in, parity, dagger, &x, QUDA_DEG_DSLASH_TWIST_CLOVER_INV, a, b, 0.0, 0.0, commDim, profile);
        flops += 1416ll*in.Volume();
      } else { // tmp1 can alias in, but tmp2 can alias x so must not use this
        twistedCloverDslashCuda(&out, gauge, clover, cloverInv, &in, parity, dagger, &x, QUDA_DEG_TWIST_CLOVER_INV_DSLASH, a, b, 0.0, 0.0, commDim, profile);
        flops += 1416ll*in.Volume();
      }
    } else {//TWIST_DOUBLET:
        errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }
  }

  void DiracTwistedCloverPC::M(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    double kappa2 = -kappa*kappa;

    bool reset = newTmp(&tmp1, in);

    if(in.TwistFlavor() == QUDA_TWIST_PLUS || in.TwistFlavor() == QUDA_TWIST_MINUS){
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
	  DslashXpay(out, *tmp1, QUDA_EVEN_PARITY, in, kappa2); 
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
	  DslashXpay(out, *tmp1, QUDA_ODD_PARITY, in, kappa2); 
      } else {//asymmetric preconditioning 
        double a = 2.0 * kappa * in.TwistFlavor() * mu;
        if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_ODD_PARITY);
          twistedCloverDslashCuda(&out, gauge, clover, cloverInv, tmp1, QUDA_EVEN_PARITY, dagger, &in, QUDA_DEG_DSLASH_TWIST_CLOVER_XPAY, a, kappa2, 0.0, 0.0, commDim, profile); 
          flops += (1320ll+96ll)*in.Volume();	 
        } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
	  Dslash(*tmp1, in, QUDA_EVEN_PARITY);
          twistedCloverDslashCuda(&out, gauge, clover, cloverInv, tmp1, QUDA_ODD_PARITY, dagger, &in, QUDA_DEG_DSLASH_TWIST_CLOVER_XPAY, a, kappa2, 0.0, 0.0, commDim, profile);
          flops += (1320ll+96ll)*in.Volume();
        }else { // symmetric preconditioning
          errorQuda("Invalid matpcType");
        }
      }
    } else { //Twist doublet
        errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }
    deleteTmp(&tmp1, reset);
  }

  void DiracTwistedCloverPC::MdagM(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    // need extra temporary because of symmetric preconditioning dagger
    bool reset = newTmp(&tmp2, in);
    M(*tmp2, in);
    Mdag(out, *tmp2);
    deleteTmp(&tmp2, reset);
  }

  void DiracTwistedCloverPC::prepare(cudaColorSpinorField* &src, cudaColorSpinorField* &sol,
				   cudaColorSpinorField &x, cudaColorSpinorField &b, 
				   const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      src = &b;
      sol = &x;
      return;
    }

    bool reset = newTmp(&tmp1, b.Even());
  
    // we desire solution to full system
    if(b.TwistFlavor() == QUDA_TWIST_PLUS || b.TwistFlavor() == QUDA_TWIST_MINUS){  
      if (matpcType == QUDA_MATPC_EVEN_EVEN) {
        // src = A_ee^-1 (b_e + k D_eo A_oo^-1 b_o)
        src = &(x.Odd());
        TwistCloverInv(*src, b.Odd());
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_EVEN_PARITY, b.Even(), kappa);
        TwistCloverInv(*src, *tmp1);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD) {
        // src = A_oo^-1 (b_o + k D_oe A_ee^-1 b_e)
        src = &(x.Even());
        TwistCloverInv(*src, b.Even());
        DiracWilson::DslashXpay(*tmp1, *src, QUDA_ODD_PARITY, b.Odd(), kappa);
        TwistCloverInv(*src, *tmp1);
        sol = &(x.Odd());
      } else if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // src = b_e + k D_eo A_oo^-1 b_o
        src = &(x.Odd());
        TwistCloverInv(*tmp1, b.Odd()); // safe even when *tmp1 = b.odd
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_EVEN_PARITY, b.Even(), kappa);
        sol = &(x.Even());
      } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // src = b_o + k D_oe A_ee^-1 b_e
        src = &(x.Even());
        TwistCloverInv(*tmp1, b.Even()); // safe even when *tmp1 = b.even
        DiracWilson::DslashXpay(*src, *tmp1, QUDA_ODD_PARITY, b.Odd(), kappa);
        sol = &(x.Odd());
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }
    } else {//doublet:
        errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }//end of doublet
    // here we use final solution to store parity solution and parity source
    // b is now up for grabs if we want

    deleteTmp(&tmp1, reset);
  }
  
  void DiracTwistedCloverPC::reconstruct(cudaColorSpinorField &x, const cudaColorSpinorField &b,
				       const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      return;
    }				

    checkFullSpinor(x, b);
    bool reset = newTmp(&tmp1, b.Even());

    // create full solution
    if(b.TwistFlavor() == QUDA_TWIST_PLUS || b.TwistFlavor() == QUDA_TWIST_MINUS){    
      if (matpcType == QUDA_MATPC_EVEN_EVEN || matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
        // x_o = A_oo^-1 (b_o + k D_oe x_e)
        DiracWilson::DslashXpay(*tmp1, x.Even(), QUDA_ODD_PARITY, b.Odd(), kappa);
        TwistCloverInv(x.Odd(), *tmp1);
      } else if (matpcType == QUDA_MATPC_ODD_ODD ||   matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
        // x_e = A_ee^-1 (b_e + k D_eo x_o)
        DiracWilson::DslashXpay(*tmp1, x.Odd(), QUDA_EVEN_PARITY, b.Even(), kappa);
        TwistCloverInv(x.Even(), *tmp1);
      } else {
        errorQuda("MatPCType %d not valid for DiracTwistedCloverPC", matpcType);
      }
    } else { //twist doublet:
        errorQuda("Non-degenrate DiracTwistedCloverPC is not implemented \n");
    }//end of twist doublet...
    deleteTmp(&tmp1, reset);
  }
} // namespace quda
