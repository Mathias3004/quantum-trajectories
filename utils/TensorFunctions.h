#ifndef __TENSORFUNCTIONS__
#define __TENSORFUNCTIONS__

# include "itensor/all.h"
using namespace itensor;


template<class T> T
HermConj_Tensor(T M)
    {
    auto HM = dag(M);
    HM.mapprime(1,2,Site);
    HM.mapprime(0,1,Site);
    HM.mapprime(2,0,Site);
    return HM;
    }//HermConj

template<class T>
void //C = BA
nmultITensor(T A, T B, T& C)
    {
    auto B_ = B;
    B_.prime();
    C = B_*A;
    C.mapprime(2,1);
    }

/*double get_entropy(MPSt<T> psi, int b){

    //"Gauge" the MPS to site b
    psi.position(b);

    //Compute two-site wavefunction for sites (b,b+1)

    auto wf = toITensor(psi.A(b)*psi.A(b+1));
    auto U = toITensor(psi.A(b));

    ITensor S,V;
    auto spectrum = svd(wf,U,S,V);

    double EE = 0.;
    for(auto p : spectrum.eigs()){
      if(p > 1E-12) EE += -p*log(p);
    }
    return EE;
  } */



#endif
