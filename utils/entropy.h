
#include "itensor/all.h"
#include "itensor/util/parallel.h"
#include "boson.h"

using namespace std;
using namespace itensor;

#ifndef entropy
#define entropy

double S_vn(IQMPS &psi, int b, int alpha=1)
{
    if (b==0 || b==psi.N()){
        return 0.;
    }
 
    
    //"Gauge" the MPS to site b
    psi.position(b);
    
    //Compute two-site wavefunction for sites (b,b+1)

    auto wf = toITensor(psi.A(b)*psi.A(b+1));
    auto U = toITensor(psi.A(b));
    
    ITensor S,V;
    auto spectrum = svd(wf,U,S,V);
     
    double ent = 0.;   
    for(auto p : spectrum.eigs())
    {
        if(p > 1E-12){
            if (alpha == 1){
                ent += -p*log(p);}
            else{
                ent += pow(p,alpha);
            }
        }
    }
    
    // rescale and logarithm
    if (alpha>1){
        ent = 1./(1.-alpha)*log(ent);
    }
    
    return ent;
}

double S_vn(MPS &psi, int b, int alpha=1)
{
    if (b==0 || b==psi.N()){
        return 0.;
    }
 
    
    //"Gauge" the MPS to site b
    psi.position(b);
    
    //Compute two-site wavefunction for sites (b,b+1)

    auto wf = psi.A(b)*psi.A(b+1);
    auto U = psi.A(b);
    
    ITensor S,V;
    auto spectrum = svd(wf,U,S,V);
     
    double ent = 0.;   
    for(auto p : spectrum.eigs())
    {
        if(p > 1E-12){
            if (alpha == 1){
                ent += -p*log(p);}
            else{
                ent += pow(p,alpha);
            }
        }
    }
    
    // rescale and logarithm
    if (alpha>1){
        ent = 1./(1.-alpha)*log(ent);
    }
    
    return ent;
}

/*double S_vn(MPS &psi, int b)
{
    if (b==0 || b==psi.N()){
        return 0.;
    }
 
    
    //"Gauge" the MPS to site b
    psi.position(b);
    
    //Compute two-site wavefunction for sites (b,b+1)

    auto wf = psi.A(b)*psi.A(b+1);
    auto U = psi.A(b);
    
    ITensor S,V;
    auto spectrum = svd(wf,U,S,V);
     
    double ent = 0.;   
    for(auto p : spectrum.eigs())
    {
        if(p > 1E-12){
            ent += -p*log(p);
        }
    }
    
    return ent;
}*/

double S_vn_interval(IQMPS &psi, int b1, int b2){

    
    //"Gauge" the MPS to site b
    psi.position(b1);
    
    auto B = psi.A(b1);
    
    for (int j=0; j < b2-b1; j++){
        B *= psi.A(b1+j+1);
    }
    
    auto rho = toITensor(prime(B,Site) * dag(B));
    
    ITensor U,D;
    auto spectrum = diagHermitian(rho,U,D);
    
    double ent = 0.;   
    for(auto p : spectrum.eigs())
    {
        if(p > 1E-12){
            ent += -p*log(p);
        }
    }
    
    return ent;
    
}

double S_vn_interval(MPS &psi, int b1, int b2){

    
    //"Gauge" the MPS to site b
    psi.position(b1);
    
    auto B = psi.A(b1);
    
    for (int j=0; j < b2-b1; j++){
        B *= psi.A(b1+j+1);
    }
    
    auto rho = prime(B,Site) * dag(B);
    
    ITensor U,D;
    auto spectrum = diagHermitian(rho,U,D);
    
    double ent = 0.;   
    for(auto p : spectrum.eigs())
    {
        if(p > 1E-12){
            ent += -p*log(p);
        }
    }
    
    return ent;
    
}

template<class T>
double S_vn_single_site(T &psi, int b){
    return S_vn_interval(psi,b,b);
}
    
    

#endif