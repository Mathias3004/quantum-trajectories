//
//  H_and_C.h
//
//
//  Created by Mathias on 20/04/2020.
//
#include <vector>

#include "boson.h"
#include "TensorFunctions.h"
#include "TGate.h"

#ifndef jumps_h
#define jumps_h


// Ze-Pei's code for the TGates

IQTensor local_jump_BEC(SiteSet& sites, int j, double kappa)
    {
    int L = j;
    int R = j+1;
    // useful operators
    IQTensor bpL = sites.op("b+",L) * sites.op("Id" ,R); // b+_L
    IQTensor bpR = sites.op("Id" ,L) * sites.op("b+",R); // b+_R
    IQTensor bmL = sites.op("b-",L) * sites.op("Id" ,R); // b-_L
    IQTensor bmR = sites.op("Id" ,L) * sites.op("b-",R); // b-_R
    auto LplsR_p = bpL+bpR; // (b+_L + b+_R)
    auto LmnsR_m = bmL-bmR; // (b-_L - b-_R)
    IQTensor l;
    nmultITensor(LmnsR_m, LplsR_p, l);
    l *= sqrt(kappa);
    return l;
    }

IQTensor local_jump_dephasing(SiteSet& sites, int j, double kappa)
    {
    return sqrt(kappa)*sites.op("N",j);
    }

IQTensor local_jump_repulsion(SiteSet& sites, int j, double kappa)
    {
    return sqrt(kappa)*sites.op("b+",j)*sites.op("b-^2",j+1)*sites.op("b+",j+2);
    }


// generate the jump operators
vector<IQOpGate> create_Cj(Boson sites,
  double gamma,
  bool periodic = 0,
  string competition = "Dephasing")
{
    int N_site = sites.N();
    vector<IQOpGate> C;

    // add BEC jumps

    for (int i=1; i<N_site; i+=2){
      auto cj = local_jump_BEC(sites, i, 1.);
      auto g = TGate<IQTensor>(cj, i, 2);
      C.push_back(g);
    }
    for (int i=2; i<N_site; i+=2){
      auto cj = local_jump_BEC(sites, i, 1.);
      auto g = TGate<IQTensor>(cj, i, 2);
      C.push_back(g);
    }
  

    // add jumps
    if (gamma > 1E-8){
        if (competition=="Dephasing"){
        for (int i=N_site; i>=1; i--){
            auto cj = local_jump_dephasing(sites, i, gamma);
            auto g = TGate<IQTensor>(cj, i, 1);
            C.push_back(g);
        }
    }
    else if (competition=="BoseHubbard"){
      // add jumps for BH model
    }
    else{
        cout << "No valid competition chosen, going with dephasing\n\n";
        C = create_Cj(sites, gamma, periodic = 0, "Dephasing");
    }
  }
  return C;
}






#endif /* jumps_h */
