//
//  H_jumps.h
// generate the Hamiltonian and jumps for TI model with decay
//
//  Created by Mathias on 20/04/2020.
//
#include <vector>

#include "TensorFunctions.h"
#include "TGate.h"

#ifndef jumps_H_h
#define jumps_H_h

AutoMPO H_TI(SiteSet& sites, double J, double hz=0, double hx=0 ){
    auto H = AutoMPO(sites);
    
    // Sx interaction
    for (int b=1; b<sites.N(); b++){
        H += -J, "Sx",b,"Sx",b+1;
    }
    
    // Magnetic fields
    if (hz>1E-8){
        for (int b=1; b<=sites.N(); b++){
            H += hz, "Sz";
        }
    }
    if (hx>1E-8){
        for (int b=1; b<=sites.N(); b++){
            H += hx, "Sx";
        }
    }
    
    return H;
}

vector<OpGate> Decay(SiteSet sites, double gamma){
    vector<OpGate> C;
    
    for (int b=1; b<=sites.N(); b++){
        C.push_back( 
            TGate<ITensor>(sqrt(gamma)*sites.op("S-",b), b, 1)
            );
    }
    
    return C;
}

#endif