//
//  H_jumps.h
// generate the Hamiltonian and jumps for TI model with decay
//
//  Created by Mathias on 20/04/2020.
//
#include <vector>

# include "itensor/all.h"
#include "TensorFunctions.h"
#include "TGate.h"

#ifndef jumps_H_h
#define jumps_H_h

AutoMPO H_drive(SiteSet sites, double delta, double Omega, double gamma ){
    auto H = AutoMPO(sites);
    
    // Sx interaction
    for (int b=1; b<=sites.N(); b++){
        H += -delta, "Sz",b;
        H += Omega, "Sx",b;
        
        // anti Herm part
        H += -Cplx_i*gamma/2.0, "projUp",b;
    }
    
    return H;
}

vector<OpGate> Decay(SiteSet sites, double gamma, double p, bool periodic=false){
    vector<OpGate> C;
    
    // coeff of BS jumps
    double coeff_BS = 0.5*sqrt((1.-p)*gamma);
    
    for (int b=1; b<sites.N(); b++){
        
        // direct measurement: probability p
        C.push_back( 
            TGate<ITensor>(sqrt(p*gamma)*sites.op("S-",b), b, 1)
            );
        
        // BS measurement: probability 1-p
        ITensor l = sites.op("S-",b) * sites.op("Id" ,b+1);
        ITensor r = sites.op("Id" ,b) * sites.op("S-",b+1);
        C.push_back(TGate<ITensor>( coeff_BS*(l+r), b, 2 ));
        C.push_back(TGate<ITensor>( coeff_BS*(l-r), b, 2 ));
    }
    
    // direct measurement last site
    C.push_back( 
        TGate<ITensor>(sqrt(p*gamma)*sites.op("S-",sites.N()), sites.N(), 1)
        );
    
    //for the boundaries
    C.push_back( 
        TGate<ITensor>(sqrt(0.5*(1.-p)*gamma)*sites.op("S-",1), 1, 1)
        );
        
    C.push_back( 
        TGate<ITensor>(sqrt(0.5*(1.-p)*gamma)*sites.op("S-",sites.N()), sites.N(), 1)
    );
    
    return C;
}

vector<OpGate> Decay_brick(SiteSet sites, double gamma, double p, bool periodic=false){
    vector<OpGate> C;
    
    for (int b=1; b<=sites.N();b++){
        // direct measurement: probability p
        C.push_back( 
            TGate<ITensor>(sqrt(p*gamma)*sites.op("S-",b), b, 1)
            );
    }
    
    ITensor c1, c2, c3, c4;
    
    //for the left boundary
    c1 = sites.op("S-",1) * sites.op("Id" ,2);
    c2 = sites.op("Id",1) * sites.op("S-" ,2);
    C.push_back(
        TGate<ITensor>(sqrt(0.5*(1.-p)*gamma)*(c1-c2), 1, 2)
        );
    
    // coeff of BS jumps
    double coeff_BS = 0.5*sqrt((1.-p)*gamma);
    
    for (int b=1; b<=sites.N()-3; b+=2){
        
        // BS measurement: probability 1-p
        c1 = sites.op("S-",b) * sites.op("Id" ,b+1) * sites.op("Id" ,b+2) * sites.op("Id" ,b+3);
        c2 = sites.op("Id",b) * sites.op("S-" ,b+1) * sites.op("Id" ,b+2) * sites.op("Id" ,b+3);
        c3 = sites.op("Id",b) * sites.op("Id" ,b+1) * sites.op("S-" ,b+2) * sites.op("Id" ,b+3);
        c4 = sites.op("Id",b) * sites.op("Id" ,b+1) * sites.op("Id" ,b+2) * sites.op("S-" ,b+3);

        C.push_back(TGate<ITensor>( coeff_BS*(c1+c2-c3+c4), b, 4 ));
        C.push_back(TGate<ITensor>( coeff_BS*(c1+c2+c3-c4), b, 4 ));
    }
    
    //for the right boundary
    c1 = sites.op("S-",sites.N()-1) * sites.op("Id" ,sites.N());
    c2 = sites.op("Id",sites.N()-1) * sites.op("S-" ,sites.N());
    C.push_back(
        TGate<ITensor>(sqrt(0.5*(1.-p)*gamma)*(c1+c2), sites.N()-1, 2)
        );
        
    
    return C;
}

#endif