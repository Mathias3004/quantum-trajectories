#ifndef TGATE_H
#define TGATE_H

# include "itensor/all.h"
using namespace itensor;

template <class T> struct TGate;
using OpGate = TGate<ITensor>;
using IQOpGate = TGate<IQTensor>;




template <class T>
struct TGate
    {
    int p ;
    int nsites ;
    T G;
    TGate() { }
    TGate(T G_, int p_, int nsites_ = 1):p(p_), nsites(nsites_), G(G_){}
    };

template<class T>
void ExpTGate(TGate<T>& gate, SiteSet const& sites, Cplx tau, int order = 5)
    {
    auto bondH = gate.G;

    bondH *= -Complex_i*tau;
    T unit = sites.op("Id",gate.p);
    for(int i = 1; i < gate.nsites; i++)
        {
        unit *= sites.op("Id", gate.p+i);
        }


    auto term = bondH;
    bondH.mapprime(1,2);
    bondH.mapprime(0,1);

    // exp(x) = 1 + x +  x^2/2! + x^3/3! ..
    // = 1 + x * (1 + x/2 *(1 + x/3 * (...
    // ~ ((x/3 + 1) * x/2 + 1) * x + 1
    for(int ord = order; ord >= 1; --ord)
        {
        term /= ord;
        gate.G = unit + term;
        term = gate.G * bondH;
        term.mapprime(2,1);
        }
    }
template<class T>
void ApplyTGate(MPSt<T>& psi, const TGate<T>& gate, Args args)
    {

    auto& sites = psi.sites();
    auto& p = gate.p;
    auto& G = gate.G;
    
    psi.position(p);
    psi.normalize();
    
    auto A = psi.A(p);
    for(int i = 1; i < gate.nsites; i++)
        {
        A *= psi.A(p+i);
        }
    A = A*G;
    A.mapprime(1,0,Site);

    if(gate.nsites == 1) psi.Aref(p) = A;
    else
        {
        for(int i = 0; i < gate.nsites-1; i++)
            {
            T S, V;
            svd(A, psi.Aref(p+i), S, V, args);
            V *= S;
            if( i == gate.nsites-2)
                {
                psi.Aref(p+i+1) = V;
                break;
                }
            A = V;
            T tem(commonIndex(A, psi.A(p+i), Link), sites(p+i+1));
            psi.Aref(p+i+1) = tem;
            }
        }
    }

template <class Iterable, class T>
void
ApplyTGates(MPSt<T>& psi, Iterable const& gatelist, Args args)
    {
    for(auto gate = gatelist.begin(); gate != gatelist.end(); gate++)
        {
        ApplyTGate<T>(psi, *gate, args);
        }
    }

template<class T>
double
OverlapTGate(MPSt<T>& psi, const TGate<T>& gate)
    {
        
    psi.position(gate.p);
    psi.normalize();
    
    auto PSI = psi.A(gate.p);
    for(int i = 1; i < gate.nsites; i++)
        {
        PSI *= psi.A(gate.p+i);
        }
    auto hh = dag(prime(PSI,Site))*gate.G*PSI;
    return hh.real();
    }

#endif
