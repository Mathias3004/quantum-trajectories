#ifndef TGateQT_h
#define TGateQT_h

# include "itensor/all.h"
# include "itensor/util/parallel.h"
# include "TGate.h"
# include <vector>
# include <mpi.h>

using namespace itensor;
using namespace std;

template <class T>
class TGateQT
{
public:
  TGateQT(){}; // default

  TGateQT( double dt,
    vector<TGate<T>> C_gates,
    MPSt<T> psi_init,
    Args arg,
    Environment const& env);

  TGateQT( double dt,
    vector<TGate<T>> U_gates,
    vector<TGate<T>> C_gates,
    MPSt<T> psi_init,
    Args arg,
    Environment const& env,
    bool nonHerm);

  TGateQT( double dt,
    MPOt<T> U,
    vector<TGate<T>> C_gates,
    MPSt<T> psi_init,
    Args arg,
    Environment const& env,
    bool nonHerm);

      // some public functions
      void Run_MC_trajectory(vector<double> times, string type_TE, bool seperate_files);

      // evaluate expectation values
      void Track_O(vector<MPOt<T>> O);

      // evaluate entropies
      void Track_S(vector<int> S_sites);
      
      // other Renyi entropies
      void Track_Renyi(vector<int> S_sites, vector<int> alpha);
      
        // evaluate single-site entropies
      void Track_S_single(vector<int> S_sites);

      // update maximum bond dimension of current mps
      void Track_MBD();

      // collect jump times and whiches
      void Track_tj();

      // save MPS
      void Track_phi();

      // append data to file or start new
      void Set_file(string filename, bool append);

      // verbose output
      void Set_verbose(bool v);

    private:
      // MPS
      Args args_;
      MPSt<T> psi_;
      int N_site_;


      // time evolution
      double dt_; // differential time step

      vector<TGate<T>> C_gates_; // jump operators
      vector<TGate<T>> CC_gates_; // jump operators time hermitian conjugate
      vector<TGate<T>> HC_gates_; // nh H gates
      int N_Cj_; // number of jumps

      int type_H_; // 0: no H, 1:TGates, 2:MPO
      bool nonHerm_; // If the given H also contains anti Herm part from dissipation
      MPOt<T> U_; // time evolution from H
      vector<TGate<T>> U_gates_; // TGates for TE

      double T_; // actual time

      // expectation values
      vector<vector<double>> exp_O_; //expectation value observables
      vector<vector<double>> exp_S_; //entropies
      vector<vector<vector<double>>> exp_Renyi_; //Renyi entropies
      vector<vector<double>> exp_S_single_; //entropies single
      vector<int> MBD_; //Maximum bond dimension

      // Multicore parameters
      Environment const* env_;
      int MPI_rank_;
      int MPI_nnodes_;

      // verbose
      bool v_;

      // track these
      bool track_O_;
      bool track_S_;
      bool track_Renyi_;
      bool track_S_single_;
      bool track_MBD_;
      bool track_tj_;
      bool track_phi_;

      // vectors for observables
      vector<MPOt<T>> O_;
      vector<int> Si_;
      vector<int> Renyii_;
      vector<int> alpha_R_; //The Renyi powers
      vector<int> Si_single_;

      // waiting times and whiches
      vector<vector<double>> list_tj_;

      // filenames to dump
      string filename_;

      // check for append
      bool append_;

      /* private functions */

      // Evolve state over time interval t
      void evolve(double t, string type_TE);

      // differential time step
      void time_step_SJ(); // only single jumps
      void time_step_MJ(); // multiple jumps allowed

      // gather the data
      void gather_data();

      // collect different things
      void collect_O();
      void collect_S();
      void collect_Renyi();
      void collect_S_single();
      void collect_MBD();

      // collect and write data to file
      void dump_data(vector<vector<double>> data, string filename, bool append, bool seperate_files);


    };



    #endif /* TGateQT_h */
