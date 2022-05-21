#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>

# include "itensor/all.h"
#include "TGateQT.h"
#include "H_jumps.h"

using namespace std;
using namespace itensor;

int main ( int argc, char *argv[] )
{

    //---------------------------------------------------------
    //----------------------initialize MPI---------------------
    Environment env(argc,argv);

    //---------------------------------------------------------
    //---------parse parameters from input file ---------------
    auto input = InputGroup(argv[1],"input");

    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",50);

    //system parameters
    auto N_site = input.getInt("N_site");
    
    //  model parameters
    auto J = input.getReal("J",1.0);
    auto gamma = input.getReal("gamma",1.0);
    
    //Quantum Trajectory parameters
    auto t_init = input.getReal("t_init", 0.);
    auto t_ev = input.getReal("t_ev",1.);
    auto dt = input.getReal("dt",1E-3);
    auto type_TE = input.getString("type_TE", "Multiple");
    auto N_sample = input.getInt("N_sample");
    auto N_loop = input.getInt("N_loop", 1);
    
    //outputfiles
    auto output = input.getString("output","data");
    auto append_i = input.getInt("append",1);
    bool append = (append_i > 0);

    // output
    if (env.rank() == 0){
      cout << "\n\n*****************************************************************\n";
      cout << "\nRunning system with gamma=";
      cout << gamma;
      cout << " and J=" << J;
      cout << " for " << N_site << " sites\n";
      cout << env.nnodes() << " cores with " << N_loop << " loops, resulting in " << env.nnodes()*N_loop << "trajectories with sampling of "  << N_sample << endl;
      cout << "*****************************************************************\n\n\n";
    }
    
    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    
    //Setup Siteset
    auto sites = SpinHalf(N_site, args);
    
    //-----------------------------------------------------
    //---------Initial Condition---------------------------
    // initial state is product state with correct number of particles

    auto state = InitState(sites);
 
    for(int i = 1; i <= N_site; ++i)
    {
        state.set(i,"Up");
    }
    auto psi_init = MPS(state);
    
    //-----------------------------------------------------
    //-------- Observable ---------------------------------
    // first order correlations through lattice
    vector<MPO> O;

    // correlations between all sites
    for (int i=1; i<=N_site; ++i){
        auto ampo = AutoMPO(sites);
        ampo += "Sz",i;
        O.push_back(MPO(ampo));
        
    }

    // include all sites for entropy
    vector<int> S_i;
    for (int i=1; i<N_site; i++)
        S_i.push_back(i);
        
        
        
    //------------------------------------------------------
    // TI Hamiltonian and decay jumps
    //------------------------------------------------------
    
    auto H = H_TI(sites,J);
    auto U = toExpH<ITensor>(H,Cplx_i*dt);
    
    auto C_gates = Decay(sites,gamma);
    
        
    //------------------------------------------------------
    // Run Quantum trajectory
    //------------------------------------------------------
    
    
    // the time vector
    vector<double> t_vec;
    for(int i=0; i<N_sample; ++i){
      t_vec.push_back(t_init + i*t_ev);
    }

    for (int it = 0; it < N_loop; it++){
      // check if data need to be appended
      bool do_append = true;
      if (it==0) do_append = append;

      // make  class instance
      TGateQT<ITensor> t_qt(dt, U, C_gates, psi_init, args, env);
      t_qt.Set_verbose(true);
      t_qt.Set_file(output, do_append);
      t_qt.Track_O(O);
      t_qt.Track_S(S_i);
      t_qt.Track_MBD();
      t_qt.Track_tj();

      t_qt.Run_MC_trajectory(t_vec, type_TE,true);

    }

    // generate info file of simulation for analysis later
    if (env.rank()==0){
      string filename_info = output + "_info" + ".txt";
      ofstream myfile;
      myfile.open(filename_info);

      myfile << "gamma=" << gamma << endl;
      myfile << "J=" << J << endl;
      myfile << "type_TE=" << type_TE << endl;
      myfile << "N_site=" << N_site << endl;
      myfile << "t_init=" << t_init << endl;
      myfile << "t_ev=" << t_ev << endl;
      myfile << "dt=" << dt << endl;
      myfile << "N_core=" << env.nnodes() << endl;
      myfile << "N_loop=" << N_loop << endl;
      myfile << "N_sample=" << N_sample << endl;
    }

    return 0;

}
    
    
    
    
    