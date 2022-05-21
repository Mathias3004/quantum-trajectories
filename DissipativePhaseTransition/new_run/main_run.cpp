#include <stdio.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>

# include "itensor/all.h"
#include "TGateQT.h"
#include "boson.h"
#include "../jumps.h"

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
    
    auto N_site = atoi(argv[2]);
    auto gamma = atof(argv[3]);

    //MPS parameters
    auto cutoff = input.getReal("cutoff",1E-8);
    auto Maxm = input.getInt("Maxm",50);

    //system parameters
   //auto N_site = input.getInt("N_site");
    auto N_particle = input.getInt("N_particle", N_site);
    auto nfock = input.getInt("nfock",6);

    //  model parameters
   //auto gamma = input.getReal("gamma",1.0);
    auto competition = input.getString("competition", "Dephasing");
    auto periodic = input.getInt("periodic", 0);

    //Quantum Trajectory parameters
    auto t_init = input.getReal("t_init", 0.);
    auto t_ev = input.getReal("t_ev",1.);
    auto dt = input.getReal("dt",1E-3) / max(1.,gamma);
    auto type_TE = input.getString("type_TE", "Multiple");
    auto N_sample = input.getInt("N_sample");
    auto N_loop = input.getInt("N_loop", 1);

    //outputfiles
    bool verbose = input.getInt("verbose",0) > 0;
    bool separate_files = input.getInt("verbose",0) > 0;
    auto output = input.getString("output","data") + "_" + to_string(N_site) + "_" + to_string(gamma);
    auto append_i = input.getInt("append",1);
    bool append = (append_i > 0);

    // output
    if (env.rank() == 0){
      cout << "\n\n*****************************************************************\n";
      cout << "\nRunning system with gamma=";
      cout << gamma;
      cout << " for " << N_site << " sites and local Fock space of " << nfock << " maximal occupancy\n";
      cout << env.nnodes() << " cores with " << N_loop << " loops, resulting in " << env.nnodes()*N_loop << "trajectories with sampling of "  << N_sample << endl;
      cout << "*****************************************************************\n\n\n";
    }

    //---------------------------------------------------------
    //----setup Hamiltonian, initial psi, jump operator.-------

    // Make args
    auto args = Args("Cutoff=",cutoff,"Maxm=",Maxm);
    args.add("nfock=",nfock);

    //Setup Siteset
    auto sites = Boson(N_site, args);

    //-----------------------------------------------------
    //---------Initial Condition---------------------------
    // initial state is product state with correct number of particles

    auto state = InitState(sites);
    int particles_per_site = floor( N_particle/ N_site);
    int particles_left = N_particle - N_site*particles_per_site;
    for(int i = 1; i <= N_site; ++i)
    {
        if (i<=particles_left){
        state.set(i,to_string(particles_per_site+1));
        }
        else{
            state.set(i,to_string(particles_per_site));
        }
    }


    auto psi_init = IQMPS(state);

    // The gates for the jumps
    auto C_gates = create_Cj(sites, gamma, periodic, competition);
    
    cout << "Number of jumps:" << C_gates.size() << endl;

    // the time vector
    vector<double> t_vec;
    for(int i=0; i<N_sample; ++i){
      t_vec.push_back(t_init + i*t_ev);
    }


    //-----------------------------------------------------
    //-------- Observable ---------------------------------
    // first order correlations through lattice
    vector<IQMPO> O;

    // correlations between all sites
    for (int i=1; i<N_site; ++i){
        auto ampo = AutoMPO(sites);
        ampo += "ad",i,"a",i+1;
        O.push_back(IQMPO(ampo));
        
    }

    // include all sites for entropy
    vector<int> S_i;
    for (int i=1; i<N_site; i++)
        S_i.push_back(i);
        
    vector<int> alpha{ 1, 2, 3 };


    //------------------------------------------------------
    // Run Quantum trajectory (make separate function for MPI to loop over)
    
    time_t t_start = time(NULL);

    for (int it = 0; it < N_loop; it++){
      // check if data need to be appended
      bool do_append = true;
      if (it==0) do_append = append;

      // make  class instance
      TGateQT<IQTensor> t_qt(dt, C_gates, psi_init, args, env);
      t_qt.Set_verbose(verbose);
      t_qt.Set_file(output, do_append);
      t_qt.Track_O(O);
      t_qt.Track_Renyi(S_i,alpha);
      t_qt.Track_MBD();
      //t_qt.Track_tj();

      t_qt.Run_MC_trajectory(t_vec, type_TE,separate_files);


    }

    // generate info file of simulation for analysis later
    if (env.rank()==0){
      string filename_info = output + "_info" + ".txt";
      ofstream myfile;
      myfile.open(filename_info);

      myfile << "gamma=" << gamma << endl;
      myfile << "competition=" << competition << endl;
      myfile << "type_TE=" << type_TE << endl;
      myfile << "N_site=" << N_site << endl;
      myfile << "t_init=" << t_init << endl;
      myfile << "t_ev=" << t_ev << endl;
      myfile << "dt=" << dt << endl;
      myfile << "N_core=" << env.nnodes() << endl;
      myfile << "N_loop=" << N_loop << endl;
      myfile << "N_sample=" << N_sample << endl;
    }
    
    cout << "\n\n\n";
    cout << "************************************************************\n";
    cout << "Total runtime: " << time(NULL) - t_start << "s\n";
    cout << "************************************************************\n\n\n";
    
    return 0;

}
