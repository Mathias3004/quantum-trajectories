#include <stdio.h>
#include <vector>
#include <iostream>
#include <type_traits>
#include <ctime>

#include "itensor/all.h"
#include "TGateQT.h"
#include "TensorFunctions.h"
#include "TGate.h"
#include "entropy.h"

using namespace itensor;
using namespace std;

/*----------------------------------
THE CONSTRUCTORS
 ----------------------------------*/

template<typename T>
TGateQT<T>::TGateQT( double dt,
  vector<TGate<T>> C_gates,
  MPSt<T> psi_init,
  Args arg,
  Environment const& env)
  : env_(&env){

  args_ = arg;
  args_.add("DoSVDBond", true);

  // MPI parameters
  MPI_rank_ = env_ -> rank();
  MPI_nnodes_ = env_ -> nnodes();

  // initial state
  normalize(psi_init);
  psi_ = psi_init; // initial state

  //system parameters
  dt_ = dt; // differential time step
  N_site_ = psi_init.N(); // number of sites

  // verbose
  v_ = false;


  T_ = 0.; //start time 0

  // setup jumps
  C_gates_ = C_gates;
  N_Cj_ = C_gates_.size();

  CC_gates_.resize(N_Cj_);
  HC_gates_.resize(N_Cj_);

  for (int i=0; i<N_Cj_; ++i){
    auto C_gate = C_gates_[i];
    auto C = C_gate.G;

    // CdagC
    auto Cdag = HermConj_Tensor(C);
    T CdagC;
    nmultITensor(C, Cdag, CdagC);

    // HC
    auto g = TGate<T>(-Cplx_i/2*CdagC, C_gate.p, C_gate.nsites);
    auto HC = g;
    ExpTGate(HC, psi_.sites(), dt);

    CC_gates_[i] = TGate<T>(CdagC, C_gate.p, C_gate.nsites);
    HC_gates_[i] = HC;
  }

  type_H_ = 0; // No herm H
  nonHerm_ = false; // Just Herm H

  //take a seed unique for every process
  srand(MPI_rank_*1000 + time(NULL));

  filename_ = "out";
  append_ = true;

  track_O_ = false;
  track_S_ = false;
  track_S_single_ = false;
  track_MBD_ = false;
  track_tj_ = false;
  track_phi_ = false;



}

template<typename T>
TGateQT<T>::TGateQT( double dt,
  vector<TGate<T>> U_gates,
  vector<TGate<T>> C_gates,
  MPSt<T> psi_init,
  Args arg,
  Environment const& env,
  bool nonHerm = false)
  : TGateQT<T>( dt, C_gates, psi_init, arg, env )
  {
    type_H_ = 1;
    U_gates_ = U_gates;
    nonHerm_ = nonHerm;
  }

template<typename T>
TGateQT<T>::TGateQT( double dt,
  MPOt<T> U,
  vector<TGate<T>> C_gates,
  MPSt<T> psi_init,
  Args arg,
  Environment const& env,
  bool nonHerm = false)
  : TGateQT<T>( dt, C_gates, psi_init, arg, env )
  {
    type_H_ = 2;
    U_ = U;
    nonHerm_ = nonHerm;
  }

/*--------------------------------------
OTHER PUBLIC FUNCTIONS
----------------------------------------*/

template<typename T>
void TGateQT<T>::Track_O(vector<MPOt<T>> O){
  track_O_ = true;
  O_ = O;
}

template<typename T>
void TGateQT<T>::Track_S(vector<int> Si){
  track_S_ = true;
  Si_ = Si;
}

template<typename T>
void TGateQT<T>::Track_Renyi(vector<int> Renyii, vector<int> alpha){
  track_Renyi_ = true;
  Renyii_ = Renyii;
  alpha_R_ = alpha;
  exp_Renyi_.resize(alpha.size());
  
}

template<typename T>
void TGateQT<T>::Track_S_single(vector<int> Si){
  track_S_single_ = true;
  Si_single_ = Si;
}

template<typename T>
void TGateQT<T>::Track_MBD(){
  track_MBD_ = true;
}

template<typename T>
void TGateQT<T>::Track_tj(){
  track_tj_ = true;
}

template<typename T>
void TGateQT<T>::Track_phi(){
  track_phi_ = true;
}

// Run a trajectory
template<typename T>
void TGateQT<T>::Run_MC_trajectory(vector<double> times, string type_TE, bool separate_files){

  // sort times, just to be sure
  sort(times.begin(), times.end());
  int nt = times.size();

  // if t=0 is not included, first evolve until start time
  if (times[0] > dt_) {
    evolve(times[0], type_TE);
  }

  // gather first data
  gather_data();
  

  // loop over times
  for (int it = 1; it < nt; it++){
      
    //take a seed unique for every process
    srand(it + MPI_rank_*1000 + time(NULL));
      
    // evolution time
    double t_ev = times[it] - times[it-1];
    // evolve
    evolve(t_ev, type_TE);
    
    // gather the data
    gather_data();
    
    
    // collect all nodes to finish
    env_ -> barrier();
  }
    //cout << env.rank() << "test 1" << endl;
  // collect all nodes to finish
  
//cout << env.rank() << "test 2" << endl;
  // dump data to corresponding files
  if (track_O_){
    auto fn = filename_ + "_O_";
    dump_data(exp_O_, fn, append_, separate_files);
  }
  if (track_S_){
    auto fn = filename_ + "_S_";
    dump_data(exp_S_, fn, append_, separate_files);
  }
  if (track_Renyi_){
      for (int i=0; i<alpha_R_.size(); i++){
        auto fn = filename_ + "_Renyi" + to_string(alpha_R_[i]) + "_";
        dump_data(exp_Renyi_[i], fn, append_, separate_files);
      }
  }
  if (track_S_single_){
    auto fn = filename_ + "_S_single_";
    dump_data(exp_S_single_, fn, append_, separate_files);
  }
  if (track_MBD_) {
    vector<vector<double>> v_MBD;
    v_MBD.resize(nt);
    for (int it = 0; it<nt; it++) v_MBD[it].push_back(MBD_[it]);
    auto fn = filename_ + "_MBD_";
    dump_data(v_MBD, fn, append_, separate_files);
  }

  if (track_tj_){
      cout << list_tj_.size() << endl;
    auto fn = filename_ + "_tj";
    dump_data(list_tj_, fn, append_, separate_files);
  }
}

// *******************************************
// PRIVATE functions
// ******************************************

// The time evollution (public)
template<typename T>
void TGateQT<T>::evolve(double t_ev, string type_TE){

  if (v_) cout << endl << endl << "Start time evolution over T=" << t_ev << "..." << endl;

  std::clock_t start = clock();

  // t_ev must be a multiple of dt_
  int n_steps = (int) round(t_ev/dt_);

  for (int it = 0; it<n_steps; ++it){
    if (type_TE == "Multiple"){
      time_step_MJ();
    }
    else if (type_TE == "Single"){
      time_step_SJ();
    }
    else{
      cout << "WARNING: no valid type_TE, Multiple is chosen\n\n";
      type_TE = "Multiple";
      time_step_MJ();
    }

    T_ += dt_;
  }

  double duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
  
  // maximal click probablity
  double pmax = 0.;
  for (size_t j=0; j < N_Cj_; j++)
  {
      double p = dt_ * real(OverlapTGate(psi_, CC_gates_[j]));
      if (p>pmax) pmax = p;
  }

  if (v_) cout << "Rank = "<< MPI_rank_ << " :TE over t_ev=" << dt_*n_steps << " determined until T=" << T_
    << ", runtime "<< duration <<"s, MBD=" << maxM(psi_) << ", max pj=" << pmax << endl;
    // << ", Pmax=" << Pmax << "\n";
}

// time steps (private)
template<typename T>
void TGateQT<T>::time_step_SJ(){

  double dp=0.;
  vector<double> dpj(N_Cj_);

  for (size_t j=0; j < N_Cj_; j++)
  {
    dpj[j] = dt_ * real(OverlapTGate(psi_, CC_gates_[j]));
    dp += dpj[j];
  }

  if(dp > 0.4)
  {
    cout << "Warning : dp = " << dp << ". Please decrease the time step dt." << endl;
  }

  // Generate randum number
  double e = ((double) rand() / (RAND_MAX));

  //No jump
  if (e>dp)
  {
    // H evolution
    if (type_H_ == 2){
      psi_ = exactApplyMPO(U_,psi_,args_);
    }
    else if (type_H_ == 1){
      ApplyTGates(psi_,U_gates_,args_); // Ze-Pei's function
    }

    // H_NH evolution, in case not in U yet
    if (!nonHerm_){
        for (int ig=0; ig<N_Cj_; ++ig){
          ApplyTGates(psi_,HC_gates_,args_);
        }
    }
  }

  // else apply jump
  else {
    // draw which jump is applied
    double e2 = ((double) rand() / (RAND_MAX));
    double sumpj=0.;
    int jjump=0;
    for (size_t j=0 ; j < dpj.size(); j++){
        sumpj += dpj[j]/dp;
        if (sumpj>e2){
            jjump=j;
            break;
        }
    }
    ApplyTGate(psi_, C_gates_[jjump], args_);
    if (track_tj_){
      vector<double> tj;
      tj.push_back((double)jjump);
      tj.push_back(T_);
      list_tj_.push_back(tj);
    }
  }

  // make sure state is normalized again
  psi_.position(1,args_);
  normalize(psi_);

}

template<typename T>
void TGateQT<T>::time_step_MJ(){

  // H evolution
  if (type_H_ == 2){
    psi_ = exactApplyMPO(U_,psi_,args_);
  }
  else if (type_H_ == 1){
    ApplyTGates(psi_,U_gates_,args_); // Ze-Pei's function
  }
  
    // make sure state is normalized again
    psi_.position(1,args_);
    normalize(psi_);
  
  // sample which jumps click
  vector<bool> click;
  click.resize(N_Cj_);
  /*for (size_t j=0; j < N_Cj_; j++)
  {
    double e = ((double) rand() / (RAND_MAX));
    click[j] = (dt_ * real(OverlapTGate(psi_, CC_gates_[j])) > e);
  }*/
  

  // apply jumps or evolve anti hermitian C^\daggerC if !nonHerm_
  for (size_t j=0; j < N_Cj_; j++)
  {
    double e = ((double) rand() / (RAND_MAX));
    click[j] = (dt_ * real(OverlapTGate(psi_, CC_gates_[j])) > e);
    
    if (click[j]){
      ApplyTGate(psi_, C_gates_[j], args_);
      if (track_tj_){
        vector<double> tj;
        tj.push_back((double)j);
        tj.push_back(T_);
        list_tj_.push_back(tj);
      }
    }
    else if (!click[j] && !nonHerm_) {
      ApplyTGate(psi_, HC_gates_[j], args_);
    }

  }
  // make sure state is normalized again
    psi_.position(1,args_);
    normalize(psi_);

}

// gather the data
template<typename T>
void TGateQT<T>::gather_data(){
  if (v_) cout << "Collecting data..." << endl;

  if (track_O_) collect_O();
  if (track_S_) collect_S();
  if (track_Renyi_) collect_Renyi();
  if (track_S_single_) collect_S_single();
  if (track_MBD_) collect_MBD();
}

// evaluate list of observables and save
template<typename T>
void TGateQT<T>::collect_O(){

  vector<double> exp_Ot;
  for (int n=0; n<O_.size(); n++){
    exp_Ot.push_back(
      real(overlapC(psi_, O_[n], psi_))
    );
  }
  exp_O_.push_back(exp_Ot);
}

// evaluate entropies at sites S_sites and save
template<typename T>
void TGateQT<T>::collect_S(){
    vector<double> exp_St;
    exp_St.resize(Si_.size());
    
    // loop over position
    for (int j=0; j<Si_.size(); j++){
        exp_St[j] = S_vn(psi_, Si_[j]);
    }
    
    exp_S_.push_back(exp_St);
}

// evaluate Renyi at sites S_sites and save
template<typename T>
void TGateQT<T>::collect_Renyi(){

    for (int i=0; i<alpha_R_.size(); i++){
        vector<double> exp_St;
        exp_St.resize(Renyii_.size());
    
        // loop over position
        for (int j=0; j<Renyii_.size(); j++){
            exp_St[j] = S_vn(psi_, Renyii_[j], alpha_R_[i]);
        }
        exp_Renyi_[i].push_back(exp_St);
    }
}

// evaluate single-site entropies at sites S_sites and save
template<typename T>
void TGateQT<T>::collect_S_single(){
    vector<double> exp_St;
    exp_St.resize(Si_.size());
    
    // loop over position
    for (int j=0; j<Si_.size(); j++){
        exp_St[j] = S_vn_single_site(psi_, Si_single_[j]);
    }
    
    exp_S_single_.push_back(exp_St);
}


// get maximal bond dimension of MPS and save
// UPDATE
template<typename T>
void TGateQT<T>::collect_MBD(){
  MBD_.push_back(maxM(psi_));
}

template<typename T>
void TGateQT<T>::dump_data(vector<vector<double>> data, string filename, bool append, bool separate_files){

  env_ -> barrier();

  // collect the data to node 0
  if(MPI_rank_ == 0){

        if (v_) cout << "Rank = "<< MPI_rank_ << ": Start receiving..." << endl;

        // collect data from all nodes
        for (int in = 0; in < MPI_nnodes_; ++in){
        vector<vector<double>> tmp;
        if (in == 0){tmp = data;}
        else{
            MailBox mailbox(*env_,in);
            mailbox.receive(tmp);
        }

        // write all data to file: left to right are time streams if horizontal,
        // otherwise everything vertically below each other
        int nt = tmp.size();
        int nO = tmp[0].size();
    
        if (!separate_files){
            ofstream myfile;
            if (!append) myfile.open(filename + ".txt");
            else myfile.open(filename + ".txt", ofstream::out | ofstream::app);
            for (int io = 0; io<nO; io++){
                for (int it = 0; it<nt; it++){
                    myfile << tmp[it][io] << " ";
                }
                myfile << endl;
            }
        }
    
        else{
            if (in>0) append=true;
            for (int io = 0; io<nO; io++){
                ofstream myfile;
                string fn = filename + to_string(io) + ".txt";
                if (!append) myfile.open(fn);
                else myfile.open(fn, ofstream::out | ofstream::app);
                for (int it = 0; it<nt; it++){
                    myfile << tmp[it][io] << " ";
                }
                myfile << endl;
            }
        }
        
    }
    cout << "Written file(s) " << filename << endl;
  }

  // send data of all other nodes to node 0
  else{
    MailBox mailbox(*env_,0);
    mailbox.send(data);

    if (v_) cout << "Rank = " << MPI_rank_ << ": Data was sent." << endl;
  }
  env_ -> barrier();
}

// append data to file or not
template<typename T>
void TGateQT<T>::Set_file(string filename, bool append){
  filename_ = filename;
  append_ = append;
}

// out set_verbose
template<typename T>
void TGateQT<T>::Set_verbose(bool v){
  v_ = v;
}




  // template explicit definitions
  template class TGateQT<ITensor>;
  template class TGateQT<IQTensor>;
