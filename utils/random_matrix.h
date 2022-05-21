#include <random>
#include <complex>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#ifndef random_m
#define random_m

// real random Gaussian matrix
MatrixXf random_gauss(int N, float mu=0., float sigma=1.){
    
    // Gaussian random number generator
    random_device rd;
    mt19937 gen(rd());  
    normal_distribution<float> dis(mu, sigma);
    
    return MatrixXf::Zero(N,N).unaryExpr([&](float dummy){return dis(gen);});
}

// complex random Gaussian matrix, normalized
MatrixXcf random_gauss_complex(int N){
    // complex unit
    std::complex<float> ii(0.,1.);
    
    auto real_part = random_gauss(N);
    auto cmplx_part = random_gauss(N);
    
    return  (real_part + ii*cmplx_part)/sqrt(2.0);
}

// random Haar unitary
MatrixXcf random_U_haar(int N){
    
    // The Gaussian random matrix
    auto RGM = random_gauss_complex(N);
    
    // QR decomposition
    HouseholderQR<MatrixXcf> qr(RGM.rows(),RGM.cols());
    qr.compute(RGM);
    
    MatrixXcf Q = qr.householderQ();
    MatrixXcf R = qr.matrixQR().triangularView<Upper>();
    
    // normalize correctly with diag R for Haar measure
    ArrayXcf d = R.diagonal();
    auto da = d/abs(d);
    MatrixXcf Lambda = da.matrix().asDiagonal();
    
    return Q*Lambda;
}




#endif