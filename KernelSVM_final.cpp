/*
 * STRUMPACK -- SVM, Copyright (c) October 2022, 
 * Developers: S.Cipolla, J. Gondzio
 *             (The University of Edinburgh).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>

#include "kernel/KernelRegression.hpp"
#include "misc/TaskTimer.hpp"


using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;


template<typename scalar_t> vector<scalar_t>
read_from_file(string filename) {
  vector<scalar_t> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  data.shrink_to_fit();
  return data;
}


int main(int argc, char *argv[]) {
  using scalar_t = double;
  string filename(".././data/2M");
  size_t d = 2;
  scalar_t lambda =1e1;  // This is the ADMM \beta
  KernelType ktype = KernelType::GAUSS;
  string mode("test");
  
  cout << "# usage: ./SVM_ADMM file d "
       << "kernel(Gauss, Laplace) mode(validation, test)" << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) lambda = stoi(argv[3]);
  if (argc > 4) ktype = kernel_type(string(argv[4]));
  if (argc > 5) mode = string(argv[5]);
  cout << endl;
  cout << "# file                = " << filename << endl;
  cout << "# data dimension      = " << d << endl;
  cout << "# kernel type         = " << get_name(ktype) << endl;
  cout << "# validation/test     = " << mode << endl;
  
  bool PrintADMMRes = true;

  // ADMM Parameters
  //scalar_t lambda =1e4;  // This is the ADMM \beta
  size_t MaxIt=10;
  cout << "# ADMM beta           = " << lambda << endl;
  cout << "# ADMM MaxIt           = " << MaxIt << endl;
  // Loading Data
  
  cout << endl << "# Reading data ..." << endl;
  // Read from csv files
  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  //cout << "# Reading took " << timer.elapsed() << endl;

  size_t n = training.size()/ d;
  size_t m = testing.size() / d;
  cout << "# training dataset = " << n << " x " << d << endl;
  cout << "# testing dataset  = " << m << " x " << d << endl << endl;
  // Wrapping Data to Stumpack format
  DenseMatrixWrapper<scalar_t>
    training_points(d, n, training.data(), d),
    test_points(d, m, testing.data(), d);
  scalar_t C;
  scalar_t h;
  vector<scalar_t> hh {1,10,0.1};
  vector<scalar_t> CC {0.1,1.,10};
  // Beginning of grid search
  for(size_t h_iter=0; h_iter < hh.size(); h_iter ++){ 
    // Kernel Parameters
    int p = 1;  // kernel degree
    h=hh[h_iter];
    cout << "# kernel h            = " << h << endl;
    cout << "# p                   = " << p << endl;
    HSSOptions<scalar_t> hss_opts;
    hss_opts.set_verbose(true);
    hss_opts.set_from_command_line(argc, argv);
    hss_opts.set_rel_tol(0.6);
    hss_opts.set_abs_tol(0.06);
    hss_opts.set_approximate_neighbors(512);
    hss_opts.set_leaf_size(512);
    hss_opts.set_ann_iterations(20);
    hss_opts.set_max_rank(2000);
    //hss_opts.set_random_engine(random::RandomEngine::MERSENNE);
    //hss_opts.set_clustering_algorithm(strumpack::ClusteringAlgorithm::PCA);
    hss_opts.describe_options();
    TaskTimer timer("compression");
    // Creating HSS Approximation
    timer.start(); 
    auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda, p);
    HSSMatrix<scalar_t> H(*K, hss_opts);
    // Permuting The Data: training_points are automatically permuted
    DenseMatrixWrapper<scalar_t> B(1, K->n(), train_labels.data(), 1);
    B.lapmt(K->permutation(), true);
    // Printing The Details For HSS Approximation
    if (hss_opts.verbose()) {
       std::cout << "# HSS compression time = "
                 << timer.elapsed() << std::endl;
       if (H.is_compressed())
         std::cout << "# created HSS matrix of dimension "
                   << H.rows() << " x " << H.cols()
                   << " with " << H.levels() << " levels" << std::endl
                   << "# compression succeeded!" << std::endl;
       else std::cout << "# compression failed!!!" << std::endl;
       std::cout << "# rank(H) = " << H.rank() << std::endl
                 << "# HSS memory(H) = "
                  << H.memory() / 1e6 << " MB " << std::endl << std::endl
                  << "# factorization start" << std::endl;
      }
   timer.start();
   auto ULV = H.factor();
   if (hss_opts.verbose())
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
    
   // Computing (H+\beta I)^-1y;
   vector<scalar_t> weights(n, 1.);
   DenseMatrixWrapper<scalar_t> weightsDM(n, 1, weights.data(), 1);
   timer.start();
   H.solve(ULV, weightsDM);
   if (hss_opts.verbose())
       std::cout << "# solve time = " << timer.elapsed() << std::endl;
   scalar_t eHbe;
   eHbe=0;
   for(size_t j=0; j<n; j++){
           eHbe += weights[j];
        } 
   for(size_t j=0; j<n; j++){
         if(train_labels[j]< 0.)
            weights[j]=-weights[j];
       }
   // Training For Different values of C
   for(size_t C_iter=0; C_iter<CC.size(); C_iter++ ){
   C = CC[C_iter];
   cout << "# C parameter for SVM = " << C << endl << endl;
   //  ADMM
   scalar_t coeff;
   vector<scalar_t> x(n,0.), mu(n,0.), z(n,0.), e(n,1.), error(n,0.);
   DenseMatrixWrapper<scalar_t> xx(n, 1, x.data(), n);
   DenseMatrixWrapper<scalar_t> zz(n, 1, z.data(), n);
   cout << "# ADMM Procedure "<< endl;
   timer.start();
   for (size_t i=0; i<MaxIt; i++)
      {// --> Solution of (H+\beta I)x = Y(K+\beta I)Yx = q^k; <-- 
       x = e;
       blas::axpy(n, lambda,  z.data(), 1, x.data(), 1);
       blas::axpy(n,    1.0, mu.data(), 1, x.data(), 1);
       coeff = blas::dotu(n,x.data(),1, weights.data(),1);
       for(size_t j=0; j<n; j++){
         if(train_labels[j] < 0.)
            {x[j]=-x[j];}
        }
       H.solve(ULV, xx);

       //coeff=0;  
       //coeff = blas::dotu(n,train_labels.data(),1, x.data(),1);
       //for(size_t j=0; j<n; j++){
          // coeff += x[j];
        //} 
       coeff = - coeff/(eHbe);
       for(size_t j=0; j<n; j++){
         if(train_labels[j]< 0.)
            {x[j]=-x[j];}
        }  
       //coeff = blas::dotu(n,train_labels.data(),1, x.data(),1); 
       blas::axpy(n, coeff, weights.data(), 1, x.data(), 1);
       z = x;
       blas::axpy(n, -1.0/lambda, mu.data(), 1, z.data(), 1);
       for(size_t j=0; j<n; j++){
         if(z[j]>C)
            {z[j]=C;}
         if (z[j] < 0.)
            {z[j]=0.;}
        }
       blas::axpy(n,  lambda, z.data(), 1, mu.data(), 1);
       blas::axpy(n, -lambda, x.data(), 1, mu.data(), 1);
       if (PrintADMMRes){
           error = x;
           blas::axpy(n, -1.0, z.data(), 1, error.data(), 1);
           std::cout << "# Primal Res = " <<blas::nrm2(n,error.data(),1) << std::endl;
           scalar_t yx;
           yx =  blas::dotu(n,train_labels.data(),1, x.data(),1);
           std::cout << "# Check on x = " << yx << std::endl;
         }
      }   
     if (hss_opts.verbose())
     std::cout << "# Training Time = " << timer.elapsed() << std::endl;      
   // End Of Training Phase
   // Start of Testing 
   // Computing the Bias
   scalar_t bias;
   scalar_t sum_train_labels = 0.;
   size_t   ne_counter=0;
   vector<scalar_t> kc(n,0.), z_alpha(n);
   DenseMatrixWrapper<scalar_t> KC(n, 1, kc.data(), n);
   timer.start();
   for(size_t j=0; j<n; j++){
     if(z[j]>1e-5 && z[j]<C-1e-5){
        kc[j]=1.;
        sum_train_labels =  sum_train_labels + train_labels[j];
        ne_counter = ne_counter + 1;
     }
    }
   cout << "# sumTL: " << sum_train_labels <<  endl;
   cout << "# NeC    " << ne_counter <<  endl;
   H.shift(-lambda); 		
   //auto Hdense=H.dense();
   //Hdense.print_to_file("name","filename.txt"); 
   H.apply(KC);    
   z_alpha = z;
   for(size_t j=0; j<n; j++){
         if(train_labels[j]==-1.)
            {z_alpha[j]=-z_alpha[j];}
     }
   bias = (blas::dotu(n,kc.data(),1, z_alpha.data(),1));
   bias = - bias + sum_train_labels;
   if(ne_counter>0){
      bias = bias/ne_counter;}
   cout << "# bias: " << bias <<  endl;
   std::cout << "# Time to compute BIAS = " << timer.elapsed() << std::endl;   
   // Testing Accuracy
   // TO DO: Make it Work for all Kernels
   vector<scalar_t> computed_labels(m,0.);
   scalar_t label;

   #pragma omp parallel for num_threads(15)
   for(size_t jj=0; jj<m; jj++){
     for(size_t j=0; j<n; j++){
     computed_labels[jj] += z_alpha[j]*(std::exp
                         (-Euclidean_distance_squared(d, training_points.ptr(0, j), test_points.ptr(0, jj))
                           / (scalar_t(2.) * h * h) ) );
     }
    } 
    #pragma omp parallel for num_threads(15)
    for(size_t jj=0; jj<m; jj++){
     computed_labels[jj] += bias;
     if (computed_labels[jj] > 0.){
        computed_labels[jj] = 1.;
       }
       else {
       computed_labels[jj] = -1.;
       } 
    }
    if (hss_opts.verbose())
       std::cout << "# Prediction Time = " << timer.elapsed() << std::endl; 
   // compute accuracy score of prediction
   // compute accuracy score of prediction
   size_t incorrect_quant = 0;
   for (size_t ii=0; ii<m; ii++)
    if ((computed_labels[ii] == 1.  && test_labels[ii] == -1.) ||
        (computed_labels[ii] == -1. && test_labels[ii] == 1.))
      {incorrect_quant++;}
   cout << "# prediction score: "
       << (float(m - incorrect_quant) / m) * 100. << "%" << endl
       << "# c-err: "
       << (float(incorrect_quant) / m) * 100. << "%"
       << endl << endl;
   }
   H.reset();
  }
 return 0;
}
