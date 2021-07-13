#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void initialization(double *x_device, double *r_device, double *p_device, double *tmp_device, double *m_A_device, double *m_b_device, int *m_n_device) {

  // Find the blockId, we know it is between 0 and 9
  // Each block is in charge of 1000 rows
  int blockId = blockIdx.x;

  // Find the threadId
  // In each block you may have a different number of threads, and this number must divide 1000
  int threadId = threadIdx.x;

  // Number of rows each thread must consider 
  int numRowsForEachThread = (int) 1000/blockDim.x;

  // Begining row index 
  int beginningRowIndex = blockId*1000 + threadId*numRowsForEachThread;

  // Creating A*x for the specific rows
  for(int i = beginningRowIndex; i < beginningRowIndex + numRowsForEachThread; i++){

    int m_n_device_int = *m_n_device; 
    for(int j = 0; j < *m_n_device; j++) {
       tmp_device[i] = tmp_device[i] + x_device[j]*m_A_device[i*m_n_device_int + j];
    }
  } 
  
  for(int i = 0; i < numRowsForEachThread; i++) {

    r_device[beginningRowIndex + i] = m_b_device[beginningRowIndex + i] - tmp_device[beginningRowIndex + i];
  }  

  // Synchronize after this kernel
  *p_device = *r_device;
}

__global__ void computeAlphaNumerator(double *r_device, double *p_device, double *Ap_device, double *m_A_device, int *m_m_device, int *m_n_device, double *numerator_alpha) {

  int blockId = blockIdx.x;
  int threadId = threadIdx.x;
  int numRowsForEachThread = (int) 1000/blockDim.x;
  int beginningRowIndex = blockId*1000 + threadId*numRowsForEachThread;

  for(int j = 0; j < *m_m_device; j++) {
     *numerator_alpha = *numerator_alpha + r_device[j]*r_device[j];
  }
     
  for(int i = beginningRowIndex; i < beginningRowIndex + numRowsForEachThread; i++){
     
     int m_n_device_int = *m_n_device;
     
     for(int j = 0; j < m_n_device_int; j++) {
        Ap_device[i] = Ap_device[i] + p_device[j]*m_A_device[i*m_n_device_int + j];
     }
  }

  // Synchronize after this kernel
}

__global__ void computeAlphaDenominator(double *x_device, double *r_device, double *p_device,double *Ap_device, int *m_m_device, double *numerator_alpha, double *denominator_alpha) {

  int blockId = blockIdx.x;
  int threadId = threadIdx.x;
  int numRowsForEachThread = (int) 1000/blockDim.x;
  int beginningRowIndex = blockId*1000 + threadId*numRowsForEachThread;

  for(int j = 0; j < *m_m_device; j++) {
     *denominator_alpha = *denominator_alpha + p_device[j]*Ap_device[j];
  }
  
  double alpha;

  if(*denominator_alpha != 0) {
     alpha = *numerator_alpha/ *denominator_alpha;
  } else {
     alpha = 0.1;
  }
   
  for(int i = 0; i < numRowsForEachThread; i++) {
     x_device[beginningRowIndex + i] = x_device[beginningRowIndex + i] + alpha*p_device[beginningRowIndex + i];
     r_device[beginningRowIndex + i] = r_device[beginningRowIndex + i] - alpha*Ap_device[beginningRowIndex + i];
  } 

  // Synchronize after this kernel
}

__global__ void computeBeta(double *r_device, double *p_device, int *m_m_device, double *numerator_alpha, double *beta) {  

  int blockId = blockIdx.x;
  int threadId = threadIdx.x;
  int numRowsForEachThread = (int) 1000/blockDim.x;
  int beginningRowIndex = blockId*1000 + threadId*numRowsForEachThread;
    
  for(int j = 0; j < *m_m_device; j++) {
     *beta = *beta + r_device[j]*r_device[j];
  }

  *beta = *beta/ *numerator_alpha;
     
  for(int i=0; i < numRowsForEachThread; i++) {
     p_device[beginningRowIndex + i] = r_device[beginningRowIndex + i] + (*beta)*p_device[beginningRowIndex + i];
  }

  // Synchronize after this kernel
}

void CGSolver::solve(std::vector<double> & x) {

  // CUDA parameters
  // --- CAN BE CHANGED --- 
  int grid_size = 10;
  int block_size = 10;

  // Arrays allocated
  std::vector<double> r(m_m);
  double *r_device;
  std::vector<double> p(m_m);
  double *p_device;
  std::vector<double> Ap(m_m);
  std::fill_n(Ap.begin(), Ap.size(), 0.); 
  double *Ap_device;
  std::vector<double> tmp(m_m);
  std::fill_n(tmp.begin(), tmp.size(), 0.); 
  double *tmp_device;

  double *x_device;

  cudaMalloc((void **) &r_device, m_m*sizeof(double));
  cudaMemcpy(r_device, &r[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &p_device, m_m*sizeof(double));
  cudaMemcpy(p_device, &p[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &Ap_device, m_m*sizeof(double));
  cudaMemcpy(Ap_device, &Ap[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &tmp_device, m_m*sizeof(double));
  cudaMemcpy(tmp_device, &tmp[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &x_device, m_m*sizeof(double));
  cudaMemcpy(x_device, &x[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  double *m_A_device;
  cudaMalloc((void **) &m_A_device, m_m*m_n*sizeof(double));
  cudaMemcpy(m_A_device, &m_A.data()[0], m_m*m_n*sizeof(double), cudaMemcpyHostToDevice);  

  double *m_b_device;
  cudaMalloc((void **) &m_b_device, m_m*sizeof(double));
  cudaMemcpy(m_b_device, &m_b.data()[0], m_m*sizeof(double), cudaMemcpyHostToDevice);

  // Scalars allocated
  int *m_m_device, *m_n_device;
  double numerator_alpha = 0.0;
  double denominator_alpha = 0.0;
  double beta = 0.0;
  double *numerator_alpha_device, *denominator_alpha_device, *beta_device;
 
  cudaMalloc((void **)&m_m_device, sizeof(int));
  cudaMemcpy(m_m_device, &m_m, sizeof(int), cudaMemcpyHostToDevice);
 
  cudaMalloc((void **)&m_n_device, sizeof(int));
  cudaMemcpy(m_n_device, &m_n, sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&numerator_alpha_device, sizeof(int));
  cudaMemcpy(numerator_alpha_device, &numerator_alpha, sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&denominator_alpha_device, sizeof(int));
  cudaMemcpy(denominator_alpha_device, &denominator_alpha, sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&beta_device, sizeof(int));
  cudaMemcpy(beta_device, &beta, sizeof(int), cudaMemcpyHostToDevice);

  double m_tolerance{1e-10};
  double stopping_criterion = 0.0;

  initialization<<<grid_size, block_size>>>(x_device, r_device, p_device, tmp_device, m_A_device, m_b_device, m_n_device);
  cudaDeviceSynchronize();
  cudaMemcpy(&r[0], &r_device, m_m*sizeof(double), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < m_m; i++) {
    stopping_criterion += r[i];
  }

  while(stopping_criterion > m_tolerance) {
     computeAlphaNumerator<<<grid_size, block_size>>>(r_device, p_device, Ap_device, m_A_device, m_m_device, m_n_device, numerator_alpha_device);
     cudaDeviceSynchronize();
     computeAlphaDenominator<<<grid_size, block_size>>>(x_device, r_device, p_device, Ap_device, m_m_device, numerator_alpha_device, denominator_alpha_device);
     cudaDeviceSynchronize();
     computeBeta<<<grid_size, block_size>>>(r_device, p_device, m_m_device, numerator_alpha_device, beta_device);  
     cudaDeviceSynchronize();

     cudaMemcpy(&r[0], &r_device, m_m*sizeof(double), cudaMemcpyDeviceToHost);
     stopping_criterion = 0.0;

     for (int i = 0; i < m_m; i++) {
        stopping_criterion += r[i];
     } 
  }

  // When the iterations are finished, copy the final answer from the device to the host variable x
  cudaMemcpy(&x[0], &x_device, m_m*sizeof(double), cudaMemcpyDeviceToHost); 
}

