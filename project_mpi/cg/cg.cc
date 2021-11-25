#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>
  
const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

/*
Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
*/

void CGSolver::solve(std::vector<double> & x) {
  
  // The value of m_n is 10000
  // The value of m_m is 10000
 
  // MPI initialization
  int prank, psize;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
 
  if(psize > 1) {

    std::vector<double> r(m_m);
    std::vector<double> smaller_r(m_m/psize);
    std::vector<double> smaller_p(m_m/psize);
    std::vector<double> p(m_m);
    std::vector<double> Ap(m_m/psize);
    std::vector<double> tmp(m_m/psize);
    std::vector<double> smaller_x(m_m/psize);
    std::vector<double> smaller_m_b(m_m/psize);

    std::fill_n(Ap.begin(), Ap.size(), 0.);
    Matrix smaller_m_A(m_m/psize, m_n);
  
    for(int i = 0; i < m_m/psize; i++) {
       for(int j = 0; j < m_n; j++) {
	
          smaller_m_A(i,j) = m_A(prank*(m_m/psize) + i,j);
       }
    }  
                                                  
    for(int i=0; i < m_m/psize; i++) {
       smaller_x[i] = x[prank*(m_m/psize)+i]; 
    }

    for(int i=0; i < m_m/psize; i++) {
       smaller_m_b[i] = m_b[prank*(m_m/psize)+i];
    }

    // Calculating A*x on smaller set of rows 
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m/psize, m_n, 1., smaller_m_A.data(), m_n, x.data(), 1, 0., Ap.data(), 1); 

    smaller_r = smaller_m_b;

    // Calculating r = b - A*x on smaller set of rows 
    cblas_daxpy(m_m/psize, -1., Ap.data(), 1, smaller_r.data(), 1); 
    
    // Setting p = r on smaller set of rows 
    smaller_p = smaller_r;

    // Gathering all the rows together
    MPI_Allgather(&smaller_r[0], m_m/psize, MPI_DOUBLE, &r[0], m_m/psize, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&smaller_p[0], m_m/psize, MPI_DOUBLE, &p[0], m_m/psize, MPI_DOUBLE, MPI_COMM_WORLD);

    // r' * r;
    auto rold = cblas_ddot(m_m, r.data(), 1, r.data(), 1);

    // for i = 1:length(b)
    int k = 0;
    for (; k < m_m/psize ; ++k) {
 
       // Ap = A * p;
       std::fill_n(Ap.begin(), Ap.size(), 0.);
       cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m/psize, m_n, 1., smaller_m_A.data(), m_n, p.data(), 1, 0., Ap.data(), 1);
     
       // alpha = rold / (p' * Ap);
       auto alpha = rold / std::max(cblas_ddot(m_m/psize, smaller_p.data(), 1, Ap.data(), 1), rold * NEARZERO);

       // x = x + alpha * p;
       cblas_daxpy(m_m/psize, alpha, smaller_p.data(), 1, smaller_x.data(), 1);

       // r = r - alpha * Ap;
       cblas_daxpy(m_m/psize, -alpha, Ap.data(), 1, smaller_r.data(), 1);
     
       // prank*m_n/psize
       MPI_Allgather(&smaller_r[0], m_m/psize, MPI_DOUBLE, &r[0], m_m/psize, MPI_DOUBLE, MPI_COMM_WORLD);
     
       // rnew = r' * r;
       auto rnew = cblas_ddot(m_m, r.data(), 1, r.data(), 1);
   
       if (std::sqrt(rnew) < m_tolerance)
          break; // Convergence test

       auto beta = rnew / rold;                                                                                                                                       
  
     // p = r + (rnew / rold) * p;
       tmp = smaller_r;
       cblas_daxpy(m_m/psize, beta, smaller_p.data(), 1, tmp.data(), 1);
       smaller_p = tmp;

       MPI_Allgather(&smaller_p[0], m_m/psize, MPI_DOUBLE, &p[0], m_m/psize, MPI_DOUBLE, MPI_COMM_WORLD);
  
       // rsold = rsnew;
       rold = rnew;
    
       if (DEBUG) {
         std::cout << "\t[STEP " << k << "] residual = " << std::scientific
         << std::sqrt(rold) << "\r" << std::flush;
       }
    }   

    MPI_Allgather(&smaller_x[0], m_m/psize, MPI_DOUBLE, &x[0], m_m/psize, MPI_DOUBLE, MPI_COMM_WORLD);
    
    if (DEBUG) {
      std::fill_n(r.begin(), r.size(), 0.);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n, x.data(), 1, 0., r.data(), 1);
      cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);

      auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
      std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));

      auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));

      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
      << std::sqrt(rold) << ", ||x|| = " << nx
      << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
  } 

  if (psize == 1) {
    std::vector<double> r(m_n);
    std::vector<double> p(m_n);
    std::vector<double> Ap(m_n);
    std::vector<double> tmp(m_n);

    // r = b - A * x;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n, x.data(), 1, 0., Ap.data(), 1);
  
    r = m_b;
    cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);
  
    // p = r;
    p = r;
  
    // rsold = r' * r;
    auto rsold = cblas_ddot(m_n, r.data(), 1, p.data(), 1);
  
    // for i = 1:length(b)
    int k = 0;
    for (; k < m_n; ++k) {

      // Ap = A * p;
      std::fill_n(Ap.begin(), Ap.size(), 0.);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n, p.data(), 1, 0., Ap.data(), 1);
  
      // alpha = rsold / (p' * Ap);
      auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1), rsold * NEARZERO);
  
      // x = x + alpha * p;
      cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);
      // r = r - alpha * Ap;
      cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);
      // rsnew = r' * r;
      auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);
      
      if (std::sqrt(rsnew) < m_tolerance)
         break; // Convergence test
  
      auto beta = rsnew / rsold;

      // p = r + (rsnew / rsold) * p;
      tmp = r;
      cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
      p = tmp;
      
      // rsold = rsnew; 
      rsold = rsnew;
  
      if (DEBUG) {
         std::cout << "\t[STEP " << k << "] residual = " << std::scientific
         << std::sqrt(rsold) << "\r" << std::flush;
      }
    }

    if (DEBUG) {
      std::fill_n(r.begin(), r.size(), 0.);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n, x.data(), 1, 0., r.data(), 1);
      cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);

      auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
      std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));

      auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));

      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
      << std::sqrt(rsold) << ", ||x|| = " << nx
      << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
  }
}

void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Sparse version of the cg solver
*/
void CGSolverSparse::solve(std::vector<double> & x) {
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_n);
  std::vector<double> tmp(m_n);

  // r = b - A * x;
  m_A.mat_vec(x, Ap);
  r = m_b;
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    m_A.mat_vec(p, Ap);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                  rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG) {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG) {
    m_A.mat_vec(x, r);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void CGSolverSparse::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
