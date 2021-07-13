#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;


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
