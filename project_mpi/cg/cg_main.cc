#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char ** argv) {
 
  MPI_Init(&argc, &argv);

  int prank, psize;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << std::endl;
    return 1;
  }

  CGSolver solver;
  solver.read_matrix(argv[1]);

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  solver.init_source_term(h);

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  if (prank == 0) { 
     std::cout << "Call CG dense on matrix size " << m << " x " << n << std::endl;
  }

  auto t1 = clk::now();
  solver.solve(x_d);
  second elapsed = clk::now() - t1;

  if(prank == 0) {
     std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";
  }

  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1]);
  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  if (prank == 0) {
     std::cout << "Call CG sparse on matrix size " << m << " x " << n << std::endl;
  }

  t1 = clk::now();
  sparse_solver.solve(x_s);
  elapsed = clk::now() - t1;

  if (prank == 0) {
     std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";
  }

  return 0;
}
