#include "matrix.hh"
#include "matrix_coo.hh"
#include <iostream>
#include <string>

void Matrix::read(const std::string & fn) {
  MatrixCOO mat;
  mat.read(fn);

  resize(mat.m(), mat.n());

  for (int z = 0; z < mat.nz(); ++z) {
    auto i = mat.irn[z];
    auto j = mat.jcn[z];
    auto a = mat.a[z];

    m_a[i * m_n + j] = a;
    if (mat.is_sym()) {
      m_a[j * m_n + i] = a;
    }
  }
}
