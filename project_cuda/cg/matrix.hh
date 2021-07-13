#include <string>
#include <vector>
// #include <cuda_runtime.h>

#ifndef __MATRIX_H_
#define __MATRIX_H_

// converting the Matrix so you can use it on the device and on the host is difficult keep it as it was
// directly send the data to the device 

class Matrix {
public:
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n), m_a(m * n) {}
  
  void resize(int m, int n) {
    m_m = m;
    m_n = n;
    m_a.resize(m * n);
  }
 
  inline double & operator()(int i, int j) { return m_a[i * m_n + j]; }

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }
  
  inline double * data() { return m_a.data(); }

  void read(const std::string & filename);

private:
  int m_m{0};
  int m_n{0};

  // needs to be an array, not a vector of doubles, so you can use it in the device ?
  std::vector<double> m_a;
  // double *m_a;
};

#endif // __MATRIX_H_
