#include <string>
#include <vector>

#ifndef __MATRIX_H_
#define __MATRIX_H_

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
 
  // defining method which can set the values in the m_a vector 
  // inline void set(int i, int j, double value) { m_a[i*m_n + j] = value; }
  inline std::vector<double> values() {return m_a; }

private:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_a;
};

#endif // __MATRIX_H_
