#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

template <typename T>
class Vector {
  // Your implementation of the Vector class starts here
};

template <typename T, typename U>
typename std::common_type<T, U>::type dot(const Vector<T>& lhs,
                                          const Vector<U>& rhs) {
  // Your implementation of the dot function starts here
}

template <typename T>
T norm(const Vector<T>& vec) {
  // Your implementation of the norm function starts here
}

template <typename T>
class Matrix {
  // Start your implementation of the matrix class here
};

template <typename T>
int bicgstab(const Matrix<T>& A, const Vector<T>& b, Vector<T>& x,
             T tol = (T)1e-8, int maxiter = 100) {
  // Your implementation of the bicgstab function starts here
}

template <typename T>
void heun(const Vector<std::function<T(Vector<T> const&, T)> >& f, Vector<T>& y,
          T h, T& t){
    // Your implementation of the heun function starts here
};

template <typename T>
class SimplestWalker {
  // Your implementation of the simplest walker class starts here
};

int main(int argc, char* argv[]) {
  // Your testing of the simplest walker class starts here
  return 0;
}
