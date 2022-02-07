#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

/**
 * @brief Vector Class
 *
 * @tparam T
 */
template <typename T>
class Vector {
 private:
  int n;
  T* data;

 public:
  /** constructor and destructor **/
  Vector() : n(0), data(nullptr) {}
  Vector(int a) : n(a), data(new T[a]) {}
  Vector(const std::initializer_list<T>& list) : Vector((int)list.size()) {
    std::uninitialized_copy(list.begin(), list.end(), data);
  }

  /** copy constructor **/
  Vector(const Vector<T>& other) : n(other.n), data(new T[other.n]) {
    for (auto i = 0; i < other.n; i++) {
      data[i] = other.data[i];
    }
  }

  /** move constructor **/
  Vector(Vector<T>&& other) : n(other.n), data(other.data) {
    other.n = 0;
    other.data = nullptr;
  }

  /** destructor**/
  ~Vector() {
    n = 0;
    delete[] data;
    data = nullptr;
  }

  /** copy assignment **/
  Vector<T>& operator=(const Vector<T>& other) {
    if (this != &other) {
      delete[] data;
      n = other.n;
      data = new T[n];
      for (int i = 0; i < n; i++) {
        data[i] = other.data[i];
      }
    }
    return *this;
  }

  /** move assignment **/
  Vector<T>& operator=(Vector<T>&& other) {
    if (this != &other) {
      delete[] data;
      data = other.data;
      n = other.n;
      other.n = 0;
      other.data = nullptr;
    }
    return *this;
  }

  /** operator for assigning every element as same constant **/
  Vector<T>& operator=(int constant) {
    for (int i = 0; i < this->len(); ++i) {
      data[i] = constant;
    }
    return *this;
  }

  /**iterators**/
  auto begin() const { return this->data.begin(); }
  auto end() const { return this->data.end(); }
  auto cbegin() const { return this->data.cbegin(); }
  auto cend() const { return this->data.cend(); }

  /**indexing operators**/
  T& operator[](int i) { 
    if(i >= this->len()){
        throw "indexing out of scope!";
    }
    return data[i]; }
  const T& operator[](int i) const { 
    if(i >= this->len()){
        throw "indexing out of scope!";
    }
    return data[i]; }

  /** operator +*/
  template <typename T2>
  auto operator+(const Vector<T2>& other) const {
    if (this->len() != other.len()) {
      throw "Incompatible dimensions of the vectors!";
    }

    Vector<typename std::common_type<T, T2>::type> v(this->n);
    for (int i = 0; i < this->n; ++i) {
      v[i] = data[i] + other[i];
    }
    return v;
  }

  /** operator -**/
  template <typename T2>
  auto operator-(const Vector<T2>& other) const {
    if (this->len() != other.len()) {
      throw "Incompatible dimensions of the vectors!";
    }

    Vector<typename std::common_type<T, T2>::type> v(this->n);
    for (int i = 0; i < this->n; ++i) {
      v[i] = data[i] - other[i];
    }
    return v;
  }

  /** operator* between vector and scalar **/
  template <typename V>
  Vector<typename std::common_type<V, T>::type> operator*(
      const V& scalar) const {
    Vector<typename std::common_type<V, T>::type> nv(this->n);
    for (int i = 0; i < this->n; i++) {
      nv.data[i] = scalar * data[i];
    }
    return nv;
  }
  /** operator * between scalar and vector **/
  template <typename V, typename U>
  friend Vector<typename std::common_type<V, U>::type> operator*(
      const V& scalar, const Vector<U>& vec);

  /** length function for retrieving the length of the vector **/
  int len(void) const { return this->n; }
};

/** overload operator << for pretty output **/
template <typename V>
std::ostream& operator<<(std::ostream& out, const Vector<V>& v) {
  out << "[";
  for (int i = 0; i < v.len() - 1; i++) {
    out << v[i] << ", ";
  }
  out << v[v.len() - 1] << "]";
  return out;
}

/** operator* between a scalar and a vector （invoke the internal method） **/
template <typename V, typename U>
Vector<typename std::common_type<V, U>::type> operator*(const V& scalar,
                                                        const Vector<U>& vec) {
  return vec * scalar;
}

/**
 * @brief dot
 * function for computing the inner product of two vectors
 * @tparam T 
 * @tparam U 
 * @param lhs 
 * @param rhs 
 * @return std::common_type<T, U>::type 
 */
template <typename T, typename U>
typename std::common_type<T, U>::type dot(const Vector<T>& lhs,
                                          const Vector<U>& rhs) {
  if (lhs.len() != rhs.len()) {
    throw "Incompatible dimensions between two vectors!";
  }

  typename std::common_type<T, U>::type sum = 0;
  for (auto i = 0; i < lhs.len(); i++) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

/**
 * @brief norm
 * norm function returning the l2 norm of the vector
 * @tparam T 
 * @param vec 
 * @return T 
 */
template <typename T>
T norm(const Vector<T>& vec) {
  T sum = 0;
  for (auto i = 0; i < vec.len(); i++) {
    sum += vec[i] * vec[i];
  }
  return sqrt(sum);
}

/**
 * @brief Matrix Class
 * @tparam T
 */
template <typename T>
class Matrix {
 private:
  int r, c;
  std::map<std::pair<int, int>, T> data;

 public:
  /** constructor **/
  Matrix(int rows, int cols) : r(rows), c(cols){};

  /** destructor **/
  ~Matrix() {
    r = 0;
    c = 0;
    data.clear();
  }

  /** getter **/
  int row(void) const { return this->r; }
  int col(void) const { return this->c; }

  /** iterator **/
  auto begin() const { return this->data.begin(); }
  auto end() const { return this->data.end(); }
  auto cbegin() const { return this->data.cbegin(); }
  auto cend() const { return this->data.cend(); }

  /** index operator **/
  T& operator[](const std::pair<int, int>& ij) {
    if(ij.first >= this->row() || ij.second >= this->col()){
        throw "indexing out of scope!";
    }
    auto it = data.begin();
    for (; it != data.end(); ++it) {
      int i = it->first.first;
      int j = it->first.second;
      if (i == ij.first && j == ij.second) {
        return it->second;
      }
    }
    /** if the queried entry does not exist **/
    data.insert(std::make_pair(ij, 0));
    return data.at(ij);
  }

  /**
   * @brief assign value v to all elements in matrix
   * @param v
   * @return T&
   */
  Matrix<T>& operator=(T v) {
    data.clear();
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        auto ij = std::pair<int, int>(i, j);
        data.insert(std::make_pair(ij, v));
      }
    }
    return *this;
  }

  /** index operator for constant reference **/
  const T& operator()(const std::pair<int, int>& ij) const {
    if(ij.first >= this->row() || ij.second >= this->col()){
        throw "indexing out of scope!";
    }
    auto it = data.begin();
    for (; it != data.end(); ++it) {
      int i = it->first.first;
      int j = it->first.second;
      if (i == ij.first && j == ij.second) {
        return it->second;
      }
    }
    /** if the queried entry does not exist **/
    throw "The queried entry does not exist!";
  }
};
/** Matrix: overload operator << for pretty output **/
template <typename V>
std::ostream& operator<<(std::ostream& out, const Matrix<V>& m) {
  out << "\n";
  for (int i = 0; i < m.row(); i++) {
    out << "| ";
    for (int j = 0; j < m.col(); j++) {
      out << m({i, j}) << " ";
    }
    out << "|\n";
  }
  return out;
}

/** vector matrix multiplication **/
template <typename V, typename U>
Vector<typename std::common_type<V, U>::type> operator*(const Matrix<V>& lhs,
                                                        const Vector<U>& rhs) {
  if (rhs.len() != lhs.col()) {
    throw "Incompatible dimensions of the vector and the matrix!";
  }
  Vector<typename std::common_type<V, U>::type> new_vec(lhs.row());
  new_vec = 0;
  for (auto it = lhs.cbegin(); it != lhs.cend(); ++it) {
    int i = it->first.first;
    int j = it->first.second;
    new_vec[i] += it->second * rhs[j];
  }
  return new_vec;
}

/**
 * @brief Biconjugate Gradient Stabilized
 *
 * @tparam T
 * @param A
 * @param b
 * @param x
 * @param tol
 * @param maxiter
 * @return int
 */
template <typename T>
int bicgstab(const Matrix<T>& A, const Vector<T>& b, Vector<T>& x,
             T tol = (T)1e-8, int maxiter = 500) {
  if (A.col() != b.len() || A.col() != x.len()) {
    throw "Incompatible dimensions of vector and matrix!";
  }
  //TODO: fail safe: make sure A, b, x have initial values!!

  int length = b.len();
  auto q_0(b - A * x); 
  auto r_k_1(b - A * x);
  auto x_k_1(x);
  Vector<T> v_k_1(length); 
  Vector<T> p_k_1(length);
  v_k_1 = 0;
  p_k_1 = 0;
  T alpha = 1;
  T rho_k_1 = 1; 
  T omega_k_1 = 1;
  T rho_k;
  T beta;
  T omega_k;
  Vector<T> p_k(length), v_k(length), h(length), x_k(length), s(length),
      t(length), r_k(length);

  for (int k = 1; k <= maxiter; ++k) {
    rho_k = dot(q_0, r_k_1);
    beta = (rho_k / rho_k_1) * (alpha / omega_k_1);
    p_k = r_k_1 + beta * (p_k_1 - omega_k_1 * v_k_1);
    v_k = A * p_k;
    if(dot(q_0, v_k) != 0){
      alpha = rho_k / dot(q_0, v_k);}
    else{
      alpha = 0;
    }
    h = x_k_1 + alpha * p_k;
    // std::cout << "################################" << std::endl;
    // std::cout << "iteration: " << k << std::endl;
    // std::cout << "rho_k: " << rho_k << std::endl;
    // std::cout << "rho_k_1: " << rho_k_1 << std::endl;
    // std::cout << "rho_k / rho_k_1: " << rho_k / rho_k_1 << std::endl;
    // std::cout << "beta: " << beta << std::endl;
    // std::cout << "r_k_1" << r_k_1 << std::endl;
    // std::cout << "p_k" << p_k << std::endl;
    // std::cout << "vk" << v_k << std::endl;
    // std::cout << "q0: " << q_0 <<std::endl;
    // std::cout << "dot(q_0, v_k): " << dot(q_0, v_k) << std::endl;
    // std::cout << "x_k_1: " << x_k_1 << std::endl;
    // std::cout << "alpha: " << alpha << std::endl;
    // std::cout << "h: " << h << std::endl;

    if (norm(b - A * h) < tol) {
      x_k = h;
      x = x_k;
      return k;
    }

    s = r_k_1 - alpha * v_k;
    t = A * s;
    omega_k = dot(t, s) / dot(t, t);
    x_k = h + omega_k * s;
    x = x_k;

    if (norm(b - A * x_k) < tol) {
      return k;
    }

    // std::cout << "error: " << norm(b - A * x) << std::endl;

    r_k = s - omega_k * t;

    // update the variables
    r_k_1 = r_k;
    rho_k_1 = rho_k;
    omega_k_1 = omega_k;
    p_k_1 = p_k;
    v_k_1 = v_k;
    x_k_1 = x_k;
  }
  return -1;
}

/**
 * @brief convert f(y, t) result to vector format
 *
 * @tparam T typename
 * @param f function, Vector of std::function
 * @param y function's argument, Vector
 * @param t function's argument, double/float/int
 * @return Vector<T>
 */
template <typename T>
Vector<T> toVector(const Vector<std::function<T(const Vector<T>&, T)>>& f,
                   const Vector<T>& y, T t) {
  int N = y.len();
  Vector<T> rst(N);
  for (int i = 0; i < 4; i++) {
    rst[i] = f[i](y, t);
  }
  return rst;
}

/**
 * @brief Heun's integration method
 *  a function that solves a system of first-order explicit ordinary
 * differential equations by Heun’s method also called modified Euler method
 * from time $t_n$ to $t_{n+1}$ via
 * $$\left. \begin{array} { l } { \overline { y } _ { n + 1 } = y _ { n } + h
 * \cdot f ( y _ { n } , t _ { n } ) } \\ { y _ { n + 1 } = y _ { n } + \frac {
 * h } { 2 } \cdot ( f ( y _ { n } , t _ { n } ) + f ( \overline { y } _ { n + 1
 * } , t _ { n + 1 } ) ) } \end{array} \right.$$
 *
 * @tparam T type
 * @param f to pass the vector-valued function f(y(t),t) as a vector of
 * std::function’s.
 * @param y serves both as input ($y_n$) and as the result ($y_{n+1}$)
 * @param h step size
 * @param t time level $t_n$
 */
template <typename T>
void heun(const Vector<std::function<T(const Vector<T>&, T)>>& f, Vector<T>& y,
          T h, T& t) {
  T tn = t;
  T t_ = tn + h;
  Vector<T> y_hat = y + h * toVector(f, y, tn);
  Vector<T> y_ = y + (toVector(f, y, tn) + toVector(f, y_hat, t_)) * (h / (T)2);

  /** return **/
  t = t_;
  y = y_;
}

template <typename T>
class SimplestWalker {
 private:
  Vector<T> y;
  T t;
  T slope;

 public:
  //** Constructor **//
  SimplestWalker(const Vector<T>& y0, T t0, T gamma)
      : y(y0), t(t0), slope(gamma) {}
  //** Derivative **//
  Vector<T> derivative(const Vector<T>& y_cur) const {
    Vector<T> x({1,1,1,1}), b(4);
    b[0] = y_cur[2];
    b[1] = y_cur[3];
    b[2] = sin(y_cur[1] - slope);
    b[3] = - y_cur[3] * y_cur[3] * sin(y_cur[0]) + cos(y_cur[1] - slope) * sin(y_cur[0]);
    Matrix<T> A(4,4);
    A[{0,0}] = 1.0;
    A[{1,1}] = 1.0;
    A[{2,3}] = 1.0;
    A[{3,3}] = 1.0;
    A[{3,2}] = -1.0;

    if(bicgstab(A, b, x)<0) {
      std::cout<<"no solution founded, remain current state";
      x[0] = 0;
      x[1] = 0;
      x[2] = 0;
      x[3] = 0;
    }
    return x;

  }

  Vector<T> derivatives;

  const Vector<T>& step(T h) {
    Vector<std::function<T(const Vector<T>&, T)>> f = {
        [this](Vector<double> const& y, double t) { this->derivatives = this->derivative(y); return derivatives[0]; },
        [this](Vector<double> const& y, double t) { return derivatives[1]; },
        [this](Vector<double> const& y, double t) { return derivatives[2]; },
        [this](Vector<double> const& y, double t) { return derivatives[3]; },
    };
    std::cout << "last state: " << y[0] << ' ' << y[1]
              << ' ' << y[2] << ' ' << y[3] << '\n';
    heun<T>(f, y, h, t);
    std::cout << "current state: " << y[0] << ' ' << y[1]
              << ' ' << y[2] << ' ' << y[3] << '\n';
    std::cout << "current derivative: " << derivatives[0] << ' ' << derivatives[1]
              << ' ' << derivatives[2] << ' ' << derivatives[3] << '\n'<<'\n';

    return y;
  }
};

int main(int argc, char* argv[]) {
  // Your testing of the simplest walker class starts here

  /** test for Heun's integration method **/
  try {
    double h = 0.01;
    double t0 = 2.0;
    const Vector<double> y0({2, 1, 1, 4});
    double t = t0;
    Vector<double> y = y0;

    /** @brief vector of functions in lambda expression */
    Vector<std::function<double(const Vector<double>&, double)>> f = {
        [](Vector<double> const& y, double t) { return 2 * t * y[2]; },
        [](Vector<double> const& y, double t) { return 3 * t * y[3]; },
        [](Vector<double> const& y, double t) { return t * y[0]; },
        [](Vector<double> const& y, double t) { return 2 * t * y[1]; },
    };

    auto rst = toVector(f, y0, t0);
    std::cout << "[toVector] rst:  " << rst << std::endl;

    heun(f, y, h, t);
    heun(f, y, h, t);
    heun(f, y, h, t);
    std::cout << "[Heun] rst: ";
    std::cout << y << '\t' << typeid(y).name() << std::endl;
    std::cout << "[Heun] rst: " << t << std::endl;
    std::cout << "[Heun] Success" << std::endl;

  } catch (const char* msg) {
    std::cerr << msg << std::endl;
  }

  /** test for simplest walker **/
  try {
    Vector<double> y0({0.4, 0.2, 0, -0.2});
    SimplestWalker<double> sw(y0, 0, 0.009);
    for (auto i = 0; i < 2000; i++) {
      sw.step(0.01);
    }
  } catch (const char* msg) {
    std::cerr << msg << std::endl;
  }

  /** test for vector's function variable type */
  Vector<double> x({1.0, 1.1, 1.2});
  Vector<int> y({2, 3, 4});
  Vector<float> z({1.0f, 2.0f, 3.0f});
  auto z_dot = dot(z, y);
  std::cout << "[dot] between float and int: " << typeid(z_dot).name() << '\t'
            << z_dot << std::endl;
  auto y_dot = dot(y, y);
  std::cout << "[dot] between int and int: " << typeid(y_dot).name() << '\t'
            << y_dot << std::endl;

  auto z_norm = norm(z);
  std::cout << "[norm] of float: " << typeid(z_norm).name() << '\t' << z_norm
            << std::endl;
  /** test for Matrix function and variable type */
  Matrix<float> M(4, 4);
  M = 1;
  Vector<double> V1(4);
  V1 = {2, 1, 2, 1};
  auto R1 = M * V1;
  std::cout << "[matrix] M:" << M << std::endl;
  std::cout << "[matrix] V1: " << V1 << std::endl;
  std::cout << "[matrix] rst: " << R1 << '\t' << typeid(R1[0]).name() << std::endl;
  
  /** test for multiplication between different types **/
  M[{0, 0}] = 1.0;
  M[{1, 1}] = 1.0;
  M[{2, 2}] = 1.0;
  Matrix<int> N(3, 3);
  N[{0, 0}] = 5;
  N[{1, 1}] = 5;
  N[{2, 2}] = 5;
  Matrix<double> O(3, 3);
  O[{1, 0}] = 1;
  O[{1, 1}] = 1;
  O[{2, 2}] = 1;
  std::cout << "Matrix<int> * Vector<double>" << '\n' << N*x << std::endl;  
  std::cout << "Matrix<double> * Vector<float>" << '\n' << O*z << std::endl;
  std::cout << "Matrix<float> * Vector<double>" << '\n' << M*x << std::endl;
  std::cout << "Matrix<double> * Vector<double>" << '\n' << O*x << std::endl;

  /** test for scalar * vector between different types **/
  std::cout << "int * Vector<double>: " << (int)5 * x << ' ' << x * (int) 5 << typeid((x * (int) 5)[0]).name() << std::endl;
  std::cout << "float * Vector<double>: " << (float)5.0f * x << ' ' << x * (float) 5.0f << typeid((x * (float) 5.0f)[0]).name() << std::endl;
  std::cout << "int * Vector<float>: " << (int)5 * z << ' ' << z * (int) 5 << typeid((z * (int) 5)[0]).name() << std::endl;

  /** test for bicgstab */
  A = 1;
  Vector<double> b(10);
  b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Vector<double> s(10);
  s = 0;
  std::cout << "BICGSTAB initial guess s: " << s << std::endl;

  int n = bicgstab(A, b, s);
  std::cout << "BICGSTAB Matrix A: " << A << std::endl;
  std::cout << "BICGSTAB Vector b: " << b << std::endl;
  std::cout << "BICGSTAB Solution: " << s << '\t' << typeid(s[0]).name() << std::endl;
  std::cout << "BICGSTAB Status n: " << n << std::endl;
  /**
   * @brief Expected value from matlab
   * -0.4727   -0.2455   -0.0182    0.2091    0.4364    0.6636    0.8909    1.1182    1.3455    1.5727
   */
  return 0;
}