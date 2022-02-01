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
  // Your implementation of the Vector class starts here
 private:
  int n;
  T* data;

 public:
  /** constructor and destructor **/
  explicit Vector() : n(0), data(nullptr) {}
  explicit Vector(int a) : n(a), data(new T[a]) {}
  explicit Vector(const std::initializer_list<T>& list)
      : Vector((int)list.size()) {
    std::uninitialized_copy(list.begin(), list.end(), data);
  }

  /** copy constructor **/
  Vector(const Vector<T>& other) : n(other.n), data(new T[other.n]) {
    for (auto i = 0; i < other.n; i++) {
      data[i] = other.data[i];
    }
    // std::cout<<"copy constructor\n";
  }

  /** move constructor **/
  Vector(Vector<T>&& other) : n(other.n), data(other.data) {
    other.n = 0;
    delete[] other.data;
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

  /** change the value of all the entry as a constant **/
  Vector<T>& operator=(int constant) {
    for (int i = 0; i < this->n; ++i) {
      T value = constant;
      data[i] = value;
    }
    return *this;
  }

  /** move assignment **/
  Vector<T>& operator=(Vector<T>&& other) {
    if (this != &other) {
      std::swap(this->data, other.data);
      std::swap(this->n, other.n);
      other.n = 0;
      delete[] other.data;
      other.data = nullptr;
    }
    return *this;
  }

  /**iterators**/
  // TODO(@edmundwsy): cannot use range-based for loop using iterators
  auto begin() const { return this->data.begin(); }
  auto end() const { return this->data.end(); }
  auto cbegin() const { return this->data.cbegin(); }
  auto cend() const { return this->data.cend(); }

  /**indexing operators**/
  T& operator[](int i) { return data[i]; }
  const T& operator[](int i) const { return data[i]; }

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

  /** operator* between a scalar and a vector **/
  template <typename V>
  Vector<typename std::common_type<V, T>::type> operator* (const V& scalar) const {
    Vector<typename std::common_type<V, T>::type> nv(this->n);
    for (int i = 0; i < this->n; i++) {
      nv.data[i] = scalar * data[i];
    }
    return nv;
  }
  template <typename V, typename U>
  friend Vector<typename std::common_type<V, U>::type> operator*(
      const V& scalar, const Vector<U>& vec);

  /** length function for retrieving the length of the vector **/
  int len(void) const { return this->n; }

  /** display the content of the vector **/
  void display(void) const{
    for(int i=0; i < this->len(); ++i){std::cout << ' ' << data[i];}
  }
};

/** operator* between a scalar and a vector （invoke the internal method） **/
template <typename V, typename U>
Vector<typename std::common_type<V, U>::type> operator*(const V& scalar,
                                                        const Vector<U>& vec){
  return vec * scalar;
}

/** dot function for computing the inner product of two vectors **/
template <typename T, typename U>
typename std::common_type<T, U>::type dot(const Vector<T>& lhs,
                                          const Vector<U>& rhs) {
  if (lhs.len() != rhs.len()) {
    throw "Incompatible dimensions between two vectors!";
  }

  typename std::common_type<T, U>::type sum;
  for (auto i = 0; i < lhs.len(); i++) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

/** norm function returning the l2 norm of the vector **/
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

  /** index operator for constant reference **/
  const T& operator()(const std::pair<int, int>& ij) const {
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

/** vector matrix multiplication **/
template <typename V, typename U>
Vector<typename std::common_type<V, U>::type> operator*(const Matrix<V>& lhs,
                                                        const Vector<U>& rhs) {
  if (rhs.len() != lhs.col()) {
    throw "Incompatible dimensions of the vector and the matrix!";
  }
  Vector<typename std::common_type<V, U>::type> new_vec(lhs.row());
  for (auto it = lhs.cbegin(); it != lhs.cend(); ++it) {
    int i = it->first.first;
    int j = it->first.second;
    new_vec[i] += it->second * rhs[j];
  }
  return new_vec;
}

template <typename T>
int bicgstab(const Matrix<T>& A, const Vector<T>& b, Vector<T>& x,
             T tol = (T)1e-8, int maxiter = 100) {
  // Your implementation of the bicgstab function starts here
  int length = b.len();
  auto q_0(b - A * x), r_k_1(b - A * x);
  auto x_k_1 = x;
  Vector v_k_1(length), p_k_1(length);
  v_k_1(length), p_k_1(length) = 0;
  double alpha, rho_k_1, omega_k_1 = 1;
  double rho_k, beta, omega_k;
  Vector p_k(length), v_k(length), h(length), x_k(length), s(length), t(length),
      r_k(length);

  for (int k = 1; k <= maxiter; ++k) {
    rho_k = dot(q_0, r_k_1);
    beta = (rho_k / rho_k_1) * (alpha / omega_k_1);
    p_k = r_k_1 + beta * (p_k_1 - omega_k_1 * v_k_1);
    v_k = A * p_k;
    alpha = rho_k / dot(q_0, v_k);
    h = x_k_1 + alpha * p_k;

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

template <typename T>
void heun(const Vector<std::function<T(Vector<T> const&, T)> >& f,
          const Vector<T>& y, T h, T& t) {
  // Your implementation of the heun function starts here
}

template <typename T>
class SimplestWalker {
 private:
  Vector<T> y_init;
  T t_init;
  T slope;

 public:
  //** Constructor **//
  SimplestWalker(const Vector<T>& y0, T t0, T gamma)
      : y_init(y0), t_init(t0), slope(gamma) {}
  //** Derivative **//
  Vector<T> derivative(const Vector<T>& y) const {
    Vector<T> dot(4);
    dot[0] = y[2];
    dot[1] = y[3];
    dot[3] = sin(dot[1] - slope);
    dot[2] = dot[3] + y[3] * y[3] * sin(y[0]) - cos(y[1] - slope) * sin(y[0]);
    return dot;
  }
};

int main(int argc, char* argv[]) {
  // Your testing of the simplest walker class starts here
  // test Matrix
  Matrix<double> M(10, 20), M1(10, 3);
  Vector<double> x({1.0, 1.1, 1.2}), y({2, 3, 4}), z({1.0f, 2.0f, 3.0f});

  try{

    /** tests for Vector object **/
    auto x_plus_y = x-y;
    for(int i=0; i<x_plus_y.len(); ++i){std::cout << x_plus_y[i] << ' ';}
    std::cout << '\n';
    Vector<double> other(5);
    other = 1;
    //std::cout << dot(x_plus_y, other) << std::endl;
    x_plus_y = 4*x_plus_y;
    for(int i=0; i<x_plus_y.len(); ++i){std::cout << x_plus_y[i] << ' ';}
    std::cout << '\n';
    std::cout << norm(x_plus_y) << std::endl;
    // for(auto iter = x_plus_y.begin(); iter != x_plus_y.end(); ++iter){
    //     std::cout << *iter << ' ';
    // }

    /** tests for Matrix object **/
    M[{1,9}] = 1.0; // set value at row 0, column 0 to 1.0
    std::cout << M[{1, 9}] << std::endl;
    std::cout << M[{0, 0}] << std::endl;
    std::cout << M({1, 9}) << std::endl;
    std::cout << typeid(M.row()).name() << ' ' << M.col() << std::endl;
    Vector<double> v2 = x;
    M1[{0, 0}] = 1; M1[{1, 1}] = 1; M1[{2, 2}] = 1;
    v2 = M1*x_plus_y;
    v2.display();
    }
  catch(const char* msg){
    std::cerr << msg << std::endl;
  }
  return 0;
}