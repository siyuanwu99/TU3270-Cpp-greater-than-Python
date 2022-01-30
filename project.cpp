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
    std::cout<<"copy constructor\n";
  }
 
 /** move constructor **/
  Vector(Vector<T>&& other) : n(other.n), data(other.data) {
    other.n = 0;
    delete[] other.data;
    other.data = nullptr;
  }
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

  /**change the value of all the entry as a constant**/
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

  T& operator[](int i) { return data[i]; }

  const T& operator[](int i) const { return data[i]; }

  // const T& operator[] (int i) const {
  // return T(this->data[i]);
  // }

  /** operator +**/
  template <typename T2, typename T1>
  friend Vector<typename std::common_type<T1, T2>::type>& operator+(const Vector<T1> & v1,
  const Vector<T2> & v2) {
    Vector<typename std::common_type<T, T2>::type> v(v1.n);
    for (int i = 0; i < v1.n; i++) {
      v[i] = v1[i] + v2[i];
    }
    return v;
  }

  /** operator -**/
  template <typename T2>
  auto operator-(const Vector<T2> & other) {
    Vector<typename std::common_type<T, T2>::type> v(this->n);
    for (int i = 0; i < this->n; ++i) {
      v[i] = data[i] - other[i];
    }
    return v;
  }

  template <typename V>
  Vector<typename std::common_type<V, T>::type>& operator* (const V& scalar) const {
    Vector<typename std::common_type<V, T>::type> nv(this->n);
    for (int i = 0; i < this->n; i++) {
      nv.data[i] = scalar * data[i];
    }
    return nv;
  }

  template <typename V>
  Vector<typename std::common_type<V, T>::type>& operator* (const V & scalar) {
    Vector<typename std::common_type<V, T>::type> nv(this->n);
    for (int i = 0; i < this->n; i++) {
      nv.data[i] = scalar * data[i];
    }
    return nv;
  }

  template <typename V, typename U>
  friend Vector<typename std::common_type<V, U>::type>& operator*(
      const V& scalar, const Vector<U>& vec);

  int len(void) const { return this->n; }
};

template <typename V, typename U>
Vector<typename std::common_type<V, U>::type>& operator*(const V& scalar,
                                                        const Vector<U>& vec) {
  // Vector<typename std::common_type<V, U>::type> nv(vec.n);
  // for (int i = 0; i < vec.n; i++) {
  //   nv[i] = scalar * vec.data[i];
  // }
  // return nv;
  return vec * scalar;
}
// TODO: throw exception if length differs

template <typename T, typename U>
typename std::common_type<T, U>::type dot(const Vector<T>& lhs,
                                          const Vector<U>& rhs) {
  typename std::common_type<T, U>::type sum;
  for (auto i = 0; i < lhs.n; i++) {
    sum += lhs[i] * rhs[i];
  }
  return sum;
}

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
 *
 * @tparam T
 */
template <typename T>
class Matrix {

 private:
  int r, c;
  std::map<std::pair<int, int>, T> data;

 public:
  // constructor
  Matrix(int rows, int cols) : r(rows), c(cols){};
  
  // destructor
  ~Matrix() {
    r = 0;
    c = 0;
    data.clear();
  }

  // getter
  int row(void)const{return this->r;}
  int col(void)const{return this->c;}

  // iterator
  auto begin()const{return this->data.begin();}
  auto end()const{return this->data.end();}
  auto cbegin()const{return this->data.cbegin();}
  auto cend()const{return this->data.cend();}

  // index operator
  T& operator[](const std::pair<int, int>& ij) {
    auto it = data.begin();
    for(;it != data.end(); ++it){
        int i = it->first.first;
        int j = it->first.second;
        if(i==ij.first && j==ij.second){
            return it->second;
        }
    }
    // if the queried entry does not exist
    data.insert(std::make_pair(ij, 0));
    return data.at(ij);
  }

  // index operator for constant reference
  const T& operator()(const std::pair<int, int>& ij) const{
    auto it = data.begin();
    for(;it != data.end(); ++it){
        int i = it->first.first;
        int j = it->first.second;
        if(i==ij.first && j==ij.second){
            return it->second;
        }
    }
    // if the queried entry does not exist
    throw "The queried entry does not exist!";
  }

};

//vector matrix multiplication
template <typename V, typename U>
Vector<typename std::common_type<V, U>::type> operator*(
const Matrix<V>& lhs, const Vector<U>& rhs){
    if(rhs.len() != lhs.col()){
        throw "Incomatible dimensions of the vector and the matrix!";
    }
    Vector<typename std::common_type<V, U>::type> new_vec(lhs.row());
    for(auto it = lhs.cbegin(); it != lhs.cend(); ++it){
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
    Vector p_k(length), v_k(length), h(length), x_k(length), s(length), t(length), r_k(length);

    for(int k=0; k<maxiter; ++k){
        rho_k = dot(q_0, r_k_1); // TODO: r_k_1 assign
        beta = (rho_k / rho_k_1) * (alpha / omega_k_1); //TODO: rho_k_1 assign
        p_k = r_k_1 + beta*(p_k_1 - omega_k_1 * v_k_1);
        v_k = A * p_k;
        alpha = rho_k / dot(q_0, v_k);
        h = x_k_1 + alpha * p_k;

        if(norm(b-A*h) < tol){
            x_k = h;
            x = x_k;
            return k;
        }

        s = r_k_1 - alpha * v_k;
        t = A * s;
        omega_k = dot(t, s) / dot(t, t);
        x_k = h + omega_k * s;
        x = x_k;

        if(norm(b-A*x_k) < tol){
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
void heun(const Vector<std::function<T(Vector<T> const&, T)> >& f, Vector<T>& y,
          T h, T& t){
    // Your implementation of the heun function starts here
}

template <typename T>
class SimplestWalker {
  // Your implementation of the simplest walker class starts here
};


int main(int argc, char* argv[]) {
  // Your testing of the simplest walker class starts here
  // test Matrix
  Matrix<double> M(10, 20), M1(10, 3);
  Vector<double> x({1.0, 1.1, 1.2}), y({2, 3, 4}), z({1.0f, 2.0f, 3.0f});

  try{
    M[{1,9}] = 1.0; // set value at row 0, column 0 to 1.0
    std::cout << M[{1, 9}] << std::endl;
    std::cout << M[{0, 0}] << std::endl;
    std::cout << M({1, 9}) << std::endl;
    std::cout << typeid(M.row()).name() << ' ' << M.col() << std::endl;
    Vector<double> v2 = x;
    v2 = M1*x;
    std::cout << 1 << std::endl;
    std::cout << x[2] << std::endl;
    std::cout << M({1, 1}) << std::endl;
  }
  catch(const char* msg){
    std::cerr << msg << std::endl;
  }
  return 0;
}