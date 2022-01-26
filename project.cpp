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
  /**
   * @brief default constructor, set length to 0
   */
  explicit Vector() : n(0), data(nullptr) {}

  /**
   * @brief constructor with given length
   */
  explicit Vector(int a) : n(a), data(new T[a]) {}

  /**
   * @brief constructor with initializer list
   */
  explicit Vector(const std::initializer_list<T>& list)
      : Vector((int)list.size()) {
    std::uninitialized_copy(list.begin(), list.end(), data);
  }

  /**
   * @brief copy constructor
   */
  explicit Vector(const Vector& other) : n(other.n) {
    for (auto i = 0; i < other.n; i++) {
      data[i] = other.data[i];
    }
  }

  /**
   * @brief move constructor
   */
  explicit Vector(const Vector&& other) : n(other.n), data(other.data) {
    other.n = 0;
    delete[] other.data;
    other.data = nullptr;
  }

  /**
   * @brief destructor
   */
  ~Vector() {
    n = 0;
    delete[] data;
    data = nullptr;
  }

  /** copy assignment **/
  const Vector<T>& operator=(const Vector<T>& other) {
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
  const Vector<T>& operator=(Vector<T>&& other) {
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

  // const T& operator[] (int i) const {
  // return T(this->data[i]);
  // }

  /** operator +**/
  template <typename T2>
  auto operator+(Vector<T2> other) {
    Vector<typename std::common_type<T, T2>::type> v(this->n);
    for (int i = 0; i < this->n; ++i) {
      v[i] = data[i] + other[i];
    }
    return v;
  }

  /** operator -**/
  template <typename T2>
  auto operator-(Vector<T2> other) {
    Vector<typename std::common_type<T, T2>::type> v(this->n);
    for (int i = 0; i < this->n; ++i) {
      v[i] = data[i] - other[i];
    }
    return v;
  }

  template <typename V>
  Vector<typename std::common_type<V, T>::type>& operator*(const V& scalar) {
    Vector<typename std::common_type<V, T>::type> new_vec(this->n);
    for (int i = 0; i < this->n; i++) {
      new_vec[i] = scalar * data[i];
    }
    return new_vec;
  }

  template <typename V, typename U>
  friend Vector<typename std::common_type<V, U>::type> operator*(
      const V& scalar, const Vector<U> vec) {
    Vector<typename std::common_type<V, U>::type> new_vec(vec.n);
    for (int i = 0; i < vec.n; i++) {
      new_vec[i] = scalar * vec[i];
    }
    return new_vec;
  }

  int len(void) { return this->n; }
};
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

template <typename T>
class Matrix {
  private:
    int r, c;
    std::map<std::pair<int, int>, T> data;
  
  public:
    Matrix(int rows, int cols) : r(rows), c(cols);
    ~Matrix(){
      rows = 0;
      cols = 0;
      data.clear();
    }

    T& operator[] const (const std::pair<int, int> & ij) {

    }

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
  // test vector
  Vector<double> x({1.0, 1.1, 1.2}), y({2, 3, 4}), z({1.0f, 2.0f, 3.0f}), w;
  w = y * 3;
  // w = 2 * x + y  * 3;
  return 0;
}
