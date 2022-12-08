#include <chrono>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// Matrix size
const int N = 1024;

// Matrix to be decomposed
double A[N][N];

// Lower and upper triangular matrices
double L[N][N], U[N][N];

int main() {
  // Initialize matrix A with some values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (double)i + j;
    }
  }

  // Copy matrix A to GPU memory
  thrust::device_vector<double> d_A(N * N);
  thrust::copy(A, A + N * N, d_A.begin());

  // Start timer
  auto start = std::chrono::steady_clock::now();

  // Perform LU decomposition on the GPU
  thrust::transform(
    thrust::make_zip_iterator(thrust::make_tuple(d_A.begin(), d_A.begin() + N * N)),
    thrust::make_zip_iterator(thrust::make_tuple(d_A.end(), d_A.end() + N * N)),
    L, U,
    [N](thrust::tuple<double, double> t) {
      int i = thrust::get<0>(t);
      int j = thrust::get<1>(t);

      if (i == j) {
        return L[i][j] * U[i][j];
      } else if (i < j) {
        double sum = 0;
        for (int k = 0; k < i; k++) {
          sum += L[i][k] * U[k][j];
        }
        return (A[i][j] - sum) / U[i][i];
      } else {
        double sum = 0;
        for (int k = 0; k < j; k++) {
          sum += L[i][k] * U[k][j];
        }
        return A[i][j] - sum;
      }
    }
  );

  // Stop timer
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Print result and elapsed time
  std::cout << "L = " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << L[i][j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "U = " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << U[i][j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Elapsed time: " << elapsed.count() << "ms" << std::endl;

  return 0;
}