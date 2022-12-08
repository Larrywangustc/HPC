#include <iostream>
#include <cmath>
#include <omp.h>

// Matrix size
const int N = 1024;

// Matrix of coefficients
double A[N][N];

// Vector of constants
double b[N];

// Vector of variables
double x[N];

// Number of iterations
const int MAX_ITER = 1000;

// Tolerance
const double TOL = 1e-6;

int main() {
  // Initialize matrix A and vectors b and x with some values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (double)i + j;
    }
    b[i] = (double)i;
    x[i] = 0;
  }

  // Jacobi method
  for (int k = 0; k < MAX_ITER; k++) {
    double max_error = 0;

    // Update variables in parallel
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
      double sum = 0;
      for (int j = 0; j < N; j++) {
        if (i != j) {
          sum += A[i][j] * x[j];
        }
      }
      double new_x = (b[i] - sum) / A[i][i];
      double error = std::fabs(new_x - x[i]);
      if (error > max_error) {
        max_error = error;
      }
      x[i] = new_x;
    }

    // Check for convergence
    if (max_error < TOL) {
      break;
    }
  }

  // Print result
  for (int i = 0; i < N; i++) {
    std::cout << "x[" << i << "] = " << x[i] << std::endl;
  }

  return 0;
}
