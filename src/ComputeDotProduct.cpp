
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct.cpp

 HPCG routine
 */

 #ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#ifdef HPCG_DDOT_USE_MKL
#include "mkl.h"
#endif // HPCG_DDOT_USE_MKL
/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication
  between processes
  @param[out] isOptimized should be set to false if this routine uses the
  reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector &x, const Vector &y,
                      double &result, double &time_allreduce,
                      bool &isOptimized) {

  // This line and the next two lines should be removed and your version of
  // ComputeDotProduct should be used.
  // // isOptimized = false;
  // // return ComputeDotProduct_ref(n, x, y, result, time_allreduce);

  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double local_result = 0.0;
  double *xv = x.values;
  double *yv = y.values;

#ifdef HPCG_DDOT_USE_MKL
  loccal_result = clabs_ddot(n, xv, 1, yv, 1);
#else // HPCG_DDOT_USE_MKL
  if (yv==xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
#endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }
#endif // HPCG_DDOT_USE_MKL

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
