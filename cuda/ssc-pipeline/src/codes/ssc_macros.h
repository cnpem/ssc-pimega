#ifndef SSC_MACROS_H
#define SSC_MACROS_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <hdf5.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

extern "C" {
#include "cmdline.h"
#include <time.h>
}

#define SSC_MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define SSC_MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )

#define SSC_SQUARE(x) ((x)*(x))

#define SSC_PI 3.141592653589793238462643383279502884
#define SSC_SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )

// Size limit definitions
#define SSC_INFINITY_INT       INT_MAX
#define SSC_INFINITY_INT_NEG   INT_MIN
#define SSC_INFINITY_LONG      LONG_MAX
#define SSC_INFINITY_LONG_NEG  LONG_MIN
#define SSC_INFINITY_ULONG     ULONG_MAX
#define SSC_INFINITY_FLT       FLT_MAX
#define SSC_INFINITY_FLT_NEG  -FLT_MAX
#define SSC_INFINITY_DBL       DBL_MAX
#define SSC_INFINITY_DBL_NEG  -DBL_MAX
#define SSC_INFINITY_LDBL      LDBL_MAX
#define SSC_INFINITY_LDBL_NEG -LDBL_MAX

/* The TPBX, TPBY, TPBZ, and SSC_TPB_LINEAR_ARRAY are parameters that can be changed and may
 * interfere the performance
 */
#define TPBX 16
#define TPBY 16
#define TPBZ 4
#define SSC_TPB_LINEAR_ARRAY 128

#define DATASET   "entry/data/data/"
#define DATAFLAT  "entry/data/data/"
#define DATADARK  "entry/data/data/"

#define DISP 0

#define NUMBEROFSTREAMS 20
#define NTHREADS 8

#define FMATRIX(array, i, j, CSTRIDE) array[i*CSTRIDE + j]
#define FBLOCK(array,k,i,j,CSTRIDE,OFFSET) array[k*OFFSET+i*CSTRIDE+j]

#define BILLION 1E9
#define CLOCK  CLOCK_REALTIME
#define TIME(End,Start) (End.tv_sec - Start.tv_sec) + (End.tv_nsec-Start.tv_nsec)/BILLION

#define SSC_XY 0
#define SSC_XZ 1

#endif //SSC_MACROS_H
