#ifndef __IPIC_DEFS_H__
#define __IPIC_DEFS_H__

typedef unsigned long long longid;
//typedef uint64_t longid; // requires #include <stdint.h>

// comment this out if OpenMP is not installed on your system.
#define USING_OMP

// uncomment the following line to use parallel hdf5
//#define USING_PARALLEL_HDF5

// use precprocessor to remove former MPI_Barrier() calls.
//#define MPI_Barrier(args...)
#define former_MPI_Barrier(args...)

#define ipicMPI_Allreduce(args...) \
  { \
    static int count=0; \
    dprint(count++); \
    MPI_Allreduce(## args); \
  }

// determine the width of the vector unit
//
#if defined(__MIC__)
  const int VECBITS = 512;
#elif defined(__AVX__)
  const int VECBITS = 256;
#elif defined(__SSE_)
  const int VECBITS = 128;
#else
  const int VECBITS = 64;
#endif
const int VECBYTES = VECBITS/8;

// the number of doubles that fill a vector
const int DVECWIDTH = VECBYTES/sizeof(double);
const int SVECWIDTH = VECBYTES/sizeof(float);
//#define SINGLE_PRECISION_PCLS
//
// single precision does not seem to help on the MIC
typedef double pfloat;
//#ifdef SINGLE_PRECISION_PCLS
//  typedef float pfloat;
//  #ifdef __MIC__
//    #define VECTOR_WIDTH 16
//  #else
//    #define VECTOR_WIDTH 8
//  #endif
//#else
//  #ifdef __MIC__
//    #define VECTOR_WIDTH 8
//  #else
//    #define VECTOR_WIDTH 4
//  #endif
//  typedef double pfloat;
//#endif

#endif
