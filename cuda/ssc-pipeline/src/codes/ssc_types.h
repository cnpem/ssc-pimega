#ifndef SSC_TYPES_H
#define SSC_TYPES_H

#include "../codes/ssc_gpu_ids.h"
#include "ssc_macros.h"

#include <hdf5.h>

typedef void (*ssc_function)(float *, float *, float *, void *);

typedef struct{

  float *proc;
  float *read;
  float *save;

  float processing;
  float reading;
  float saving;
  float total;
  
}ssc_times;

typedef struct{

  int Nx, Ny, Nz, rankn, order;

  char *datasetName, *path, *rankS, *orderS;

  hsize_t dimsin[4];
  hid_t file, dataset, dataspace, datatype;
  
}ssc_h5; 


typedef struct {

  int blockSize, Init, Final;  
  int ei, si, nblock, nblocks;

  int *oshape, oNx, oNy, oNz; 
  
  int numberOfstreams;
  int streamSize;

  float *volume;
  float *output;
  float *outputwc; //output without crop

  bool printTiming; 
  bool save;
  char *outputPath;
  char *uuid;

  ssc_times elapsed;
  ssc_h5    *H5;
  ssc_gpus  *gpus;

  int       gpuId;

  //
  // (P)ipeline parameters: (T)hreads and (S)lots 
  //
  // This matrix can be defined by user 
  // PTS[2][1]: defined internally according to the number of GPUs
  //
  //
  int PTS[5][2] = {
    { 4,  4  },  // create queue items
    { 8 , 1  },  // read hyperslabs
    { 8,  1  },  // process hyperslabs
    { 32, 32 },  // save hyperslabs
    { 1,  1  }   // destroy queue items
  };
  
  void         *userData;
  ssc_function functionPointer;
  
}ssc_volume;


#endif // #ifndef SSC_TYPES_H

