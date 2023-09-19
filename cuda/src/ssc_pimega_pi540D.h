#ifndef SSC_PIMEGA_PI540D_H
#define SSC_PIMEGA_PI540D_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define SSC_MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define SSC_MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )
#define SSC_SQUARE(x) ((x)*(x))
#define SSC_SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )
#define SSC_PI 3.141592653589793238462643383279502884

#define TPBX 16
#define TPBY 16
#define TPBZ 4
#define N540D 3072   
	//540D project dimension
#define N135D 1536   
	//135D project dimension
#define NCHIP 256    
	//chip dimension (medipix)
#define NSTRIPES 24  
	//number of stripes
#define NSMOD 6      
	//number of stripes per module
#define NMOD 4      
	//number of modules

static const float SUSP_VALUE = -1.0f;
static const float INIT_VALUE = -10.0f;

#define BILLION 1E9
#define CLOCK  CLOCK_REALTIME
#define TIME(End,Start) (End.tv_sec - Start.tv_sec) + (End.tv_nsec-Start.tv_nsec)/BILLION

typedef struct{
 
  float *temp;	
  float *output;
  float *input, *input0, *input1, *input2, *input3;
  int *ix, *iy;
  int *xmin, *xmax;
  int *ymin, *ymax;
  float *gaps;
  int fill;

}ssc_pi540D_plan;

#ifdef __cplusplus
extern "C" {
#endif
  
  void ssc_pimega_pi540D_restoration_worker(ssc_pi540D_plan *workspace,
					    int z);
					      
  void ssc_pimega_pi540D_create_gpu_plan(ssc_pi540D_plan *workspace,
					 int z);
    
  void ssc_pimega_pi540D_free_gpu_plan(ssc_pi540D_plan *workspace);
  
  void ssc_pimega_pi540D_set_gpu_data(ssc_pi540D_plan *workspace,
				      float *input0,
				      float *input1,
				      float *input2,
				      float *input3,
				      int *ix,
				      int *iy,
				      int *xmin,
				      int *xmax,
				      int *ymin,
				      int *ymax,
				      int z,
				      int fill);
    
  void ssc_pimega_pi540D_get_gpu_data(float *host,
				      ssc_pi540D_plan *workspace,
				      int z);
    
  void ssc_pimega_pi540D_create_gpu_plan(ssc_pi540D_plan *workspace, int z);

  void ssc_pimega_pi540D_set_gpu_lut(ssc_pi540D_plan *workspace,
				     int *ix,
				     int *iy,
				     int *xmin,
				     int *xmax,
				     int *ymin,
				     int *ymax,
				     int fill);

  void ssc_pimega_pi540D_set_gpu_data_block(ssc_pi540D_plan *workspace,
					    float *input,
					    int blocksize);

  
  void ssc_pimega_pi540D_restoration_worker(ssc_pi540D_plan *workspace, int z);

  void ssc_pimega_pi540D_get_gpu_data_block(float *host, ssc_pi540D_plan *workspace, int z);

  void ssc_pimega_pi540D_restoration_worker_block(ssc_pi540D_plan *workspace, int z);

  void ssc_pimega_pi540D_create_gpu_plan_block(ssc_pi540D_plan *workspace, int blocksize);

  int  ssc_pimega_pi540D_backward_pipeline( int *ishape,
		 			    char *path,
					    char *outputPath,
					    char *volOrder,
					    char *rank,
					    char *datasetName,
					    int ngpus,
					    int *gpu,
					    int *shape,
					    int Init,
					    int Final,
					    int blockSize,
					    int timing,
					    int saving,
					    int *ix,
					    int *iy,
					    int *xmin,
					    int *xmax,
					    int *ymin,
					    int *ymax,
					    int *center,
					    int roi,
					    float *flat,
					    float *empty,
					    float *mask,
				  	    float *daxpyimg,
					    float daxpycon,
					    int susp,
					    char *uuid,
					    float *gaps,
					    int fill);

#ifdef __cplusplus
} // extern "C" {
#endif

#endif // #ifndef SSC_PIMEGA_PI540D_H

