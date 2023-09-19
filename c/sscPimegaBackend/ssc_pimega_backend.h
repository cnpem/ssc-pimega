#ifndef SSC_PIMEGA_BACKEND_H
#define SSC_PIMEGA_BACKEND_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SSC_MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define SSC_MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )
#define SSC_SQUARE(x) ((x)*(x))
#define SSC_SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )
#define SSC_PI 3.141592653589793238462643383279502884

#define N540D 3072   
	//540D project dimension
#define N135D 1536   
	//135D project dimension
#define NCHIP 256    
	//chip dimension (medipix)
#define NSTRIPES_540D 24  
	//number of stripes for 540D
#define NSTRIPES_135D 6  
	//number of stripes for 135D
#define NSMOD 6      
	//number of stripes per module
#define NMOD 4      
	//number of modules

static const float SSC_INIT_VALUE = -10.0f;

#define BILLION 1E9
#define CLOCK  CLOCK_REALTIME
#define TIME(End,Start) (End.tv_sec - Start.tv_sec) + (End.tv_nsec-Start.tv_nsec)/BILLION

typedef struct{
 
  float *output;
  float *input;
  int *ix, *iy;
  int type;
  int blocksize;
  
}ssc_pimega_backend_plan;


enum SSC_PIMEGA_TYPE {
  SSC_PIMEGA_540D = 0,
  SSC_PIMEGA_135D = 1
};


#ifdef __cplusplus
//extern "C" {
#endif

void ssc_pimega_backend_restoration(float *output,
				    float *input,
				    void *data);

void ssc_pimega_backend_restoration_540D(float *output,
					 float *input,
					 void *data);

void ssc_pimega_backend_restoration_135D(float *output,
					 float *input,
					 void *data);

void ssc_pimega_backend_set_plan(ssc_pimega_backend_plan *workspace,
				 int *ix,
				 int *iy);

void ssc_pimega_backend_create_plan(ssc_pimega_backend_plan *workspace,
				    int blocksize,
				    int type);

void ssc_pimega_backend_free_plan(ssc_pimega_backend_plan *workspace);
  
#ifdef __cplusplus
//} // extern "C" {
#endif

#endif // #ifndef SSC_PIMEGA_BACKEND_H

