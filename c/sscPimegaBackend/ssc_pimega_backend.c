#include "ssc_pimega_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NTHREADS 32

typedef void (*event_restoration)(float *, float *, void *);

event_restoration handler_restoration[] = { ssc_pimega_backend_restoration_540D ,
					    ssc_pimega_backend_restoration_135D };


void kernel_sscpimega_backend_restoration_stripe_135D(float *output,
						      float *input,
						      int *ix,
						      int *iy,
						      int z,
						      int j)
{
  int idx, idy, tx;
  size_t voxel_o, voxel_i, _voxel_i_;
  size_t pixelsAtstripe = N135D * NCHIP;
  size_t x, y, pixel; 

  for ( tx = 0; tx < pixelsAtstripe; tx++)     
    {
      //pixel for output value 
      pixel = (j * pixelsAtstripe + tx);
      idx = ix[ pixel ];
      idy = iy[ pixel ];

      //pixel for input value
      x     = tx % N135D;
      y     = tx / N135D;

      _voxel_i_ =  (NCHIP - 1 - y) * N135D + x + (5 - j) * NCHIP*N135D;
      
      for ( int k = 0 ; k < z; k++)
	{
	  voxel_o = k * N135D * N135D + idy * N135D + idx;
	  
	  voxel_i = k * N135D * N135D + _voxel_i_;
	    
	  output[ voxel_o ] = input[ voxel_i ];
	}
      
    }
}


void kernel_sscpimega_backend_restoration_stripe_540D(float *output,
						      float *input,
						      int *ix,
						      int *iy,
						      int z,
						      int s,
						      int j,
						      int module)
{
  int idx, idy, tx;
  size_t voxel_o, voxel_i, _voxel_i_;
  size_t pixelsAtstripe = N135D * NCHIP;
  size_t x, y, pixel; 
  
  for ( tx = 0; tx < pixelsAtstripe; tx++)     
    {
      //pixel for output value 
      pixel = (s * pixelsAtstripe + tx);
      idx = ix[ pixel ];
      idy = iy[ pixel ];
      
      //pixel for input value
      x     = tx % N135D;
      y     = tx / N135D;
      
      switch(module)
	{
	case 3:
	  _voxel_i_ = (NCHIP - 1 - y) * N540D + x + (5 - j) * NCHIP*N540D + N135D*N540D; 
	  
	  break;
	    
	case 2:

	  _voxel_i_ = (NCHIP - 1 - y) * N540D + (x + N135D) + (5 - j) * NCHIP*N540D + N135D*N540D;

	  break;
	    
	case 1:

	  _voxel_i_ = (NCHIP - 1 - y) * N540D + (x + N135D) + (5 - j) * NCHIP*N540D;

	  break;
	    
	default:
	  
	  _voxel_i_ =  (NCHIP - 1 - y) * N540D + x + (5 - j) * NCHIP*N540D;
	}
	
      for ( int k = 0 ; k < z; k++)
	{
	  voxel_o = k * N540D * N540D + idy * N540D + idx;
	    	    
	  voxel_i = k * N540D * N540D + _voxel_i_;
	    
	  output[ voxel_o ] = input[ voxel_i ];
	}
    }
  }



void kernel_sscpimega_backend_initialize_540D(float *output, int z)
{
  
  size_t tx, nvoxels = N540D * N540D * z;
  
  for( tx = 0 ; tx < nvoxels ; tx++)
    {
      output[ tx ] = SSC_INIT_VALUE;
    }
}

void kernel_sscpimega_backend_initialize_135D(float *output, int z)
{
  
  size_t tx, nvoxels = N135D * N135D * z;
  
  for( tx = 0 ; tx < nvoxels ; tx++)
    {
      output[ tx ] = SSC_INIT_VALUE;
    }
}


void ssc_pimega_backend_restoration_135D(float *output,
					 float *input,
					 void *data)
{
  ssc_pimega_backend_plan *workspace = (ssc_pimega_backend_plan *)data;
  
  int s, j;
  size_t pixelsAtstripe; 

  kernel_sscpimega_backend_initialize_135D( workspace->output, workspace->blocksize );
  
  for( j = 0; j < NSMOD; j++ )
    {	  
      kernel_sscpimega_backend_restoration_stripe_135D(output,
						       input,
						       workspace->ix,
						       workspace->iy,
						       workspace->blocksize,
						       j);
      
    }
}

void ssc_pimega_backend_restoration_540D(float *output,
					 float *input,
					 void *data)
{
  ssc_pimega_backend_plan *workspace = (ssc_pimega_backend_plan *)data;
  
  int s, j, module;
  size_t pixelsAtstripe; 

  clock_t begin = clock();

  kernel_sscpimega_backend_initialize_540D( workspace->output, workspace->blocksize );

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  //fprintf(stdout,"-- initialization: %lf\n", time_spent);

  for( s = 0; s < NSTRIPES_540D; s++)
    {
      j       = s % NSMOD;
      module  = s / NSMOD;
      
      kernel_sscpimega_backend_restoration_stripe_540D(output,
						       input,
						       workspace->ix,
						       workspace->iy,
						       workspace->blocksize,
						       s,
						       j,
						       module);
      
    }
   
}

void ssc_pimega_backend_create_plan(ssc_pimega_backend_plan *workspace,
				    int blocksize,
				    int type)
{
  size_t voxels, voxelsLUT;
  
  switch ( type )
    {
    case SSC_PIMEGA_540D:
      //
      voxels     = blocksize * N540D * N540D;
      voxelsLUT  = NSTRIPES_540D * N135D * NCHIP;

      workspace->output = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->input  = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->ix     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
      workspace->iy     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
      break;
	
    case SSC_PIMEGA_135D:
      //
      voxels     = blocksize * N135D * N135D;
      voxelsLUT  = NSTRIPES_135D * N135D * NCHIP;

      workspace->output = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->input  = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->ix     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
      workspace->iy     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
      break;
    
    default:
      //
      voxels     = blocksize * N540D * N540D;
      voxelsLUT  = NSTRIPES_540D * N135D * NCHIP;
	
      workspace->output = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->input  = (float *) malloc( voxels    * sizeof(float) ) ;
      workspace->ix     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
      workspace->iy     = (int *)   malloc( voxelsLUT * sizeof(int) ) ;
    }

  workspace->type      = type;
  workspace->blocksize = blocksize;
}

void ssc_pimega_backend_free_plan(ssc_pimega_backend_plan *workspace){
  
  free(workspace->output);
  free(workspace->input);
  free(workspace->ix);
  free(workspace->iy);
} 

void ssc_pimega_backend_set_plan(ssc_pimega_backend_plan *workspace,
				 int *ix,
				 int *iy)
{
  size_t voxelsLUT;

  switch( workspace->type )
    {
    case SSC_PIMEGA_540D:
      //
      voxelsLUT = NSTRIPES_540D * N135D * NCHIP;
  
      memcpy( workspace->ix, ix, voxelsLUT * sizeof(int));
      memcpy( workspace->iy, iy, voxelsLUT * sizeof(int));

      break;

    case SSC_PIMEGA_135D:
      //
      voxelsLUT = NSTRIPES_135D * N135D * NCHIP;
  
      memcpy( workspace->ix, ix, voxelsLUT * sizeof(int));
      memcpy( workspace->iy, iy, voxelsLUT * sizeof(int));

      break;

    default:
      
      voxelsLUT = NSTRIPES_540D * N135D * NCHIP;
  
      memcpy( workspace->ix, ix, voxelsLUT * sizeof(int));
      memcpy( workspace->iy, iy, voxelsLUT * sizeof(int));
      
    }
}

void ssc_pimega_backend_restoration(float *output, float *input, void *data)  
{
  ssc_pimega_backend_plan *workspace = (ssc_pimega_backend_plan *)data;
  
  //from miqueles: Pimega restoration
  //             : workspace.input  points to the raw data
  //             : workspace.output points to the processed data;
      
  handler_restoration[ workspace->type ]( output, input, data );
}

