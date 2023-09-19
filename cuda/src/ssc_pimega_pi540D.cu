#include "../ssc-pipeline/src/ssc_pipeline.h"
#include "ssc_pimega_pi540D.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

static const float SUSPV = -1.0f;
static const float INIV = -10.0f;

extern "C" {
  __global__ void kernel_pi540D_backward_stripe(float *d_output,
						float *d_input,
						int *ix,
						int *iy,
					     	int z,
						int s,
						int j)
  {
    int idx, idy, tx = threadIdx.x + blockIdx.x*blockDim.x;
    size_t voxel_o, voxel_i;
    size_t pixelsAtstripe = N135D * NCHIP;
    size_t x, y, pixel; 
    
    if( (tx < pixelsAtstripe) )
      {
	//pixel for output value 
	pixel = (s * pixelsAtstripe + tx);
       	idx = ix[ pixel ];
	idy = iy[ pixel ];
	
	//pixel for input value
	x     = tx % N135D;
	y     = tx / N135D;
	
	for ( int k = 0 ; k < z; k++)
	  {
	    voxel_o = k * N540D * N540D + idy * N540D + idx;
	    
	    voxel_i = (k * N135D * N135D + (NCHIP - y) * N135D + x + (5 - j) * NCHIP*N135D);
	    
	    d_output[ voxel_o ] = d_input[ voxel_i ];
	  }
      }
  }
}


extern "C" {
  __global__ void kernel_pi540D_backward_stripe_block(float *d_output,
						      float *d_input,
						      int *ix,
						      int *iy,
						      int z,
						      int s,
						      int j,
						      int module)
  {
    int idx, idy, tx = threadIdx.x + blockIdx.x*blockDim.x;
    size_t voxel_o, voxel_i, _voxel_i_;
    size_t pixelsAtstripe = N135D * NCHIP;
    size_t x, y, pixel; 

    if( (tx < pixelsAtstripe) )
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
	    
	    d_output[ voxel_o ] = d_input[ voxel_i ];

          }
      }
  }
}


extern "C" {
  __global__ void kernel_pi540D_fill_missing(float *d_output,
					     float *d_input,
					     int blockSize,
					     int xmin,
					     int xmax,
					     int ymin,
					     int ymax)
  {      
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
    size_t voxel;
    int step = 0;

    if ( (tx < xmax-step) && (tx >= xmin+step) && (ty < ymax-step) && (ty >= ymin+step) && (tz < blockSize) )
      {
	voxel = tz * N540D * N540D + ty * N540D + tx;
	
	if ( fabs( d_input[voxel] - INIV ) < 1e-5 ) // dinput[voxel] == INIV
	  {
	    int v0, v1, v2, v3;
	    //int ptx = tx, pty = ty, ntx = tx, nty = ty;		
	
	    step = 1;
	    v0 = tz * N540D * N540D + (ty+step)*N540D + (tx+step);	    
	    v1 = tz * N540D * N540D + (ty-step)*N540D + tx-step;
	    v2 = tz * N540D * N540D + (ty-step)*N540D + tx+step;
	    v3 = tz * N540D * N540D + (ty+step)*N540D + tx-step;
	    
	    
	    /*
	    v0 = tz * N540D * N540D + (ty+step)*N540D + tx;	    
	    v1 = tz * N540D * N540D + (ty)*N540D + tx+step;
	    v2 = tz * N540D * N540D + (ty-step)*N540D + tx;
	    v3 = tz * N540D * N540D + (ty)*N540D + tx-step;
	    */
	   
	    /* 
	    do {
	      	pty += 1;
		v0 = tz * N540D * N540D + (pty)*N540D + tx;	      
	    } while ( d_input[v0] < SUSPV && pty < ymax);	    

	    do{
	      	ptx += 1;
		v1 = tz * N540D * N540D + (ty)*N540D + ptx;	    
	    }while ( d_input[v1] < SUSPV && ptx < xmax);

	    do{
	      	nty -= 1;
		v2 = tz * N540D * N540D + (nty)*N540D + tx;	    
	    }while ( d_input[v2] < SUSPV && nty >= ymin );

	    do{
    		ntx -= 1;
		v3 = tz * N540D * N540D + (ty)*N540D + ntx;	    
	    }while ( d_input[v3] < SUSPV && ntx >= xmin);
	    

	    v0 = tz * N540D * N540D + (pty)*N540D + tx;	    
	    v1 = tz * N540D * N540D + (ty)*N540D + ptx;
	    v2 = tz * N540D * N540D + (nty)*N540D + tx;
	    v3 = tz * N540D * N540D + (ty)*N540D + ntx;
	    */

	    d_output[ voxel ] = ( 0.5 * (SSC_SIGN( d_input[v0] ) + 1 ) * d_input[v0] +
				  0.5 * (SSC_SIGN( d_input[v1] ) + 1 ) * d_input[v1] +
				  0.5 * (SSC_SIGN( d_input[v2] ) + 1 ) * d_input[v2] +
				  0.5 * (SSC_SIGN( d_input[v3] ) + 1 ) * d_input[v3] ) / 4;
	    
	  }
	else
	  {
	    d_output[ voxel ] = d_input[voxel ];
	  }
	
      }
    
  }
}



extern "C" {
  __global__ void kernel_pi540D_setNegative(float *d_input, float *mask, int z){
    
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
    size_t pxl, voxel;
    
    if( (tx < N540D) && (ty < N540D) && (tz < z) )
      {
	voxel = tz * N540D * N540D + ty * N540D + tx;
        pxl   = ty * N540D + tx;

	if ( mask[ pxl ] > 0 || d_input[voxel] < 0 )
	{
	    d_input[ voxel ] = SUSPV;
      	}
      }
  }
}


extern "C" {
  __global__ void kernel_pi540D_memcpy(float *d_output, float *d_input, int z){
    
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
    size_t voxel;
    
    if( (tx < N540D) && (ty < N540D) && (tz < z) )
      {
	voxel = tz * N540D * N540D + ty * N540D + tx; 
	
	d_output[ voxel ] = d_input[ voxel ];
      }
  }
}



extern "C" {
  __global__ void kernel_pi540D_initialize(float *d_output, int z){
    
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
    size_t voxel;
    
    if( (tx < N540D) && (ty < N540D) && (tz < z) )
      {
	voxel = tz * N540D * N540D + ty * N540D + tx; 
	
	d_output[ voxel ] = INIV;
      }
  }
}


extern "C" {

  void function_pi540D_initialize(float *d_output, int z){
    
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock((int)ceil(N540D/threadsPerBlock.x),
		   (int)ceil(N540D/threadsPerBlock.y),
		   (int)ceil(z/threadsPerBlock.z));
    
    kernel_pi540D_initialize<<<gridBlock, threadsPerBlock>>>(d_output, z);

  }
}


extern "C" {

  void ssc_pimega_pi540D_restoration_worker(ssc_pi540D_plan *workspace, int z){
    
    int s, j, module;
    size_t pixelsAtstripe; 
    
    float *d_input[] = {workspace->input0, workspace->input1, workspace->input2, workspace->input3};

    function_pi540D_initialize(workspace->output, z);
    
    pixelsAtstripe = N135D*NCHIP;
    
    dim3 threadsPerBlock(NCHIP);
    dim3 gridBlock((int)ceil(pixelsAtstripe/threadsPerBlock.x));
		   
    for( module = 0; module < NMOD; module ++)
      {
	for( j = 0; j < NSMOD; j++ )
	  {
	    s = j + NSMOD * module;
	    
	    kernel_pi540D_backward_stripe<<<gridBlock, threadsPerBlock>>>(workspace->output,
									  d_input[module],
								    	  workspace->ix,
									  workspace->iy,
					       			  	  z,
									  s,
									  j);
	    

	   
	  }
      }
  }
}

extern "C" {

  void ssc_pimega_pi540D_restoration_worker_block(ssc_pi540D_plan *workspace, int z){
    
    
    int s, j, module;
    size_t pixelsAtstripe; 
    
    function_pi540D_initialize(workspace->output, z);
    
    pixelsAtstripe = N135D*NCHIP;
    
    dim3 threadsPerBlock(TPBX);
    dim3 gridBlock((int)ceil(pixelsAtstripe/threadsPerBlock.x)); 

    
    dim3 threadsPerBlockM(TPBX,TPBY,TPBZ);
    dim3 gridBlockM((int)ceil((N540D)/threadsPerBlockM.x),
		    (int)ceil((N540D)/threadsPerBlockM.y),
		    (int)ceil(z/threadsPerBlockM.z));
    
    
    for( module = 0; module < NMOD; module ++)
      {
	for( j = 0; j < NSMOD; j++ )
	  {
	    s = j + NSMOD * module;

	    kernel_pi540D_backward_stripe_block<<<gridBlock, threadsPerBlock>>>(workspace->output,
										workspace->input,
										workspace->ix,
										workspace->iy,
										z,
										s,
										j,
										module);
	    cudaDeviceSynchronize();

	    if (workspace->fill == 1)
	      {
		kernel_pi540D_fill_missing<<<gridBlockM, threadsPerBlockM>>>(workspace->temp,
									     workspace->output,
									     z,
									     workspace->xmin[s],
									     workspace->xmax[s],
									     workspace->ymin[s],
									     workspace->ymax[s]);
		
		cudaDeviceSynchronize();

		kernel_pi540D_memcpy<<<gridBlockM, threadsPerBlockM>>>(workspace->output, workspace->temp, z);
	      
		cudaDeviceSynchronize();
	      }
	  }
      }
  }
}


extern "C" {
  void ssc_pimega_pi540D_create_gpu_plan(ssc_pi540D_plan *workspace, int z){
    
    size_t voxels = N540D * N540D * z, voxelsModule = (N135D * N135D * z);
    size_t pixelsLUT = NSTRIPES * N135D * NCHIP;
    
    cudaMalloc((void**)&workspace->temp, voxels*sizeof(float)) ;
    cudaMalloc((void**)&workspace->output, voxels*sizeof(float)) ;
    cudaMalloc((void**)&workspace->input0, voxelsModule*sizeof(float)) ;
    cudaMalloc((void**)&workspace->input1, voxelsModule*sizeof(float)) ;
    cudaMalloc((void**)&workspace->input2, voxelsModule*sizeof(float)) ;
    cudaMalloc((void**)&workspace->input3, voxelsModule*sizeof(float)) ;
    
    cudaMalloc((void**)&workspace->ix, pixelsLUT*sizeof(int)) ;
    cudaMalloc((void**)&workspace->iy, pixelsLUT*sizeof(int)) ;

    workspace->xmin = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->xmax = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->ymin = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->ymax = (int *) malloc( NSTRIPES * sizeof(int) );
  }
}


extern "C" {
  void ssc_pimega_pi540D_create_gpu_plan_block(ssc_pi540D_plan *workspace, int blocksize){
    
    size_t voxels = N540D * N540D * blocksize;
    size_t pixelsLUT = NSTRIPES * N135D * NCHIP;
    
    cudaMalloc((void**)&workspace->output, voxels*sizeof(float)) ;
    cudaMalloc((void**)&workspace->input, voxels*sizeof(float)) ;
    cudaMalloc((void**)&workspace->temp, voxels*sizeof(float)) ;
    
    cudaMalloc((void**)&workspace->ix, pixelsLUT*sizeof(int)) ;
    cudaMalloc((void**)&workspace->iy, pixelsLUT*sizeof(int)) ;

    workspace->xmin = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->xmax = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->ymin = (int *) malloc( NSTRIPES * sizeof(int) );
    workspace->ymax = (int *) malloc( NSTRIPES * sizeof(int) );
  }
}


extern "C" {
  void ssc_pimega_pi540D_free_gpu_plan(ssc_pi540D_plan *workspace){

    cudaFree(workspace->temp);
    cudaFree(workspace->output);
    cudaFree(workspace->input0);
    cudaFree(workspace->input1);
    cudaFree(workspace->input2);
    cudaFree(workspace->input3);
    cudaFree(workspace->ix);
    cudaFree(workspace->iy);

    free(workspace->xmin);
    free(workspace->xmax);
    free(workspace->ymin);
    free(workspace->ymax);
  } 
}


extern "C" {
  void ssc_pimega_pi540D_free_gpu_plan_block(ssc_pi540D_plan *workspace){

    cudaFree(workspace->temp);
    cudaFree(workspace->output);
    cudaFree(workspace->input);
    cudaFree(workspace->ix);
    cudaFree(workspace->iy);
    
    free(workspace->xmin);
    free(workspace->xmax);
    free(workspace->ymin);
    free(workspace->ymax);
  } 
}


extern "C" {
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
				      int fill){
    
    size_t voxelsModule = (N135D * N135D * z);
    size_t pixelsLUT = NSTRIPES * N135D * NCHIP;

    workspace->fill = fill;

    //----------------------
    //copying data to device
    cudaMemcpy(workspace->input0, input0, voxelsModule * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->input1, input1, voxelsModule * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->input2, input2, voxelsModule * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->input3, input3, voxelsModule * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(workspace->ix, ix, pixelsLUT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->iy, iy, pixelsLUT * sizeof(int), cudaMemcpyHostToDevice);
    
    memcpy( workspace->xmin, xmin, NSTRIPES * sizeof(int));
    memcpy( workspace->xmax, xmax, NSTRIPES * sizeof(int));
    memcpy( workspace->ymin, ymin, NSTRIPES * sizeof(int));
    memcpy( workspace->ymax, ymax, NSTRIPES * sizeof(int));
  }  
}


extern "C" {
  void ssc_pimega_pi540D_set_gpu_lut(ssc_pi540D_plan *workspace,
				     int *ix,
				     int *iy,
				     int *xmin,
				     int *xmax,
				     int *ymin,
				     int *ymax,
				     int fill){
    
    size_t pixelsLUT = NSTRIPES * N135D * NCHIP;

    workspace->fill = fill;

    //----------------------
    //copying data to device
    
    cudaMemcpy(workspace->ix, ix, pixelsLUT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(workspace->iy, iy, pixelsLUT * sizeof(int), cudaMemcpyHostToDevice);

    memcpy( workspace->xmin, xmin, NSTRIPES * sizeof(int));
    memcpy( workspace->xmax, xmax, NSTRIPES * sizeof(int));
    memcpy( workspace->ymin, ymin, NSTRIPES * sizeof(int));
    memcpy( workspace->ymax, ymax, NSTRIPES * sizeof(int));
 
  }  
}


extern "C" {
  void ssc_pimega_pi540D_set_gpu_data_block(ssc_pi540D_plan *workspace,
					    float *input,
					    int blocksize){
    
    size_t voxels = (N540D * N540D * blocksize);


    //----------------------
    //copying data to device

    cudaMemcpy(workspace->input, input, voxels * sizeof(float), cudaMemcpyHostToDevice);
  }  
}


extern "C" {
  void ssc_pimega_pi540D_get_gpu_data(float *host, ssc_pi540D_plan *workspace, int z){
    
    size_t voxels = N540D * N540D * z;

    cudaMemcpy( host , workspace->output, voxels * sizeof(float), cudaMemcpyDeviceToHost); 
    
  }
}

extern "C" {
  void ssc_pimega_pi540D_get_gpu_data_block(float *host, ssc_pi540D_plan *workspace, int z){
    
    size_t voxels = N540D * N540D * z;
    
    cudaMemcpy( host , workspace->output, voxels * sizeof(float), cudaMemcpyDeviceToHost); 
    
  }
}


//==============================================
//
//
// Functions to read and restore a given HDF5
// using the superscalar pipeline given in
// <ssc_pipeline.h>
//
//===============================================


extern "C" {

  typedef struct {

    int *ix;
    int *iy;
    int *xmin, *xmax, *ymin, *ymax;
    int blocksize;
    int fill;
    bool timing;

    int susp;
    int *center;
    int roi;
    float *flat;
    float *empty;
    float *mask;
    float *gaps;
    float *daxpyimg;
    float daxpycon;
    
  }localparams_t;


  __global__ void kernel_get_preprocessing( float *d_input,
					    float *d_flat,
					    float *d_empty,
					    float *d_mask,
					    float *d_daxpyimg,
					    float daxpycon,
					    int Nx,
					    int Ny,
					    int blocksize)
  {
    int i, j, z;
    int vxl, pxl;
  
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);
    
    if ( ((i)<Nx) && ((j)<Ny) && ((z)<blocksize)  ){

      vxl = z * (Ny * Nx) + j * Nx + i;
      pxl = j * Nx + i;
      
      if ( d_empty[ pxl ] > 0 || d_mask[ pxl ] > 0 )
	{
	  d_input[ vxl ] = - 1.0;
	}
      else
	{
	  if ( d_flat[pxl] == 0 || isnan( d_flat[pxl]  ))
	    {
	      d_input[ vxl ] = -1.0;
	    }
	  else
	    {
	      //d_input[ vxl ] = d_input[ vxl ] * d_flat[ pxl ] ;

	      
	      //---- daxpy
	      d_input[ vxl ] = d_input[ vxl ] * d_flat[pxl] + daxpycon * d_daxpyimg[ pxl ] * d_flat[ pxl ];
      	    
	      if ( d_input[ vxl ] < 0 )
	      {
	      	d_input [ vxl ] = -1;
	      }
	      
	    }

	}
      
    }
  }


  __global__ void kernel_set_suspicious( float *d_input,
					 int Nx,
					 int Ny,
					 int blocksize,
					 int susp)
  {
    int J=256, M=6, P=6;
    int i, j, z;
    int vxl;
    int left, right, up, down, _from_, _to_;
    
    i = (blockDim.x * blockIdx.x + threadIdx.x); //x
    j = (blockDim.y * blockIdx.y + threadIdx.y); //y
    z = (blockDim.z * blockIdx.z + threadIdx.z); //z
    
    if ( ((i)<Nx) && ((j)<Ny) && ((z)<blocksize)  ){

      if( susp > 0 )
	{
	  vxl = z * (Ny * Nx) + j * Nx + i;

	  //====================
	  // module:0 (top/left)
	  if ( i < N135D && j < N135D )
	    {
	      // columns
	      for (int x = 0; x < M-1; x ++ )
		{
		  left  = (x+1) * J;
		  right = SSC_MIN( (x+1)*J, J*M );

		  _from_ = left - susp + 1;
		  _to_   = left + 1;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }

		  _from_ = right;
		  _to_   = right + susp - 1;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}

	      //rows
	      for (int y = 0; y < P; y ++)
		{
		  up   = y*J - 1;
		  down = y*J;
		  
		  _from_ = up-susp+1;
		  _to_   = up+1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		    
		  _from_ = down;
		  _to_   = down + susp -1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}

	    }
	  
	  //======================
	  // module:1 (top/right)
	  if ( i >= N135D && j < N135D )
	    {
	      // columns
	      for (int x = 0; x < M-1; x ++ )
		{
		  left  = (x+1) * J + N135D;
		  right = SSC_MIN( (x+1)*J, J*M ) + N135D;

		  _from_ = left - susp + 1;
		  _to_   = left + 1;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }

		  _from_ = right;
		  _to_   = right + susp - 1 ;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}

	      //rows
	      for (int y = 0; y < P; y ++)
		{
		  up   = y*J - 1;
		  down = y*J;
		  
		  _from_ = up-susp+1;
		  _to_   = up+1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		    
		  _from_ = down;
		  _to_   = down + susp -1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}
	      
	    }

	  //========================
	  // module:2 (bottom/right)
	  if ( i >= N135D && j >= N135D )
	    {
	      // columns
	      for (int x = 0; x < M-1; x ++ )
		{
		  left  = (x+1) * J + N135D;
		  right = SSC_MIN( (x+1)*J, J*M ) + N135D;

		  _from_ = left - susp + 1;
		  _to_   = left + 1;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }

		  _from_ = right;
		  _to_   = right + susp - 1 ;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}

	      //rows
	      for (int y = 0; y < P; y ++)
		{
		  up   = y*J - 1 + N135D;
		  down = y*J + N135D; 
		  
		  _from_ = up-susp+1;
		  _to_   = up+1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		    
		  _from_ = down;
		  _to_   = down + susp -1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}
	    }
	  
	  //=======================
	  // module:3 (bottom/left)
	  if ( i < N135D && j>= N135D )
	    {
	      // columns
	      for (int x = 0; x < M-1; x ++ )
		{
		  left  = (x+1) * J ;
		  right = SSC_MIN( (x+1)*J, J*M );

		  _from_ = left - susp + 1;
		  _to_   = left + 1;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }

		  _from_ = right;
		  _to_   = right + susp - 1 ;
		  
		  if ( i >= _from_ && i < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}

	      //rows
	      for (int y = 0; y < P; y ++)
		{
		  up   = y*J - 1 + N135D;
		  down = y*J + N135D;
		  
		  _from_ = up-susp+1;
		  _to_   = up+1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		    
		  _from_ = down;
		  _to_   = down + susp -1;

		  if ( j >= _from_ && j < _to_ )
		    {
		      d_input[ vxl ] = SUSPV;
		    }
		}
	      
	      
	    }
	  
	}
      
      
    }
  }
   
  void get_cropping(	float *output,
			float *input,
			int blocksize,
			int roi,
			int *center)
  {
    int i, j, z, w, r;
    int px, py;
    int size = 2 * roi;
    int voxels = size*size*blocksize;
    
    
    for (  w = 0; w < voxels; w++)
      {
	z = w / (size*size);
	r = w % (size*size);

	j = r / size;
	i = r % size; 

	py = center[0] + (roi - 1 - j);
	px = center[1] + (roi - 1 - i);
	    
	output[ z * (size*size) + (size-1-i) * size + (size-1-j)] = input[ z * N540D*N540D + N540D * px + py];
       
      }
  }

  __global__ void kernel_get_cropping(	float *d_output,
				  	float *d_input,
				 	int Nx,
				  	int Ny,
				  	int blocksize,
				  	int roi,
				  	int center0,
					int center1)
  {
    int i, j, z;
    int px, py;
  
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);

    int size = 2 * roi;
    
    if ( ((i)<size) && ((j) < size) && ((z)<blocksize)  ){

      py = center0 + (roi - 1 - j);
      px = center1 + (roi - 1 - i);

      d_output[ z * (size * size) + (size-1-i) * size + (size-1-j)] = d_input[ z * (Nx*Ny) + Ny * px + py]; 
      
    }
  }


  void SSCcuLOG(void)
  {
    checkCudaErrors(cudaPeekAtLastError());
  } 
 
 
  void processBlock(float *output, float *outputwc, float *input, void *data)  
  {
    localparams_t *P = (localparams_t *)data;
    
    ssc_pi540D_plan workspace;
    
    //size_t available, total;
    //cudaMemGetInfo(&available, &total);
    //fprintf(stderr,"\n Available: %ld \ Total: %ld\n\n", available, total);

    int blocksize  = P->blocksize;
    int *ix        = P->ix;
    int *iy        = P->iy;
    int *xmin      = P->xmin;
    int *xmax      = P->xmax;
    int *ymin      = P->ymin;
    int *ymax      = P->ymax;
    int *center    = P->center;
    int roi        = P->roi;
    float *flat    = P->flat;
    float *empty   = P->empty;
    float *mask    = P->mask;
    int susp       = P->susp;
    float *daxpyimg= P->daxpyimg;
    float daxpycon = P->daxpycon;
    float *gaps    = P->gaps;
    int fill       = P->fill;


    ssc_pimega_pi540D_create_gpu_plan_block( &workspace, blocksize );
    SSCcuLOG();

    ssc_pimega_pi540D_set_gpu_lut( &workspace, ix,iy, xmin, xmax, ymin, ymax, fill);
    SSCcuLOG();

    float *d_input, *d_flat, *d_empty, *d_mask, *d_daxpyimg, *d_gaps;

    cudaMalloc((void**)&d_input, blocksize * N540D * N540D * sizeof(float)) ;
    SSCcuLOG();
   
    cudaMalloc((void**)&d_gaps, N540D * N540D * sizeof(float)) ;
    SSCcuLOG();
    
    cudaMalloc((void**)&d_flat, N540D * N540D * sizeof(float)) ;
    SSCcuLOG();
    
    cudaMalloc((void**)&d_empty, N540D * N540D * sizeof(float)) ;
    SSCcuLOG();
    
    cudaMalloc((void**)&d_mask, N540D * N540D * sizeof(float)) ;
    SSCcuLOG();

    cudaMalloc((void**)&d_daxpyimg, N540D * N540D * sizeof(float)) ;
    SSCcuLOG();

    cudaMemcpy( (void *) d_input, (void *) input, blocksize * N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();
    
    cudaMemcpy(d_gaps,  gaps, N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();
    
    cudaMemcpy(d_flat,  flat, N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();
    
    cudaMemcpy(d_empty,empty, N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();
    
    cudaMemcpy(d_mask,  mask, N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();
    
    cudaMemcpy(d_daxpyimg,  daxpyimg, N540D * N540D * sizeof(float), cudaMemcpyHostToDevice);
    SSCcuLOG();


    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);

    dim3 gridBlockPre((int)ceil((N540D)/threadsPerBlock.x),
		      (int)ceil((N540D)/threadsPerBlock.y),
		      (int)ceil(blocksize/threadsPerBlock.z));
    
    kernel_pi540D_memcpy<<<gridBlockPre, threadsPerBlock>>>(workspace.input, d_input, blocksize);
    SSCcuLOG();
    
    //@miqueles: Set suspicious pixels
    //
    
    kernel_set_suspicious<<<gridBlockPre, threadsPerBlock>>>(workspace.input, N540D, N540D, blocksize, susp);
    SSCcuLOG();
    
    //@miqueles: Preprocessing data: 
    //           a) data = data - alpha * img (daxpy operation)
    //		 b) data = data * flat
    //		 c) data[ empty > 0 ] = -1
    //		 d) data[ mask > 0 ] = -1    
    //
    
    kernel_get_preprocessing<<<gridBlockPre, threadsPerBlock>>>(workspace.input, d_flat, d_empty, d_mask, d_daxpyimg, daxpycon, N540D, N540D, blocksize);
    SSCcuLOG();
    

    //@miqueles: Pimega restoration
    //         : workspace.input  points to the preprocessed data, existing at the device
    //         : workspace.output points to the processed data, existing at the device;
    
    ssc_pimega_pi540D_restoration_worker_block( &workspace, blocksize );
    SSCcuLOG();
    
    //@miqueles: set gaps
    
    kernel_pi540D_setNegative<<<gridBlockPre, threadsPerBlock>>>(workspace.output, d_gaps, blocksize);
    SSCcuLOG();
    
    
    //@miqueles: Cropping the data
    //

    float *d_output;

    cudaMalloc((void**)&d_output, (roi*2)*(roi*2)*blocksize * sizeof(float)) ;
    
    dim3 gridBlockCrop((int)ceil((roi*2)/threadsPerBlock.x),
		       (int)ceil((roi*2)/threadsPerBlock.y),
		       (int)ceil(blocksize/threadsPerBlock.z));
    
    kernel_get_cropping<<<gridBlockCrop, threadsPerBlock>>>(d_output, workspace.output, N540D, N540D, blocksize, roi, center[0], center[1]);
    SSCcuLOG(); 
   
    cudaMemcpy( output, d_output, (2*roi)*(2*roi)* blocksize * sizeof(float), cudaMemcpyDeviceToHost );
    SSCcuLOG();
    
    cudaFree(d_output);

    //float *houtput = (float *)malloc(N540D*N540D*blocksize*sizeof(float));
    /*
    cudaMemcpy( outputwc, workspace.output, N540D*N540D*blocksize*sizeof(float), cudaMemcpyDeviceToHost);

    SSCcuLOG();

    get_cropping( output, outputwc , blocksize, roi, center);
    */
    //free(houtput);
    
    ///////////////    
    
    cudaFree(d_input);
    cudaFree(d_flat);
    cudaFree(d_empty);
    cudaFree(d_mask);
    cudaFree(d_daxpyimg);
    cudaFree(d_gaps);

    ssc_pimega_pi540D_free_gpu_plan_block( &workspace );    
  }  

  int ssc_pimega_pi540D_backward_pipeline( int *ishape,
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
					   int fill)
  {  
    ssc_volume workspace;
    ssc_h5 H5;
    ssc_gpus *gpus;
    int *oshape;

    bool time, save;

    if( timing == 1 )
      time = true;
    else
      time = false;
  
    if( saving == 1 )
      save = true;
    else
      save = false;

    gpus = ssc_configure_gpus( ngpus, gpu );

    oshape = ssc_configure_oshape( shape );
 
    ssc_create_h5info( &H5, rank, path, datasetName, volOrder);
  
    ssc_create_workspace(&workspace,
			 &H5,
			 gpus,
			 oshape,
			 blockSize,
			 outputPath,
			 uuid,
			 time,
			 save,
			 Init,
			 Final);

    
    
    localparams_t data;
    data.ix        = ix;
    data.iy        = iy;
    data.xmin      = xmin;
    data.xmax      = xmax;
    data.ymin      = ymin;
    data.ymax      = ymax;
    data.blocksize = blockSize;
    data.timing    = time;
    data.center    = center;
    data.roi       = roi;
    data.flat      = flat;
    data.mask      = mask;
    data.empty     = empty;
    data.susp      = susp;
    data.daxpyimg  = daxpyimg;
    data.daxpycon  = daxpycon;
    data.gaps      = gaps;
    data.fill      = fill;

    ishape[0] = H5.Nx;
    ishape[1] = H5.Ny;
    ishape[2] = H5.Nz;

    int nimgs;

    if ( H5.order == SSC_XY )
    {
	if (Final == -1)
	{
       	   nimgs = H5.Nz;
	}
	else
	{
	   nimgs = Final;
	}

    }
    else
    {
	if (Final == -1)
	{
	    nimgs = H5.Ny;
	}
	else
	{
	    nimgs = Final; 
	}
    }

    ssc_processing_pipeline( &workspace, &processBlock, &data );

    ssc_disp_times( &workspace, timing );
   
    ssc_destroy_workspace(workspace);
  
    ssc_destroy_h5info(H5);
    
    return nimgs;
  }
}


