#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ssc_workspace.h"
#include "ssc_types.h"

/*--------------------------------------------------------
  Destroy workspace for processing
  --------------------------------------------------------*/

__host__ void ssc_destroy_workspace(ssc_volume workspace)
{
  free(workspace.volume);
  free(workspace.output);
  
  free(workspace.elapsed.proc);
  free(workspace.elapsed.read);
  free(workspace.elapsed.save);
  
  destroyGpuIds(workspace.gpus);
  
  cudaDeviceReset();
}

/*--------------------------------------------------------
  Create workspace for processing 
  --------------------------------------------------------*/

//   Find number of blocks, given a blockSize. This function 
//   calculates how many blocks are needed to process all data,
//   based on number of slices and block size.
 
__host__ int ssc_numberOfBlocks(int qtdSlices,
				int blockSize)
{
  int nblocks;
  int rest;
  
  rest = (qtdSlices%blockSize);
  
  if(qtdSlices>blockSize && rest!=0)
  {
    nblocks = (int) floor(qtdSlices/blockSize);
    nblocks = nblocks + 1;
  }
  else
    if(qtdSlices>blockSize && rest==0)
      nblocks = (int) floor(qtdSlices/blockSize);
  else
    nblocks = 1;
  
  return nblocks;
}


__host__ void ssc_create_workspace(ssc_volume * workspace,
				   ssc_h5 *H5,
				   ssc_gpus *gpus,
				   int *oshape,
				   int blockSize,
				   char *outputPath,
			   	   char *uuid,
				   bool printTiming,
				   bool save,
				   int Init,
				   int Final)
{
  workspace->gpus      = gpus;
  workspace->gpuId     = gpus->gpuIds[0];
  workspace->Init      = Init;
  workspace->Final     = Final;
  workspace->blockSize = blockSize;
  workspace->oshape    = oshape;

  
  if ( H5->order == SSC_XY )
    {
      workspace->volume = (float *)malloc(blockSize * H5->Nx * H5->Ny * sizeof(float));

      workspace->oNx  = oshape[1];
      workspace->oNy  = oshape[0];
      
      workspace->output = (float *)malloc(blockSize * oshape[0] * oshape[1] * sizeof(float));

      workspace->outputwc = (float *)malloc(blockSize * H5->Nx * H5->Ny * sizeof(float));

      if ( Final == -1)
      {  // read everything!
	 workspace->Final = H5->Nz;
      }
    }
      
  if ( H5->order == SSC_XZ )
    {
      workspace->volume = (float *)malloc(blockSize * H5->Nx * H5->Nz * sizeof(float));

      workspace->oNx  = oshape[1];
      workspace->oNz  = oshape[0];
      
      workspace->output = (float *)malloc(blockSize * oshape[0] * oshape[1] * sizeof(float));
 
      workspace->outputwc = (float *)malloc(blockSize * H5->Nx * H5->Ny * sizeof(float));
      
      if ( Final == -1)
      {  // read everything!
	 workspace->Final = H5->Ny;
      }
   }
  
  int noSlices       = workspace->Final - workspace->Init;
  int numberofblocks = ssc_numberOfBlocks(noSlices, workspace->blockSize);

  workspace->nblocks   = numberofblocks;
    
  workspace->elapsed.proc = (float *)malloc( numberofblocks * sizeof(float));
  workspace->elapsed.read = (float *)malloc( numberofblocks * sizeof(float));
  workspace->elapsed.save = (float *)malloc( numberofblocks * sizeof(float));
  
  workspace->uuid        = uuid;
  workspace->outputPath  = outputPath;
  workspace->printTiming = printTiming;
  workspace->save        = save;
  workspace->H5          = H5;
  
  workspace->elapsed.processing = 0;
  workspace->elapsed.reading    = 0;
  workspace->elapsed.saving     = 0;
  workspace->elapsed.total      = 0;
}
