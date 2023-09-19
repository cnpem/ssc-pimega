#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <pthread.h>

#include <sys/shm.h>
#include <sys/ipc.h> 
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>

extern "C" {
#include "../codes/cmdline.h"
#include <time.h>
}

#include "ssc_pipeline.h"
#include "ssc_pipeline_processingFunctions.h"

#include "../codes/ssc_h5.h"
#include "../codes/ssc_workspace.h"
#include "../codes/ssc_types.h"
#include "../codes/ssc_save.h"

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/

void configureOutputPath(const struct gengetopt_args_info* arg, char* outputPath) 
{
  if (arg->outpath_given)
    copyPathString(outputPath, arg->outpath_arg);
  else {
    copyPathString(outputPath, arg->path_arg);
    strcat(outputPath, "/recon");
  }
}

ssc_gpus *ssc_configure_gpus(int ngpus, int *gpus)
{
  int ids[] = { 0 };
  
  if (ngpus > 0)
    return createGpuIds(ngpus, gpus);
  else
    return createGpuIds(1, ids);
}

int *ssc_configure_oshape(int *shape) 
{
  int arraySize = sizeof(int) * 2;
  int size = sizeof(int) + arraySize;
  int* oshape = (int*)calloc(size, 1);
  
  if (shape != NULL)
    memcpy(oshape, shape, arraySize);
  
  return oshape;
}


void copyPathString(char* destination, 
		    const char* sourcePath) 
{
  int length = strlen(sourcePath);
  int pathEnd = length - 1;
  char lastChar = sourcePath[pathEnd];
  
  while (lastChar == '/') {
    --pathEnd;
    
    lastChar = sourcePath[pathEnd];
  }
  
  int pathLength = pathEnd + 1;

  strncpy(destination, sourcePath, pathLength);
  destination[pathLength] = '\0';
}

void  ssc_processing_pipeline(ssc_volume* workspace,
			      ssc_function executor,
			      void *userData) 
{
  int blockSize      = workspace->blockSize;
  int numberofblocks = workspace->nblocks;
  int blockInit      = (int) floor(workspace->Init/blockSize);
  int blockFinal     = (int) floor(workspace->Final/blockSize);

  workspace->functionPointer = executor;
  workspace->userData        = userData;
  
  struct timespec TimeStart, TimeEnd;
  clock_gettime(CLOCK, &TimeStart);
  
  int numGpus = workspace->gpus->numGpus;
  int numCpuThreads = 1;
  int numGpuThreads;
  
  if(numGpus==1)
    numGpuThreads = numGpus * numStreamsPerGpu;
  else
    numGpuThreads = 2 * numGpus * numStreamsPerGpu;
  
  pipeline_stage_params_t stages[] = {
    { "Create work item",  workspace->PTS[0][0], workspace->PTS[0][1],
      createWorkItem },
    { "Read volume block", workspace->PTS[1][0], workspace->PTS[1][1],
      readTomoBlock },
    { "ProcessBlock (GPU)", workspace->PTS[2][0], numGpuThreads,
      processBlockInGpu },
    { "Save result block",  workspace->PTS[3][0], workspace->PTS[3][1],
      saveBlockAtGpu },
    { "Destroy work item", workspace->PTS[4][0], workspace->PTS[4][0],
      destroyGpuThreadWorkItem },
  };

  ssc_volume* workspaces[numGpus];
  queue_t workItems;
  
  int numStages = sizeof(stages) / sizeof(stages[0]);
  pipeline_t* pipeline = createPipeline(numStages, stages);

  createGpuWorkspaces(workspaces, workspace);
  createWorkItems(&workItems, blockInit, blockSize, workspaces, numberofblocks);
  
  runPipeline(pipeline, &workItems);
  
  destroyPipeline(pipeline);

  releaseQueue(&workItems);

  //destroyGpuWorkspaces(workspaces);

  clock_gettime(CLOCK, &TimeEnd);
  float elapsed = TIME(TimeEnd,TimeStart);

  workspace->elapsed.total = elapsed;
}


void createGpuWorkspaces(ssc_volume **gpuWorkspaces,
			 ssc_volume *globalWorkspace) 
{
  ssc_gpus* gpus = globalWorkspace->gpus;
  int numGpus = gpus->numGpus;
  int* gpuIds = gpus->gpuIds;
  
  for (int gpuIndex = 0; gpuIndex < numGpus; ++gpuIndex)
    {
      ssc_volume *gpuWorkspace = cloneWorkspace(globalWorkspace);
      
      int gpuId = gpuIds[gpuIndex];
      
      gpuWorkspace->gpuId = gpuId;
      
      cudaSetDevice( gpuId );
      
      gpuWorkspaces[gpuIndex] = gpuWorkspace;
  }
}

void destroyGpuWorkspaces(ssc_volume** gpuWorkspaces) 
{
  int numGpus = gpuWorkspaces[0]->gpus->numGpus;

  // empty function! Include here texture and other CUDA stuff!
}


void createWorkItems(queue_t* workItems, 
		     int firstBlock, 
		     int blockSize,
		     ssc_volume** workspaces, 
		     int numBlocks) 
{
  int numGpus = workspaces[0]->gpus->numGpus;

  initializeQueue(workItems);

  for (int currentBlock = 0; currentBlock < numBlocks; ++currentBlock) {
    int gpuIndex = currentBlock % numGpus;

    void* parameters = createWorkItemParameters(firstBlock, currentBlock,
						blockSize, workspaces[gpuIndex]);
    
    queuePush(workItems, parameters);
  }
}

work_item_creation_parameters_t* createWorkItemParameters(int firstBlock,
							  int currentBlock, 
							  int blockSize, 
							  ssc_volume* workspace) {
  work_item_creation_parameters_t* workItemParameters;
  
  workItemParameters = (work_item_creation_parameters_t*)
    malloc(sizeof(*workItemParameters));
  
  workItemParameters->firstBlock = firstBlock;
  workItemParameters->currentBlock = currentBlock;
  workItemParameters->blockSize = blockSize;
  workItemParameters->workspace = workspace;
  
  return workItemParameters;
}

void* createWorkItem(void* parametersData)
{  
  work_item_creation_parameters_t* parameters =
    (work_item_creation_parameters_t*)parametersData;
  
  ssc_volume* workItem = createGpuThreadWorkItem(parameters->currentBlock,
						 parameters->firstBlock,
						 parameters->blockSize,
						 parameters->workspace);
  
  free(parameters);
  
  return workItem;
}

void* processBlockInGpu(void* wrappedWorkspace) 
{
  ssc_volume* workspace = (ssc_volume*)wrappedWorkspace;
  float elapsed;

  int dev = workspace->gpuId;
  
  cudaSetDevice(dev);

  struct timespec TimeStart, TimeEnd;
  clock_gettime(CLOCK, &TimeStart);

  (*workspace->functionPointer)(workspace->output, workspace->outputwc, workspace->volume,  workspace->userData );
  
  clock_gettime(CLOCK, &TimeEnd);
  elapsed = TIME(TimeEnd,TimeStart);

  if(workspace->printTiming)	  
    fprintf(stderr,"\n\t%lf\tBlock[%d]: Processing hyperslab (GPU[%d]) ",elapsed,workspace->nblock,dev);

  //>>>>>>>>>>>>>>>>> some GPU usage <<<<<<<<<<<<<<<<<<<<<<<
  /*
  int devicesCount;
  cudaError_t CuStatus;
  CuStatus = cudaGetDeviceCount(&devicesCount);
  if(CuStatus == cudaSuccess)
    {
      if(workspace->printTiming)	
	{
	  int dev = workspace->gpuId;

	  struct timespec TimeStart, TimeEnd;
	  clock_gettime(CLOCK, &TimeStart);

	  //something at GPU no. dev:
	  cudaDeviceProp prop;
	  cudaGetDeviceProperties(&prop,  dev);
	  //
	  
	  clock_gettime(CLOCK, &TimeEnd);
	  elapsed = TIME(TimeEnd,TimeStart); 

	  fprintf(stderr,"\n\t%lf\tBlock[%d]: Processing hyperslab (GPU[%d]) ",elapsed,workspace->nblock,dev);
	  
	}
    }
  */
  //>>>>>>>>>>>>>>>>> ------------- <<<<<<<<<<<<<<<<<<<<<<<

  workspace->elapsed.proc[ workspace->nblock ] = elapsed;
 
  return workspace;
}



ssc_volume* createGpuThreadWorkItem(int currentBlock, 
				   int blockInit,
				   int blockSize, 
				   ssc_volume* globalWorkspace) 
{
  int Nx = globalWorkspace->H5->Nx;
  int Ny = globalWorkspace->H5->Ny;
  int Nz = globalWorkspace->H5->Nz;
  
  int oNx = globalWorkspace->oNx;
  int oNy = globalWorkspace->oNy;
  int oNz = globalWorkspace->oNz;
  
  ssc_volume* workspace = cloneWorkspace(globalWorkspace);
  
  workspace->nblock = currentBlock;
 
  //
  //starting image: si (within the block)
  //ending image: ei (within the block)
  //
  workspace->si = (blockSize*currentBlock)+(blockInit*blockSize);
  workspace->ei = (blockSize*(currentBlock+1))+(blockInit*blockSize);

  if ( workspace->H5->order == SSC_XY )
    {
      workspace->volume   = (float*)malloc(blockSize * Nx * Ny * sizeof(float));
      workspace->outputwc = (float*)malloc(blockSize * Nx * Ny * sizeof(float));
      workspace->output   = (float*)malloc(blockSize * oNx * oNy * sizeof(float));
    }
  
  if ( workspace->H5->order == SSC_XZ )
    {
      workspace->volume   = (float*)malloc(blockSize * Nx * Nz * sizeof(float));
      workspace->outputwc = (float*)malloc(blockSize * Nx * Nz * sizeof(float));
      workspace->output   = (float*)malloc(blockSize * oNx * oNz * sizeof(float));
    }
  
  return workspace;
}

ssc_volume* cloneWorkspace(ssc_volume* sourceWorkspace) 
{
  ssc_volume* workspace = (ssc_volume*)malloc(sizeof(*workspace));
  
  memcpy(workspace, sourceWorkspace, sizeof(*workspace));

  return workspace;
}

void* readTomoBlock(void* wrappedWorkspace) 
{
  
  ssc_volume* workspace = (ssc_volume*)wrappedWorkspace;
  
  struct timespec TimeStart, TimeEnd;
  clock_gettime(CLOCK, &TimeStart);
  
  if ( workspace->H5->order == SSC_XY )
    {
      ssc_read_block_xy(workspace);
    }
  
  if ( workspace->H5->order == SSC_XZ )
    {
      ssc_read_block_xz(workspace);
    }
  
  clock_gettime(CLOCK, &TimeEnd);
  float elapsed = TIME(TimeEnd,TimeStart);
  
  if(workspace->printTiming)	
    fprintf(stderr,"\n\t%lf\tBlock[%d]: Read H5 hyperslab (CPU)",elapsed,workspace->nblock);  

  workspace->elapsed.read[ workspace->nblock ] = elapsed;

  return workspace;
}


void* saveBlockAtGpu(void* wrappedWorkspace) 
{
  ssc_volume* workspace = (ssc_volume*)wrappedWorkspace;

  struct timespec TimeStart, TimeEnd;
  clock_gettime(CLOCK, &TimeStart);

  ssc_save_block(workspace);
    
  clock_gettime(CLOCK, &TimeEnd);
  float elapsed = TIME(TimeEnd,TimeStart);
  if(workspace->printTiming)	
    fprintf(stderr,"\n\t%lf\tBlock[%d]: Save processed hyperslab (CPU)",elapsed,workspace->nblock);  

  workspace->elapsed.save[ workspace->nblock ] = elapsed;
  
  return workspace;
}

void* destroyGpuThreadWorkItem(void* wrappedWorkspace) 
{
  ssc_volume* workspace = (ssc_volume *)wrappedWorkspace;
  
  free(workspace->volume);    
  free(workspace->output);
  free(workspace->outputwc);
  free(workspace);
    
  return NULL;
}


void ssc_disp_times(ssc_volume *workspace, bool print)
{
  float proc, read, save;

  proc = 0;
  read = 0;
  save = 0;

  for( int k = 0 ; k < workspace->nblocks ; k ++)
    {
      proc += workspace->elapsed.proc[k];
      read += workspace->elapsed.read[k];
      save += workspace->elapsed.save[k];
    }

  workspace->elapsed.processing = proc;
  workspace->elapsed.reading    = read;
  workspace->elapsed.saving     = save;

  if (print)
    {
      fprintf(stderr,"\n");
      fprintf(stderr,"\n SSC Pipeline - Total Elapsed time: %lf", workspace->elapsed.total );  
      fprintf(stderr,"\n              - Reading hyperslabs: %lf", workspace->elapsed.reading );
      fprintf(stderr,"\n              - Procesing hyperslabs: %lf", workspace->elapsed.processing );
      fprintf(stderr,"\n              - Saving processed hyperslabs: %lf\n", workspace->elapsed.saving );
    }
}
