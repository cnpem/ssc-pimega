#ifndef SSC_PIPELINE_PROCESSINGFUNCTIONS_H
#define SSC_PIPELINE_PROCESSINGFUNCTIONS_H

#include "../codes/ssc_types.h"
#include "ssc_pipeline.h"

#define CPUTHREADS 0

const int numStreamsPerGpu = 1;

typedef struct 
{
  int firstBlock;
  int currentBlock;
  int blockSize;
  ssc_volume* workspace;

}work_item_creation_parameters_t;


void configureOutputPath(const struct gengetopt_args_info* arg, char* outputPath);

void copyPathString(char* destination, const char* sourcePath);

void createGpuWorkspaces(ssc_volume** gpuWorkspaces, ssc_volume* globalWorkspace);

void destroyGpuWorkspaces(ssc_volume** gpuWorkspaces);

void createWorkItems(queue_t* workItems, int firstBlock, int blockSize, ssc_volume** workspace, int numBlocks);

work_item_creation_parameters_t* createWorkItemParameters(int firstBlock, int currentBlock, int blockSize, ssc_volume* workspace);

void* createWorkItem(void* parametersData);

ssc_volume* createGpuThreadWorkItem(int currentBlock, int blockInit,int blockSize, ssc_volume* globalWorkspace);

ssc_volume* cloneWorkspace(ssc_volume* sourceWorkspace);

void* readTomoBlock(void* wrappedWorkspace);

void* saveBlockAtGpu(void* wrappedWorkspace);

void* destroyGpuThreadWorkItem(void* wrappedWorkspace);

void *processBlockInGpu(void* wrappedWorkspace);

void ssc_processing_pipeline(ssc_volume* workspace, ssc_function executor, void *data);

ssc_gpus *ssc_configure_gpus(int npugs, int *gpus);

int *ssc_configure_oshape(int *shape);

void ssc_disp_times(ssc_volume *workspace, bool print);

#endif // SSC_PIPELINE_PROCESSINGFUNCTIONS_H


