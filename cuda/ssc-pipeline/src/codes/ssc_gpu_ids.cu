#include <stdlib.h>

#include "ssc_gpu_ids.h"

ssc_gpus* createGpuIds(int numGpus, int* ids) 
{
  int arraySize = sizeof(int) * numGpus;
  int size = sizeof(ssc_gpus) + arraySize;
  ssc_gpus* gpuIds = (ssc_gpus*)calloc(size, 1);

  gpuIds->numGpus = numGpus;

  if (numGpus > 0) {
    if (ids != NULL)
      memcpy(gpuIds->gpuIds, ids, arraySize);
  }

  return gpuIds;
}

void destroyGpuIds(ssc_gpus* gpuIds) 
{
  free(gpuIds);
}
