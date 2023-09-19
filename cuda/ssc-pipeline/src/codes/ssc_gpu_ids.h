#ifndef SSC_GPU_IDS_H
#define SSC_GPU_IDS_H

typedef struct
{
  int numGpus;

  int gpuIds[0];

}ssc_gpus;

ssc_gpus *createGpuIds(int numGpus, int* ids);
void destroyGpuIds(ssc_gpus *gpuIds);

#endif
