#ifndef PIPELINE_STAGE_H
#define PIPELINE_STAGE_H

#include "ssc_bounded_closeable_queue.h"
#include "ssc_queue.h"

typedef void* (*pipeline_stage_function_t)(void*);

typedef struct {
    const char* name;
    int numSlots;
    int numThreads;

    pipeline_stage_function_t function;
} pipeline_stage_params_t;

typedef struct pipeline_stage {
    const char* name;
    pipeline_stage_function_t function;

    int numThreads;
    queue_t activeThreads;
    queue_t finishedThreads;

    bounded_closeable_queue_t workQueue;

    struct pipeline_stage* nextStage;
} pipeline_stage_t;

void initializePipelineStage(pipeline_stage_t* pipelineStage,
        pipeline_stage_params_t* parameters);
void releasePipelineStage(pipeline_stage_t* pipelineStage);

void connectPipelineStages(pipeline_stage_t* before, pipeline_stage_t* after);
void disconnectPipelineStages(pipeline_stage_t* before,
        pipeline_stage_t* after);

#endif
