#ifndef PIPELINE_H
#define PIPELINE_H

#include <pthread.h>

#include "ssc_pipeline_stage.h"
#include "ssc_timing.h"

typedef void* (*pipeline_stage_function_t)(void*);

typedef struct {
    int numStages;
    pipeline_stage_t stages[0];
} pipeline_t;

typedef struct {
    pthread_t thread;

    timing_t executionTime;
    timing_t inputWaitTime;
    timing_t outputWaitTime;

    const char* name;
} pipeline_stage_thread_t;

typedef struct {
    pipeline_stage_t* stage;
    pipeline_stage_thread_t* thread;
} pipeline_thread_data_t;

pipeline_t* createPipeline(int numStages, pipeline_stage_params_t* stages);
void destroyPipeline(pipeline_t* pipeline);

void runPipeline(pipeline_t* pipeline, queue_t* workItems);
void runScalarPipeline(pipeline_t* pipeline, queue_t* workItems);
void runPipelineSequentially(pipeline_t* pipeline, queue_t* workItems);

#endif
