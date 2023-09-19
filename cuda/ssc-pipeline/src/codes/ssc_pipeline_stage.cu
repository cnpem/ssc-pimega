#include <stdio.h>

#include "ssc_pipeline_stage.h"

void initializePipelineStage(pipeline_stage_t* pipelineStage,
        pipeline_stage_params_t* parameters) {
    pipelineStage->name = parameters->name;
    pipelineStage->function = parameters->function;
    pipelineStage->numThreads = parameters->numThreads;

    initializeQueue(&pipelineStage->activeThreads);
    initializeQueue(&pipelineStage->finishedThreads);

    initializeBoundedCloseableQueue(&pipelineStage->workQueue,
            parameters->numSlots);

    pipelineStage->nextStage = NULL;
}

void releasePipelineStage(pipeline_stage_t* pipelineStage) {
    releaseBoundedCloseableQueue(&pipelineStage->workQueue);
    releaseQueue(&pipelineStage->activeThreads);
    releaseQueue(&pipelineStage->finishedThreads);
}

void connectPipelineStages(pipeline_stage_t* before, pipeline_stage_t* after) {
    before->nextStage = after;
}

void disconnectPipelineStages(pipeline_stage_t* before,
        pipeline_stage_t* after) {
    before->nextStage = NULL;
}
