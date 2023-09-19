#include <stdio.h>

#include "ssc_pipeline.h"

#define POPPED 0
#define TRUE 1

static void runAllWorkItemsOnSequentialPipeline(pipeline_t* pipeline,
        queue_t* workItems);
static void createFakeStageThreads(pipeline_t* pipeline);
static void runWorkItemOnSequentialPipeline(pipeline_t* pipeline);
static void startScalarPipelineExecution(pipeline_t* pipeline);

static void fillPipeline(pipeline_stage_t* firstStage, queue_t* workItems);
static void startPipelineExecution(pipeline_t* pipeline);

static void createPipelineStageThread(pipeline_stage_t* stage, int threadId);
static pipeline_thread_data_t* createThreadData(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread);
static void* pipelineStageThreadEntry(void* wrappedData);

static int executeStage(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread);
static void* processWorkItem(pipeline_stage_thread_t* thread,
        pipeline_stage_function_t function, void* workItem);

static int waitForInput(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread, void** workItemPointer);
static void forwardResult(pipeline_stage_t* nextStage,
        pipeline_stage_thread_t* thread, void* workResult);

static void waitForPipelineCompletion(pipeline_t* pipeline);
static void printPipelineTimingStatistics(pipeline_t* pipeline);
static void printStageTimingStatistics(pipeline_stage_t* stage);
static void collectStageTimingStatistics(pipeline_stage_t* stage,
        timing_t* inputWaitTime, timing_t* outputWaitTime,
        timing_t* executionTime);
static void printTimingStatistics(const char* name, timing_t* timingStatistics);

pipeline_t* createPipeline(int numStages,
        pipeline_stage_params_t* stageParams) {
    int stagesArraySize = numStages * sizeof(pipeline_stage_t);
    int pipelineSize = sizeof(pipeline_t) + stagesArraySize;

    pipeline_t* pipeline = (pipeline_t*)malloc(pipelineSize);
    pipeline_stage_t* stages = pipeline->stages;

    pipeline->numStages = numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];
        pipeline_stage_params_t* params = &stageParams[stageIndex];

        initializePipelineStage(stage, params);
    }

    for (int stageIndex = 1; stageIndex < numStages; ++stageIndex) {
        pipeline_stage_t* previousStage = &stages[stageIndex - 1];
        pipeline_stage_t* nextStage = &stages[stageIndex];

        connectPipelineStages(previousStage, nextStage);
    }

    return pipeline;
}

void destroyPipeline(pipeline_t* pipeline) {
    int numStages = pipeline->numStages;
    pipeline_stage_t* stages = pipeline->stages;

    for (int index = 0; index < numStages; ++index)
        releasePipelineStage(&stages[index]);

    free(pipeline);
}

void runPipeline(pipeline_t* pipeline, queue_t* workItems) {
    pipeline_stage_t* firstStage = &pipeline->stages[0];

    startPipelineExecution(pipeline);
    fillPipeline(firstStage, workItems);
    waitForPipelineCompletion(pipeline);
    
    //printPipelineTimingStatistics(pipeline);
}

void runScalarPipeline(pipeline_t* pipeline, queue_t* workItems) {
    pipeline_stage_t* firstStage = &pipeline->stages[0];

    startScalarPipelineExecution(pipeline);
    fillPipeline(firstStage, workItems);
    waitForPipelineCompletion(pipeline);

    printPipelineTimingStatistics(pipeline);
}

void runPipelineSequentially(pipeline_t* pipeline, queue_t* workItems) {
    runAllWorkItemsOnSequentialPipeline(pipeline, workItems);

    printPipelineTimingStatistics(pipeline);
}

static void runAllWorkItemsOnSequentialPipeline(pipeline_t* pipeline,
        queue_t* workItems) {
    pipeline_stage_t* firstStage = &pipeline->stages[0];
    void* workItem;

    createFakeStageThreads(pipeline);

    while (queueTryPop(workItems, &workItem) == POPPED) {
        boundedCloseableQueuePush(&firstStage->workQueue, workItem);
        runWorkItemOnSequentialPipeline(pipeline);
    }
}

static void createFakeStageThreads(pipeline_t* pipeline) {
    pipeline_stage_thread_t* fakeThread;
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];

        fakeThread = (pipeline_stage_thread_t*)malloc(sizeof(*fakeThread));
        fakeThread->name = stage->name;

	//fprintf(stderr, "Creating fake thread for [%s] stage\n", stage->name);

        initializeTiming(&fakeThread->executionTime);
        initializeTiming(&fakeThread->inputWaitTime);
        initializeTiming(&fakeThread->outputWaitTime);

        queuePush(&stage->activeThreads, fakeThread);
    }
}

static void runWorkItemOnSequentialPipeline(pipeline_t* pipeline) {
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];
        pipeline_stage_thread_t* thread;

        thread = (pipeline_stage_thread_t*)queuePop(&stage->activeThreads);

        executeStage(stage, thread);

        queuePush(&stage->activeThreads, thread);
    }
}

static void fillPipeline(pipeline_stage_t* firstStage, queue_t* workItems) {
    void* workItem;

    while (queueTryPop(workItems, &workItem) == POPPED)
        boundedCloseableQueuePush(&firstStage->workQueue, workItem);

    boundedCloseableQueueClose(&firstStage->workQueue);
}

static void startPipelineExecution(pipeline_t* pipeline) {
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++ stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];
        int numThreads = stage->numThreads;

        for (int threadIndex = 0; threadIndex < numThreads; ++threadIndex)
            createPipelineStageThread(stage, threadIndex);
    }
}

static void startScalarPipelineExecution(pipeline_t* pipeline) {
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++ stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];

        createPipelineStageThread(stage, 0);
    }
}

static void createPipelineStageThread(pipeline_stage_t* stage, int threadId) {
    pipeline_stage_thread_t* stageThread =
            (pipeline_stage_thread_t*)malloc(sizeof(pipeline_stage_thread_t));

    //fprintf(stderr, "Creating thread for [%s] stage\n", stage->name);

    int nameLength = strlen(stage->name) + 10;
    char* name = (char*)calloc(nameLength, 1);
    snprintf(name, nameLength, "%s (%d)", stage->name, threadId);

    stageThread->name = name;

    initializeTiming(&stageThread->executionTime);
    initializeTiming(&stageThread->inputWaitTime);
    initializeTiming(&stageThread->outputWaitTime);

    pipeline_thread_data_t* threadData = createThreadData(stage, stageThread);

    pthread_create(&stageThread->thread, NULL, pipelineStageThreadEntry,
            threadData);

    queuePush(&stage->activeThreads, stageThread);
}

static pipeline_thread_data_t* createThreadData(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread) {
    pipeline_thread_data_t* data;

    data = (pipeline_thread_data_t*)malloc(sizeof(pipeline_thread_data_t));
    data->stage = stage;
    data->thread = thread;

    return data;
}

static void* pipelineStageThreadEntry(void* wrappedData) {
    pipeline_thread_data_t* threadData = (pipeline_thread_data_t*)wrappedData;
    pipeline_stage_t* stage = threadData->stage;
    pipeline_stage_thread_t* thread = threadData->thread;
    //const char* threadName = thread->name;

    int shouldContinue = TRUE;

    //fprintf(stderr, "[%s] Started pipeline stage thread\n", threadName);

    while (shouldContinue)
        shouldContinue = executeStage(stage, thread);

    //fprintf(stderr, "[%s] Finished processing data\n", threadName);
    free(threadData);

    return NULL;
}

static int executeStage(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread) {
    const int CONTINUE = 1;
    const int STOP = 0;

    pipeline_stage_function_t function = stage->function;

    void* workItem;
    int inputStatus = waitForInput(stage, thread, &workItem);

    if (inputStatus != POPPED)
        return STOP;

    void* workResult = processWorkItem(thread, function, workItem);

    forwardResult(stage->nextStage, thread, workResult);

    return CONTINUE;
}

static void* processWorkItem(pipeline_stage_thread_t* thread,
        pipeline_stage_function_t function, void* workItem) {
    void* workResult;

    //fprintf(stderr, "[%s] Processing data\n", thread->name);
    startTimingMeasurement(&thread->executionTime);

    workResult = function(workItem);

    endTimingMeasurement(&thread->executionTime);

    return workResult;
}

static int waitForInput(pipeline_stage_t* stage,
        pipeline_stage_thread_t* thread, void** workItemPointer) {
    int inputStatus;

    startTimingMeasurement(&thread->inputWaitTime);

    //fprintf(stderr, "[%s] Waiting for data\n", thread->name);
    inputStatus = boundedCloseableQueuePop(&stage->workQueue, workItemPointer);

    endTimingMeasurement(&thread->inputWaitTime);

    return inputStatus;
}

static void forwardResult(pipeline_stage_t* nextStage,
        pipeline_stage_thread_t* thread, void* workResult) {
    if (nextStage != NULL) {
        startTimingMeasurement(&thread->outputWaitTime);

        //fprintf(stderr, "[%s] Waiting for output slot\n", thread->name);
        boundedCloseableQueuePush(&nextStage->workQueue, workResult);

        endTimingMeasurement(&thread->outputWaitTime);
    }
}

static void waitForPipelineCompletion(pipeline_t* pipeline) {
    pipeline_stage_thread_t* stageThread;
    void** stageThreadPointer = (void**)&stageThread;
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;

    for (int stageIndex = 0; stageIndex < numStages; ++ stageIndex) {
        pipeline_stage_t* stage = &stages[stageIndex];
        queue_t* activeThreads = &stage->activeThreads;

        boundedCloseableQueueClose(&stage->workQueue);

        while (queueTryPop(activeThreads, stageThreadPointer) == POPPED) {
            pthread_join(stageThread->thread, NULL);
            queuePush(&stage->finishedThreads, stageThread);
        }
    }
}

static void printPipelineTimingStatistics(pipeline_t* pipeline) {
    pipeline_stage_t* stages = pipeline->stages;
    int numStages = pipeline->numStages;
    
    printf("\n\n");
    printf("      STAGE\t\t\t\tMIN\tAVG\tMAX\tTOTAL\n");
    printf("------------------------------------------------------------------------------------------------\n");

    for (int index = 0; index < numStages; ++index)
        printStageTimingStatistics(&stages[index]);
}

static void printStageTimingStatistics(pipeline_stage_t* stage) {
    timing_t inputWaitTime;
    timing_t outputWaitTime;
    timing_t executionTime;

    collectStageTimingStatistics(stage, &inputWaitTime, &outputWaitTime,
            &executionTime);

    printTimingStatistics("Input wait time", &inputWaitTime);
    printTimingStatistics("Output wait time", &outputWaitTime);
    printTimingStatistics(stage->name, &executionTime);

    printf("------------------------------------------------------------------------------------------------\n");
}

static void collectStageTimingStatistics(pipeline_stage_t* stage,
        timing_t* inputWaitTime, timing_t* outputWaitTime,
        timing_t* executionTime) {
    pipeline_stage_thread_t* thread;
    void** threadPointer = (void**)&thread;

    initializeTiming(inputWaitTime);
    initializeTiming(outputWaitTime);
    initializeTiming(executionTime);

    while (queueTryPop(&stage->finishedThreads, threadPointer) == POPPED) 
      {
	mergeTiming(inputWaitTime, &thread->inputWaitTime);
	mergeTiming(outputWaitTime, &thread->outputWaitTime);
	mergeTiming(executionTime, &thread->executionTime);
	free(thread);
      }
    
    calculateAverageTime(inputWaitTime);
    calculateAverageTime(outputWaitTime);
    calculateAverageTime(executionTime);
}

static void printTimingStatistics(const char* name,
        timing_t* timingStatistics) {
    if (timingStatistics->total == 0.0)
        return;

    calculateAverageTime(timingStatistics);

    printf(" %-17s\t\t\t%-8.3f\t%-8.3f\t%-8.3f\t%-8.3f\n", name,
            timingStatistics->minimum,
            timingStatistics->average,
            timingStatistics->maximum,
            timingStatistics->total);
}
