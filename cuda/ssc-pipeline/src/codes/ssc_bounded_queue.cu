#include "ssc_bounded_queue.h"

void initializeBoundedQueue(bounded_queue_t* queue, int numSlots) {
    const int INTERNAL_TO_PROCESS = 0;

    sem_init(&queue->slots, INTERNAL_TO_PROCESS, numSlots);

    initializeQueue(&queue->unboundedQueue);
}

void releaseBoundedQueue(bounded_queue_t* queue) {
    releaseQueue(&queue->unboundedQueue);

    sem_destroy(&queue->slots);
}

void boundedQueuePush(bounded_queue_t* queue, void* data) {
    sem_wait(&queue->slots);

    queuePush(&queue->unboundedQueue, data);
}

void* boundedQueuePop(bounded_queue_t* queue) {
    void* data = queuePop(&queue->unboundedQueue);

    sem_post(&queue->slots);

    return data;
}

int boundedQueueTryPop(bounded_queue_t* queue, void** elementPointer) {
    const int POPPED = 0;

    int popResult = queueTryPop(&queue->unboundedQueue, elementPointer);

    if (popResult == POPPED)
        sem_post(&queue->slots);

    return popResult;
}
