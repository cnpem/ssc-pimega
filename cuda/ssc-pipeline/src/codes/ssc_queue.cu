#include "ssc_queue.h"


void initializeQueue(queue_t* queue) {
    const int INTERNAL_TO_PROCESS = 0;

    sem_init(&queue->size, INTERNAL_TO_PROCESS, 0);

    initializeAsyncQueue(&queue->queue);
}

void releaseQueue(queue_t* queue) {
    sem_destroy(&queue->size);

    releaseAsyncQueue(&queue->queue);
}

void queuePush(queue_t* queue, void* data) {
    asyncQueuePush(&queue->queue, data);

    sem_post(&queue->size);
}

void* queuePop(queue_t* queue) {
    void* element;

    sem_wait(&queue->size);

    asyncQueuePop(&queue->queue, &element);

    return element;
}

int queueTryPop(queue_t* queue, void** elementPointer) {
    const int FAILURE = -1;
    const int SUCCESS = 0;

    int waitResult = sem_trywait(&queue->size);
    if (waitResult != 0)
        return FAILURE;

    asyncQueuePop(&queue->queue, elementPointer);

    return SUCCESS;
}
