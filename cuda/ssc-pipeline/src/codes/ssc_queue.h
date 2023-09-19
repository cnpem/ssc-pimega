#ifndef SSC_QUEUE_H
#define SSC_QUEUE_H

#include <semaphore.h>

#include "ssc_async_queue.h"

typedef struct {
    async_queue_t queue;

    sem_t size;
} queue_t;

void initializeQueue(queue_t* queue);
void releaseQueue(queue_t* queue);
void queuePush(queue_t* queue, void* data);
void* queuePop(queue_t* queue);
int queueTryPop(queue_t* queue, void** elementPointer);

#endif
