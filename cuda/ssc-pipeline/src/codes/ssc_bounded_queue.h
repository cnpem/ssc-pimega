#ifndef BOUNDED_QUEUE_H
#define BOUNDED_QUEUE_H

#include "ssc_queue.h"

typedef struct {
    queue_t unboundedQueue;
    sem_t slots;
} bounded_queue_t;

void initializeBoundedQueue(bounded_queue_t* queue, int numSlots);
void releaseBoundedQueue(bounded_queue_t* queue);
void boundedQueuePush(bounded_queue_t* queue, void* data);
void* boundedQueuePop(bounded_queue_t* queue);
int boundedQueueTryPop(bounded_queue_t* queue, void** elementPointer);

#endif
