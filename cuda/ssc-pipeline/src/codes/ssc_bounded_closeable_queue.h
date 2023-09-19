#ifndef SSC_BOUNDED_CLOSEABLE_QUEUE_H
#define SSC_BOUNDED_CLOSEABLE_QUEUE_H

#include "ssc_closeable_queue.h"

typedef struct {
    closeable_queue_t unboundedQueue;
    pthread_mutex_t lock;
    sem_t slots;

    unsigned int numThreadsWaitingToPush;
} bounded_closeable_queue_t;

void initializeBoundedCloseableQueue(bounded_closeable_queue_t* queue,
        int numSlots);
void releaseBoundedCloseableQueue(bounded_closeable_queue_t* queue);

int boundedCloseableQueuePush(bounded_closeable_queue_t* queue, void* data);
int boundedCloseableQueuePop(bounded_closeable_queue_t* queue,
        void** elementPointer);

void boundedCloseableQueueClose(bounded_closeable_queue_t* queue);
void boundedCloseableQueueWaitForClosure(bounded_closeable_queue_t* queue);

#endif
