#ifndef SSC_CLOSEABLE_QUEUE_H
#define SSC_CLOSEABLE_QUEUE_H

#include <semaphore.h>

#include "ssc_async_queue.h"

typedef enum {
    OPEN,
    CLOSING,
    CLOSED
} closeable_queue_status_t;

typedef struct {
    async_queue_t queue;

    sem_t size;
    sem_t closeComplete;
    pthread_mutex_t lock;

    unsigned int numThreadsWaitingForClosure;
    unsigned int numThreadsWaitingToPop;

    closeable_queue_status_t status;
} closeable_queue_t;

void initializeCloseableQueue(closeable_queue_t* queue);
void releaseCloseableQueue(closeable_queue_t* queue);

int closeableQueuePush(closeable_queue_t* queue, void* data);
int closeableQueuePop(closeable_queue_t* queue, void** elementPointer);

closeable_queue_status_t closeableQueueStatus(closeable_queue_t* queue);
void closeableQueueClose(closeable_queue_t* queue);
void closeableQueueWaitForClosure(closeable_queue_t* queue);

#endif
