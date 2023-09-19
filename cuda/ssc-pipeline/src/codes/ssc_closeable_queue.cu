#include "ssc_closeable_queue.h"

#include <errno.h>

#define SUCCESS 0
#define FAILURE -1

static void lock(closeable_queue_t* queue);
static void unlock(closeable_queue_t* queue);

static void waitForQueueElement(closeable_queue_t* queue);
static int popFromLockedQueue(closeable_queue_t* queue, void** elementPointer);

static void maybeFinishClosingQueue(closeable_queue_t* queue);
static void finishClosingQueue(closeable_queue_t* queue);
static void releaseThreadsWaitingToPop(closeable_queue_t* queue);
static void notifyQueueClosure(closeable_queue_t* queue);

void initializeCloseableQueue(closeable_queue_t* queue) {
    pthread_mutexattr_t* DEFAULT_ATTRIBUTES = NULL;
    const int INTERNAL_TO_PROCESS = 0;

    queue->numThreadsWaitingForClosure = 0;
    queue->numThreadsWaitingToPop = 0;
    queue->status = OPEN;

    sem_init(&queue->size, INTERNAL_TO_PROCESS, 0);
    sem_init(&queue->closeComplete, INTERNAL_TO_PROCESS, 0);
    pthread_mutex_init(&queue->lock, DEFAULT_ATTRIBUTES);
    initializeAsyncQueue(&queue->queue);
}

void releaseCloseableQueue(closeable_queue_t* queue) {
    closeableQueueClose(queue);
    closeableQueueWaitForClosure(queue);

    releaseAsyncQueue(&queue->queue);
    pthread_mutex_destroy(&queue->lock);
    sem_destroy(&queue->closeComplete);
    sem_destroy(&queue->size);
}

static void lock(closeable_queue_t* queue) {
    pthread_mutex_lock(&queue->lock);
}

static void unlock(closeable_queue_t* queue) {
    pthread_mutex_unlock(&queue->lock);
}

int closeableQueuePush(closeable_queue_t* queue, void* data) {
    int result = FAILURE;

    lock(queue);

    if (queue->status == OPEN) {
        asyncQueuePush(&queue->queue, data);
        sem_post(&queue->size);
        result = SUCCESS;
    }

    unlock(queue);

    return result;
}

int closeableQueuePop(closeable_queue_t* queue, void** elementPointer) {
    int result = FAILURE;

    lock(queue);

    if (queue->status != CLOSED) {
        waitForQueueElement(queue);
        result = popFromLockedQueue(queue, elementPointer);
    }

    unlock(queue);

    return result;
}

static void waitForQueueElement(closeable_queue_t* queue) {
    queue->numThreadsWaitingToPop += 1;
    unlock(queue);

    sem_wait(&queue->size);

    lock(queue);
    queue->numThreadsWaitingToPop -= 1;
}

static int popFromLockedQueue(closeable_queue_t* queue, void** elementPointer) {
    if (queue->status == CLOSED)
        return FAILURE;

    asyncQueuePop(&queue->queue, elementPointer);

    if (queue->status == CLOSING)
        maybeFinishClosingQueue(queue);

    return SUCCESS;
}

static void maybeFinishClosingQueue(closeable_queue_t* queue) {
    if (sem_trywait(&queue->size) == 0)
        sem_post(&queue->size);
    else if (errno == EAGAIN)
        finishClosingQueue(queue);
}

static void finishClosingQueue(closeable_queue_t* queue) {
    queue->status = CLOSED;

    releaseThreadsWaitingToPop(queue);
    notifyQueueClosure(queue);
}

static void releaseThreadsWaitingToPop(closeable_queue_t* queue) {
    for (int index = 0; index < queue->numThreadsWaitingToPop; ++index)
        sem_post(&queue->size);
}

static void notifyQueueClosure(closeable_queue_t* queue) {
    for (int index = 0; index < queue->numThreadsWaitingForClosure; ++index)
        sem_post(&queue->closeComplete);
}

closeable_queue_status_t closeableQueueStatus(closeable_queue_t* queue) {
    closeable_queue_status_t status;

    lock(queue);
    status = queue->status;
    unlock(queue);

    return status;
}

void closeableQueueClose(closeable_queue_t* queue) {
    lock(queue);

    if (queue->status == OPEN)
        queue->status = CLOSING;

    maybeFinishClosingQueue(queue);

    unlock(queue);
}

void closeableQueueWaitForClosure(closeable_queue_t* queue) {
    lock(queue);

    if (queue->status == CLOSING) {
        queue->numThreadsWaitingForClosure += 1;
        unlock(queue);

        sem_wait(&queue->closeComplete);
    } else
        unlock(queue);
}
