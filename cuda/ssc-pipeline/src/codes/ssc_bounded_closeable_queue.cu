#include "ssc_bounded_closeable_queue.h"

#define POPPED 0

#define SUCCESS 0
#define FAILURE -1

static void lock(bounded_closeable_queue_t* queue);
static void unlock(bounded_closeable_queue_t* queue);

static int maybePushToQueue(bounded_closeable_queue_t* queue, void* data);
static void releaseThreadsWaitingToPush(bounded_closeable_queue_t* queue);

void initializeBoundedCloseableQueue(bounded_closeable_queue_t* queue, int numSlots) {
  pthread_mutexattr_t* DEFAULT_ATTRIBUTES = NULL;
  const int INTERNAL_TO_PROCESS = 0;

  queue->numThreadsWaitingToPush = 0;

  sem_init(&queue->slots, INTERNAL_TO_PROCESS, numSlots);
  pthread_mutex_init(&queue->lock, DEFAULT_ATTRIBUTES);

  initializeCloseableQueue(&queue->unboundedQueue);
}

void releaseBoundedCloseableQueue(bounded_closeable_queue_t* queue) {
  boundedCloseableQueueClose(queue);
  boundedCloseableQueueWaitForClosure(queue);

  releaseCloseableQueue(&queue->unboundedQueue);

  pthread_mutex_destroy(&queue->lock);
  sem_destroy(&queue->slots);
}

static void lock(bounded_closeable_queue_t* queue) {
  pthread_mutex_lock(&queue->lock);
}

static void unlock(bounded_closeable_queue_t* queue) {
  pthread_mutex_unlock(&queue->lock);
}

int boundedCloseableQueuePush(bounded_closeable_queue_t* queue, void* data) {
  lock(queue);

  if (closeableQueueStatus(&queue->unboundedQueue) != OPEN) {
    unlock(queue);
    return FAILURE;
  }

  queue->numThreadsWaitingToPush += 1;
  unlock(queue);

  sem_wait(&queue->slots);

  return maybePushToQueue(queue, data);
}

static int maybePushToQueue(bounded_closeable_queue_t* queue, void* data) {
  int result = SUCCESS;

  lock(queue);

  if (closeableQueueStatus(&queue->unboundedQueue) == OPEN)
    closeableQueuePush(&queue->unboundedQueue, data);
  else
    result = FAILURE;

  queue->numThreadsWaitingToPush -= 1;

  unlock(queue);

  return result;
}

int boundedCloseableQueuePop(bounded_closeable_queue_t* queue,
			     void** elementPointer) {
  int popResult = closeableQueuePop(&queue->unboundedQueue, elementPointer);

  if (popResult == POPPED)
    sem_post(&queue->slots);

  return popResult;
}

void boundedCloseableQueueClose(bounded_closeable_queue_t* queue) {
  lock(queue);

  closeableQueueClose(&queue->unboundedQueue);

  unlock(queue);
}

void boundedCloseableQueueWaitForClosure(bounded_closeable_queue_t* queue) {
  closeableQueueWaitForClosure(&queue->unboundedQueue);
  releaseThreadsWaitingToPush(queue);
}

static void releaseThreadsWaitingToPush(bounded_closeable_queue_t* queue) {
  lock(queue);

  int remainingThreads = queue->numThreadsWaitingToPush;

  for (int index = 0; index < remainingThreads; ++index)
    sem_post(&queue->slots);

  unlock(queue);
}
