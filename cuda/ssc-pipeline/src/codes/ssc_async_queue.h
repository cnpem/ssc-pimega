#ifndef RAFT_ASYNC_QUEUE_H
#define RAFT_ASYNC_QUEUE_H

#include <pthread.h>

typedef struct queue_node {
  void* element;
  struct queue_node* next;
} queue_node_t;

typedef struct {
  pthread_mutex_t lock;

  queue_node_t* front;
  queue_node_t* back;
} async_queue_t;

void initializeAsyncQueue(async_queue_t* queue);
void releaseAsyncQueue(async_queue_t* queue);
void asyncQueuePush(async_queue_t* queue, void* data);
int asyncQueuePop(async_queue_t* queue, void** elementPointer);

#endif
