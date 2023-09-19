#include "ssc_async_queue.h"

static void lock(async_queue_t* queue);
static void unlock(async_queue_t* queue);

static queue_node_t* createQueueNode(void* data);
static void destroyQueueNode(queue_node_t* node);
static void* removeQueueFront(async_queue_t* queue);
static void appendToQueue(async_queue_t* queue, queue_node_t* node);

void initializeAsyncQueue(async_queue_t* queue) {
  pthread_mutexattr_t* DEFAULT_ATTRIBUTES = NULL;

  pthread_mutex_init(&queue->lock, DEFAULT_ATTRIBUTES);

  queue->front = NULL;
  queue->back = NULL;
}

void releaseAsyncQueue(async_queue_t* queue) {
  while (queue->front != NULL) {
    queue_node_t* removedNode = queue->front;
    queue->front = removedNode->next;
    destroyQueueNode(removedNode);
  }

  pthread_mutex_destroy(&queue->lock);
}

static void lock(async_queue_t* queue) {
  pthread_mutex_lock(&queue->lock);
}

static void unlock(async_queue_t* queue) {
  pthread_mutex_unlock(&queue->lock);
}

void asyncQueuePush(async_queue_t* queue, void* data) {
  queue_node_t* node = createQueueNode(data);

  lock(queue);
  appendToQueue(queue, node);
  unlock(queue);
}

static void appendToQueue(async_queue_t* queue, queue_node_t* node) {
  if (queue->front == NULL) {
    queue->front = node;
    queue->back = node;
  } else {
    queue->back->next = node;
    queue->back = node;
  }
}

int asyncQueuePop(async_queue_t* queue, void** elementPointer) {
  const int FAILURE = -1;
  const int SUCCESS = 0;

  int result = SUCCESS;

  lock(queue);

  if (queue->front != NULL)
    *elementPointer = removeQueueFront(queue);
  else
    result = FAILURE;

  unlock(queue);

  return result;
}

static void* removeQueueFront(async_queue_t* queue) {
  queue_node_t* node = queue->front;
  void* element = node->element;

  queue->front = node->next;

  destroyQueueNode(node);

  return element;
}

static queue_node_t* createQueueNode(void* data) {
  queue_node_t* node = (queue_node_t*)malloc(sizeof(queue_node_t));

  node->element = data;
  node->next = NULL;

  return node;
}

static void destroyQueueNode(queue_node_t* node) {
  free(node);
}
