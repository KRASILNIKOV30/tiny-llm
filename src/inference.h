#include <pthread.h>
#include <stdatomic.h>
#include "engine.h"

extern atomic_int stop_requested;
extern int s_batch_mode;

typedef struct {
    char *buf;
    int   len;
    int   cap;
} ReplyBuf;

typedef struct {
    Engine     *engine;
    const char *prompt;
    ReplyBuf   *reply;
} GenArgs;

void inference_start_thread(pthread_t *thread_id);
void inference_stop_thread(pthread_t thread_id);
void run_job(GenArgs *a);