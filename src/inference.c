#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>

static pthread_mutex_t s_job_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  s_job_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  s_job_done  = PTHREAD_COND_INITIALIZER;

static GenArgs *s_job_args    = NULL;
static int      s_job_pending = 0;
static int      s_job_running = 0;
static int      s_inf_exit    = 0;

static void reply_append(ReplyBuf *r, const char *piece) {
    if (!r) return;
    int plen = (int)strlen(piece);
    if (r->len + plen + 1 > r->cap) {
        r->cap = r->cap ? r->cap * 2 : 4096;
        if (r->len + plen + 1 > r->cap) r->cap = r->len + plen + 1;
        r->buf = realloc(r->buf, r->cap);
    }
    memcpy(r->buf + r->len, piece, plen);
    r->len += plen;
    r->buf[r->len] = '\0';
}

static int token_cb(Engine *e, int token_id, void *ctx) {
    if (atomic_load(&stop_requested)) return 1;

    const char *piece = engine_decode_token(e, token_id);

    // Печатаем в stdout только если мы не в пакетном режиме
    if (!s_batch_mode) {
        fputs(piece, stdout);
        fflush(stdout);
    }

    reply_append((ReplyBuf *)ctx, piece);
    return 0;
}

static void *inf_thread(void *arg) {
    (void)arg;
    pthread_mutex_lock(&s_job_mutex);
    for (;;) {
        while (!s_job_pending && !s_inf_exit)
            pthread_cond_wait(&s_job_ready, &s_job_mutex);

        if (s_inf_exit) {
            pthread_mutex_unlock(&s_job_mutex);
            return NULL;
        }

        GenArgs *a    = s_job_args;
        s_job_pending = 0;
        s_job_running = 1;
        pthread_mutex_unlock(&s_job_mutex);

        engine_generate(a->engine, a->prompt, a->max_tokens, token_cb, a->reply);

        pthread_mutex_lock(&s_job_mutex);
        s_job_running = 0;
        pthread_cond_signal(&s_job_done);
    }
}

void inference_start_thread(pthread_t *thread_id) {
    pthread_create(thread_id, NULL, inf_thread, NULL);
}

void inference_stop_thread(pthread_t thread_id) {
    pthread_mutex_lock(&s_job_mutex);
    s_inf_exit = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);
    pthread_join(thread_id, NULL);
}

void run_job(GenArgs *a) {
    pthread_mutex_lock(&s_job_mutex);
    s_job_args    = a;
    s_job_pending = 1;
    s_job_running = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);

    for (;;) {
        pthread_mutex_lock(&s_job_mutex);
        int done = !s_job_running;
        pthread_mutex_unlock(&s_job_mutex);
        if (done) break;

        fd_set fds; FD_ZERO(&fds); FD_SET(STDIN_FILENO, &fds);
        struct timeval tv = { 0, 20000 }; /* 20 мс */
        if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
            char ch;
            if (read(STDIN_FILENO, &ch, 1) == 1 && ch == 27) {
                atomic_store(&stop_requested, 1);
                break;
            }
        }
    }

    pthread_mutex_lock(&s_job_mutex);
    while (s_job_running)
        pthread_cond_wait(&s_job_done, &s_job_mutex);
    pthread_mutex_unlock(&s_job_mutex);
}