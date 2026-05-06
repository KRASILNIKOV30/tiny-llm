// Интерактивный CLI для запуска инференса модели Qwen2.5

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/select.h>
#include <sys/time.h>

#include "engine.h"
#include "chat.h"

// Вспомогательные функции терминала

static struct termios saved_termios;

static void term_raw(void) {
    tcgetattr(STDIN_FILENO, &saved_termios);
    struct termios raw = saved_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);
}

static void handle_sigint(int sig) {
    (void)sig;
    term_restore();
    printf("\n");
    exit(0);
}

// Общие данные для главного потока и инференс-потока.

static atomic_int stop_requested = 0;  // если 1, то остановить генерацию
static int s_batch_mode = 0;

// Буфер ответов, заполняемы в token_cb. Используется для записи ответов ассистента.
typedef struct {
    char *buf;
    int   len;
    int   cap;
} ReplyBuf;

static void reply_append(ReplyBuf *r, const char *piece) {
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

// Коллбэк тоенов: печатает каждый токен, добавляет его в буфер ответа.
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

// Инференс-поток.

typedef struct {
    Engine     *engine;
    const char *prompt;  // сформатированная строка в формате ChatML, время жизни управляется вызывающим кодом
    ReplyBuf   *reply;
} GenArgs;

static pthread_t       s_inf_thread;
static pthread_mutex_t s_job_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  s_job_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  s_job_done  = PTHREAD_COND_INITIALIZER;

static GenArgs *s_job_args    = NULL; // текущая задача, NULL = простой
static int      s_job_pending = 0;    // 1 = ожидается новая задача
static int      s_job_running = 0;    // 1 = задача в прогрессе
static int      s_inf_exit    = 0;    // 1 = нужно завершить генерацию

static void *inf_thread(void *arg) {
    (void)arg;
    pthread_mutex_lock(&s_job_mutex);
    for (;;) {
        // Ожидаем, пока не появится новая задача
        while (!s_job_pending && !s_inf_exit)
            pthread_cond_wait(&s_job_ready, &s_job_mutex);

        if (s_inf_exit) {
            pthread_mutex_unlock(&s_job_mutex);
            return NULL;
        }

        // Забираем задачу
        GenArgs *a    = s_job_args;
        s_job_pending = 0;
        s_job_running = 1;
        pthread_mutex_unlock(&s_job_mutex);

        // Запускаем инференс
        engine_generate(a->engine, a->prompt, token_cb, a->reply);

        pthread_mutex_lock(&s_job_mutex);
        s_job_running = 0;
        pthread_cond_signal(&s_job_done);
    }
}

// Запускаем задачу и ждём завершения, также ждём нажатия Escape для отмены.
static void run_job(GenArgs *a) {
    // Опубликовать задачу
    pthread_mutex_lock(&s_job_mutex);
    s_job_args    = a;
    s_job_pending = 1;
    s_job_running = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);

    // Отслеживаем Escape, пока работает инференс
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

    // Ожидаем завершения инференс-потока
    pthread_mutex_lock(&s_job_mutex);
    while (s_job_running)
        pthread_cond_wait(&s_job_done, &s_job_mutex);
    pthread_mutex_unlock(&s_job_mutex);
}

// Прочитать строку из stdin.
// Возвращает 1, если нажат Escape.
static int read_user_input(char *buf, int maxlen) {
    term_restore();

    fputs(">> ", stdout);
    fflush(stdout);

    int n = 0;
    int c;

    c = fgetc(stdin);
    if (c == 27) {          /* Escape */
        term_raw();
        return 1;
    }
    if (c == EOF || c == '\n') {
        buf[0] = '\0';
        term_raw();
        return 0;
    }
    buf[n++] = (char)c;

    if (fgets(buf + n, maxlen - n, stdin) == NULL) {
        buf[n] = '\0';
    } else {
        int len = (int)strlen(buf);
        if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
    }

    term_raw();
    return 0;
}

// Чтение содержимого файла целиком
static char* read_file_content(const char* path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *string = malloc(fsize + 1);
    if (fread(string, 1, fsize, f) != (size_t)fsize) {
        free(string);
        fclose(f);
        return NULL;
    }
    fclose(f);
    string[fsize] = '\0';
    return string;
}

// Запись результата в JSON с базовым экранированием спецсимволов
static void write_json(const char *path, const char *prompt, const char *response) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ошибка: не удалось открыть файл для записи JSON\n");
        return;
    }

    fprintf(f, "{\n  \"prompt\": \"");
    for(const char *c = prompt; *c; c++) {
        if(*c == '"') fprintf(f, "\\\"");
        else if(*c == '\\') fprintf(f, "\\\\");
        else if(*c == '\n') fprintf(f, "\\n");
        else if(*c == '\r') fprintf(f, "\\r");
        else if(*c == '\t') fprintf(f, "\\t");
        else fputc(*c, f);
    }

    fprintf(f, "\",\n  \"response\": \"");
    for(const char *c = response; *c; c++) {
        if(*c == '"') fprintf(f, "\\\"");
        else if(*c == '\\') fprintf(f, "\\\\");
        else if(*c == '\n') fprintf(f, "\\n");
        else if(*c == '\r') fprintf(f, "\\r");
        else if(*c == '\t') fprintf(f, "\\t");
        else fputc(*c, f);
    }
    fprintf(f, "\"\n}\n");
    fclose(f);
}

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *prompt_file = NULL;
    const char *output_json = NULL;

    // Парсинг аргументов
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch-mode") == 0) {
            s_batch_mode = 1;
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (strcmp(argv[i], "--output-json") == 0 && i + 1 < argc) {
            output_json = argv[++i];
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "ошибка: неизвестный аргумент '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Использование: %s <model.gguf> [--batch-mode] [--prompt-file <path>] [--output-json <path>]\n", argv[0]);
        return 1;
    }

    if (s_batch_mode && !prompt_file) {
        fprintf(stderr, "ошибка: для --batch-mode требуется указать --prompt-file\n");
        return 1;
    }

    signal(SIGINT, handle_sigint);

    Engine *e = engine_load(model_path);
    if (!e) {
        fprintf(stderr, "ошибка загрузки модели.\n");
        return 1;
    }

    // Запуск инференс-потока (общий для обоих режимов)
    pthread_create(&s_inf_thread, NULL, inf_thread, NULL);

    // ==========================================
    // ПАКЕТНЫЙ РЕЖИМ (HEADLESS)
    // ==========================================
    if (s_batch_mode) {
        char *user_msg = read_file_content(prompt_file);
        if (!user_msg) {
            fprintf(stderr, "ошибка: не удалось прочитать %s\n", prompt_file);
            exit(1);
        }

        ChatHistory history;
        chat_init(&history, NULL);
        chat_append(&history, ROLE_USER, user_msg);

        // Форматируем в ChatML для Qwen
        char *prompt = chat_format_delta(&history, 0, 0, 1);

        ReplyBuf reply = { NULL, 0, 0 };
        GenArgs args = { e, prompt, &reply };

        atomic_store(&stop_requested, 0);
        run_job(&args); // Блокирующий вызов, ждем генерацию всего ответа

        if (output_json) {
            write_json(output_json, user_msg, reply.buf ? reply.buf : "");
        } else {
            // Если output_json не указан, просто выводим результат в stdout
            printf("%s\n", reply.buf ? reply.buf : "");
        }

        free(user_msg);
        free(prompt);
        free(reply.buf);
        chat_free(&history);

    }
        // ==========================================
        // ИНТЕРАКТИВНЫЙ РЕЖИМ
        // ==========================================
    else {
        fprintf(stderr, "Чат с Qwen2.5-0.5B-Instruct (нажмите Esc или Ctrl-C для выхода)\n\n");

        term_raw(); // Включаем сырой режим терминала только для чата

        ChatHistory history;
        chat_init(&history, NULL);
        char user_msg[4096];
        ReplyBuf reply = { NULL, 0, 0 };
        int sent_msgs = 0;

        for (;;) {
            int esc = read_user_input(user_msg, sizeof(user_msg));
            if (esc || user_msg[0] == '\0') break;

            printf("\n");
            fflush(stdout);

            chat_append(&history, ROLE_USER, user_msg);
            int close_prev = (sent_msgs > 0);
            char *prompt = chat_format_delta(&history, sent_msgs, close_prev, 1);

            reply.len = 0;
            if (reply.buf) reply.buf[0] = '\0';

            atomic_store(&stop_requested, 0);
            GenArgs args = { e, prompt, &reply };
            run_job(&args);
            free(prompt);

            if (atomic_load(&stop_requested)) break;

            if (reply.buf && reply.len > 0) {
                chat_append(&history, ROLE_ASSISTANT, reply.buf);
                sent_msgs = history.len;
            } else {
                sent_msgs = history.len;
            }

            printf("\n\n");
            fflush(stdout);
        }

        term_restore();
        printf("\nПока.\n");
        free(reply.buf);
        chat_free(&history);
    }

    // ==========================================
    // ОСТАНОВКА И СТАТИСТИКА
    // ==========================================
    pthread_mutex_lock(&s_job_mutex);
    s_inf_exit = 1;
    pthread_cond_signal(&s_job_ready);
    pthread_mutex_unlock(&s_job_mutex);
    pthread_join(s_inf_thread, NULL);

    EngineStats stats;
    engine_get_stats(e, &stats);
    if (stats.prefill_tokens > 0 || stats.gen_tokens > 0) {
        fprintf(stderr, "\n--- Статистика ---\n");
        if (stats.prefill_tokens > 0)
            fprintf(stderr, "prefill : %ld токенов  %.0f мс  %.1f ток/сек\n",
                    stats.prefill_tokens, stats.prefill_ms,
                    stats.prefill_tokens / (stats.prefill_ms / 1e3));
        if (stats.gen_tokens > 0)
            fprintf(stderr, "generate: %ld токенов  %.0f мс  %.1f ток/сек\n",
                    stats.gen_tokens, stats.gen_ms,
                    stats.gen_tokens / (stats.gen_ms / 1e3));
    }

    engine_free(e);
    return 0;
}
