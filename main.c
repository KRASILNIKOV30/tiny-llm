#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include <stdatomic.h>

#include "engine.h"
#include "chat.h"
#include "utils.h"
#include "inference.h"

atomic_int stop_requested = 0;
int s_batch_mode = 0;

static struct termios saved_termios;

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios);
}

static void handle_sigint(int sig) {
    (void)sig;
    term_restore();
    printf("\n");
    exit(0);
}

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *prompt_file = NULL;
    const char *output_json = NULL;
    int s_eval_ppl = 0;

    // Парсинг аргументов
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch-mode") == 0) {
            s_batch_mode = 1;
        } else if (strcmp(argv[i], "--eval-ppl") == 0) {
            s_eval_ppl = 1;
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

    // Сохранение терминала и установка обработчика сигналов
    tcgetattr(STDIN_FILENO, &saved_termios);
    signal(SIGINT, handle_sigint);

    Engine *e = engine_load(model_path);
    if (!e) {
        fprintf(stderr, "ошибка загрузки модели.\n");
        return 1;
    }

    pthread_t inf_thread_id;
    inference_start_thread(&inf_thread_id);

    if (s_batch_mode) {
        char *user_msg = read_file_content(prompt_file);
        if (!user_msg) exit(1);

        if (s_eval_ppl) {
            int probs_len = 0;
            float *probs = engine_eval_sequence(e, user_msg, &probs_len);

            if (output_json && probs) {
                write_json_ppl(output_json, probs, probs_len);
            } else if (probs) {
                printf("Собрано %d вероятностей.\n", probs_len);
            }
            free(probs);
        } else {
            ChatHistory history;
            chat_init(&history, NULL);
            chat_append(&history, ROLE_USER, user_msg);
            char *prompt = chat_format_delta(&history, 0, 0, 1);

            ReplyBuf reply = { NULL, 0, 0 };
            GenArgs args = { e, prompt, &reply };

            atomic_store(&stop_requested, 0);
            run_job(&args);

            if (output_json) {
                write_json(output_json, user_msg, reply.buf ? reply.buf : "");
            }

            free(prompt);
            free(reply.buf);
            chat_free(&history);
        }
        free(user_msg);
    }

    // Чистое завершение потока и модели
    inference_stop_thread(inf_thread_id);

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