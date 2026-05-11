#pragma once

// Ядро инференса модели Qwen2.5

#include <stdint.h>
#include "gguf.h"
#include "hashmap.h"

// Opaque-структура, детали реализации в engine.c
typedef struct Engine Engine;

// Вызывается один раз на каждый сгенерированный токен.
typedef int (*TokenCallback)(Engine *e, int token_id, void *ctx);

// Запускает движок, загружая веса из GGUF-файла по указанному пути.
// Возвращает NULL в случае ошибки.
Engine *engine_load(const char *model_path);

// Запускает инференс на предформатированную строку промпта.
// Вызыввет cb(engine, token_id, cb_ctx) на каждый сгенерированный токен.
void engine_generate(Engine *e, const char *prompt, int max_tokens,
                     TokenCallback cb, void *cb_ctx);

// Декодирует ID токена в UTF8-строку.
// Возвращённый указатель валиден до следующего вызова.
const char *engine_decode_token(Engine *e, int token_id);

// Статистика генерации.
typedef struct {
    long   prefill_tokens;
    double prefill_ms;
    long   gen_tokens;
    double gen_ms;
} EngineStats;

void engine_get_stats(const Engine *e, EngineStats *out);

void engine_free(Engine *e);

// Оценивает последовательность текста и возвращает массив вероятностей (после softmax)
// для каждого правильного следующего токена.
// Массив нужно освободить через free(). Количество элементов запишется в *out_len.
float *engine_eval_sequence(Engine *e, const char *text, int *out_len);

// Получает сырые логиты (размером vocab_size) для следующего токена после переданного текста
float *engine_get_logits(Engine *e, const char *text);

// Обертка для токенизации строки без доступа к внутренностям Engine
int engine_encode(Engine *e, const char *text, int *out);

// Возвращает размер словаря модели
int engine_get_vocab_size(Engine *e);

// Добавить в src/engine.h
void engine_set_layer_mask(Engine *e, const char *mask_str);

void engine_set_head_mask(Engine *e, const char *mask_str);
