#pragma once

#include <stddef.h>
#include <stdint.h>

// Текущее время в миллисекундах
double now_ms(void);

// Выделяет память и копрует массив float32
float *copy_f32(const float *src, size_t n);

// Выделяет память и деквантизирует массив float16 → float32
float *copy_f16(const uint16_t *src, size_t n);

char* read_file_content(const char* path);
void write_json(const char *path, const char *prompt, const char *response);
void write_json_ppl(const char *path, float *probs, int probs_len);
