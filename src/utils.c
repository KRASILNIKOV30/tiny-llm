#include "utils.h"
#include "math_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

float *copy_f32(const float *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    memcpy(dst, src, n * sizeof(float));
    return dst;
}

float *copy_f16(const uint16_t *src, size_t n) {
    if (!src) return NULL;
    float *dst = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) dst[i] = f16_to_f32(src[i]);
    return dst;
}

char* read_file_content(const char* path) {
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

void write_json(const char *path, const char *prompt, const char *response) {
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

void write_json_ppl(const char *path, float *probs, int probs_len) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ошибка: не удалось открыть файл для записи JSON\n");
        return;
    }

    fprintf(f, "{\n  \"target_probs\": [");
    for (int i = 0; i < probs_len; i++) {
        fprintf(f, "%.6f%s", probs[i], (i == probs_len - 1) ? "" : ", ");
    }
    fprintf(f, "]\n}\n");
    fclose(f);
}
