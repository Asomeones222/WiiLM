/*
 * MIT License
 *
 * Copyright (c) 2023 Andrej
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ULLM_ULLM_LLAMA2_INTERNAL_H_
#define ULLM_ULLM_LLAMA2_INTERNAL_H_

#include <stdint.h>

// Internal data structures for inference.

#ifdef __cplusplus
extern "C" {
#endif

// The hyperparameters of the architecture (the blueprint).
// Loaded from the model bin.
typedef struct {
  int32_t dim;
  int32_t hidden_dim;
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t vocab_size;
  int32_t seq_len;
} __attribute__((packed)) UllmLlama2Config;

typedef struct {
  // token embedding table
  float* token_embedding_table;    // (vocab_size, dim)
  // weights for rmsnorms
  float* rms_att_weight; // (layer, dim) rmsnorm weights
  float* rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float* wq; // (layer, dim, n_heads * head_size)
  float* wk; // (layer, dim, n_kv_heads * head_size)
  float* wv; // (layer, dim, n_kv_heads * head_size)
  float* wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float* w1; // (layer, hidden_dim, dim)
  float* w2; // (layer, dim, hidden_dim)
  float* w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  float* rms_final_weight; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float* wcls;
} UllmLlama2TransformerWeights;

typedef struct {
  // current wave of activations
  float *x; // activation at current time stamp (dim,)
  float *xb; // same, but inside a residual branch (dim,)
  float *xb2; // an additional buffer just for convenience (dim,)
  float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  float *q; // query (dim,)
  float *k; // key (dim,)
  float *v; // value (dim,)
  float *att; // buffer for scores/attention values (n_heads, seq_len)
  float *logits; // output logits
  // kv cache
  float* key_cache;   // (layer, seq_len, dim)
  float* value_cache; // (layer, seq_len, dim)
} UllmLlama2RunState;

typedef struct {
  UllmLlama2Config config;
  UllmLlama2TransformerWeights weights;
  UllmLlama2RunState state;
} UllmLlama2Transformer;

typedef struct {
  const char *str;
  int id;
} UllmLlama2TokenIndex;

typedef struct {
  char** vocab;
  float* vocab_scores;
  UllmLlama2TokenIndex *sorted_vocab;
  char* token_buffer;
} UllmLlama2Tokenizer;

typedef struct {
  float prob;
  int index;
} UllmLlama2ProbIndex;

typedef struct {
  UllmLlama2ProbIndex* probindex;
  uint64_t rng_state;
} UllmLlama2Sampler;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_ULLM_LLAMA2_INTERNAL_H_
