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

#ifndef ULLM_ULLM_LLAMA2_H_
#define ULLM_ULLM_LLAMA2_H_

#include <stdint.h>

#include "file.h"
#include "llama2-internal.h"
#include "status.h"

// The public API for inference for Llama-2 Transformer model.

#ifdef __cplusplus
extern "C" {
#endif

// The runtime state for the inference engine.
typedef struct {
  UllmLlama2Transformer transformer;
  UllmLlama2Tokenizer tokenizer;
  UllmLlama2Sampler sampler;
} UllmLlama2State;

// The runtime config for the inference operation.
typedef struct {
  // The prompt to generate a response to.
  const char* prompt;

  // Model configuration.
  float temperature;
  float topp;
  unsigned int steps;

  // The source of entropy.
  uint64_t rng_seed;

  // The callback and context for generated output.
  void (*output_callback)(const char* token, void* cookie);
  void* cookie;
} UllmLlama2RunConfig;

// Default initialize an UllmLlama2RunConfig.
void UllmLlama2RunConfigInit(UllmLlama2RunConfig* config);

// Load models and prepare state to run inference.
UllmStatus UllmLlama2Init(const UllmLlama2RunConfig* config,
    UllmLlama2State* state);

// Runs UllmLlama2 inference in generate mode with the supplied config.
UllmStatus UllmLlama2Generate(const UllmLlama2RunConfig* config,
    UllmLlama2State* state);

// Release UllmLlama2 state.
void UllmLlama2Deinit(UllmLlama2State* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_ULLM_LLAMA2_H_
