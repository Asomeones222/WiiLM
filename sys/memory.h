/*
 * Copyright 2025 Andrew Rossignol andrew.rossignol@gmail.com
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ULLM_SYS_MEMORY_H_
#define ULLM_SYS_MEMORY_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Allocate memory.
void* UllmMemoryAlloc(size_t size);

// Release memory.
void UllmMemoryFree(void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_SYS_MEMORY_H_
