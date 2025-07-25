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

#include "memory.h"
#include <stdlib.h>
#include <stdio.h>
#include "log.h"
#include <stdint.h>
#include <inttypes.h>

#define ULLM_LOG_TAG "ullm.sys.memory"

void *UllmMemoryAlloc(uint32_t size)
{
//  // printf("Called UllmMemoryAlloc with size: %" PRIu32 "\n", size);

  // ULOGD("Allocating %zu bytes", size);
  return malloc(size);
}

void UllmMemoryFree(void* ptr) {
//  // printf("attempted free");
  // free(ptr);
}
