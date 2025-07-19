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

#include "sys/time.h"

#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

uint64_t UllmTimeNanos() {
#ifdef __APPLE__
  static mach_timebase_info_data_t timebase_info = {};
  if (timebase_info.denom == 0) {
    mach_timebase_info(&timebase_info);
  }

  return mach_absolute_time() * timebase_info.numer / timebase_info.denom;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec * 1000000000) + t.tv_nsec;
#endif
}
