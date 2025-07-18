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

#include "log.h"

#include <stdarg.h>
#include <stdio.h>

char UllmLogLevelChar(int level) {
  switch (level) {
    case ULLM_LOG_LEVEL_ERROR:
      return 'E';
    case ULLM_LOG_LEVEL_INFO:
      return 'I';
    case ULLM_LOG_LEVEL_DEBUG:
      return 'D';
    case ULLM_LOG_LEVEL_VERBOSE:
      return 'V';
    default:
      return 'X';
  }
}

void UllmLog(const char* tag, int level, const char* file, int line,
    const char* fmt, ...) {
  // Print level and tag.
  fprintf(stderr, "%c %s: ", UllmLogLevelChar(level), tag);

  // Print the formatted log.
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);

  // Print a trailing newline.
  fprintf(stderr, "\n");
}
