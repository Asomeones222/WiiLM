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

#ifndef ULLM_UTIL_LOG_H_
#define ULLM_UTIL_LOG_H_

#ifdef __cplusplus
extern "C" {
#endif

// The possible log levels.
#define ULLM_LOG_LEVEL_ERROR 0
#define ULLM_LOG_LEVEL_INFO 1
#define ULLM_LOG_LEVEL_DEBUG 2
#define ULLM_LOG_LEVEL_VERBOSE 3

// Emit logs.
#define ULOGE(fmt, ...) UllmLog(ULLM_LOG_TAG, ULLM_LOG_LEVEL_ERROR, \
    __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define ULOGI(fmt, ...) UllmLog(ULLM_LOG_TAG, ULLM_LOG_LEVEL_INFO, \
    __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define ULOGD(fmt, ...) UllmLog(ULLM_LOG_TAG, ULLM_LOG_LEVEL_DEBUG, \
    __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define ULOGV(fmt, ...) UllmLog(ULLM_LOG_TAG, ULLM_LOG_LEVEL_VERBOSE, \
    __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// Emits a log. This is the back end of the logging macros.
__attribute__((format (printf, 5, 6)))
void UllmLog(const char* tag, int level, const char* file, int line,
    const char* fmt, ...);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_UTIL_LOG_H_
