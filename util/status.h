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

#ifndef ULLM_UTIL_STATUS_H_
#define ULLM_UTIL_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

// Returns if the status is not OK.
#define ULLM_RETURN_IF_ERROR(status) \
  do { \
    UllmStatus _status = (status); \
    if (_status != ULLM_STATUS_OK) { \
      return _status; \
    } \
  } while(0)

#define ULLM_GOTO_IF_ERROR(label, status_var, status) \
  do { \
    status_var = (status); \
    if (status_var != ULLM_STATUS_OK) { \
      goto label; \
    } \
  } while(0)

// The possible status of performing inference.
typedef enum {
  // Success.
  ULLM_STATUS_OK,

  // Invalid arguments were supplied.
  ULLM_STATUS_INVALID_ARGUMENT,

  // There was an error accessing a file.
  ULLM_STATUS_IO_ERROR,

  // A memory allocation failed.
  ULLM_STATUS_OOM,
} UllmStatus;

// Returns a string representation of a given status.
const char *UllmStatusToString(UllmStatus status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_UTIL_STATUS_H_
