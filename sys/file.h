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

#ifndef ULLM_SYS_FILE_H_
#define ULLM_SYS_FILE_H_

#include <stddef.h>
#include <stdint.h>

#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

// File details.
typedef struct UllmFile
{
  uint64_t offset;
  uint64_t size;
  unsigned char* data;
} UllmFile;

// // Opens the supplied file, and populates the size or returns an error.
// UllmStatus UllmFileOpen(const char* path, UllmFile* file);

// Reads the supplied size into a destination buffer.
UllmStatus UllmFileRead(UllmFile* file, void* dst, uint64_t size);

// // Seeks the file.
UllmStatus UllmFileSeek(UllmFile* file, uint64_t advance);

// // Obtains the current position within the file.
// UllmStatus UllmFileGetPos(const UllmFile* file, uint64_t* pos);

// // Closes the file. This invalidates any pointers to file contents.
// void UllmFileClose(UllmFile* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ULLM_SYS_FILE_H_
