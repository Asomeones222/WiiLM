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

#include "file.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
// #include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory.h"
#include "log.h"

#define ULLM_LOG_TAG "ullm.file"

// UllmStatus UllmFileOpen(const char* path, UllmFile* file) {
//   memset(file, 0, sizeof(UllmFile));

//   struct stat st;
//   if (stat(path, &st) != 0) {
//     ULOGE("Failed to stat file '%s': %s (%d)", path, strerror(errno), errno);
//     return ULLM_STATUS_IO_ERROR;
//   }

//   file->fd = open(path, O_RDONLY);
//   if (file->fd < 0) {
//     ULOGE("Failed to open file '%s': %s (%d)", path, strerror(errno), errno);
//     return ULLM_STATUS_IO_ERROR;
//   }

//   file->size = st.st_size;
//   ULOGI("Opened file '%s' with size %" PRIu64, path, file->size);
//   return ULLM_STATUS_OK;
// }

UllmStatus UllmFileRead(UllmFile* file, void* dst, uint64_t size) {
  // ssize_t bytes_read = read(file->fd, dst, size);
  memcpy(dst, file->data + file->offset, size);
  file->offset += size;
  // const uint8_t *bytes = (const uint8_t *)dst;

  // char *hex_str = malloc(size * 3 + 1); // 2 hex chars per byte + null terminator

  // for (size_t i = 0; i < size; i++)
  //   sprintf(&hex_str[i * 3], "%02X|", bytes[i]);
  // hex_str[size * 3] = '\0';
  // if(size < 12)
  // {
  //   printf("starting hex %02X", file->data[file->offset - size]);
  //   printf("Printing copied bytes: %s with size %" PRIu64 "\n", hex_str, size);
  // }
  // free(hex_str);
  // if (bytes_read < 0)
  // {
  //   ULOGE("Failed to read file: %s (%d)", strerror(errno), errno);
  //   return ULLM_STATUS_IO_ERROR;
  // }

  return ULLM_STATUS_OK;
}

UllmStatus UllmFileSeek(UllmFile* file, uint64_t advance) {
  file->offset += advance;
  return ULLM_STATUS_OK;
}

// UllmStatus UllmFileGetPos(const UllmFile* file, uint64_t* pos) {
//   off_t result = lseek(file->fd, 0, SEEK_CUR);
//   if (result < 0) {
//     ULOGE("Failed to seek file for pos: %s (%d)", strerror(errno), errno);
//     return ULLM_STATUS_IO_ERROR;
//   }

//   return ULLM_STATUS_OK;
// }

// void UllmFileClose(UllmFile* file) {
//   if (file->fd >= 0) {
//     close(file->fd);
//   }
// }
