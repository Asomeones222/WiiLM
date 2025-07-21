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

#include "memory.h"
#include "log.h"

#define ULLM_LOG_TAG "ullm.file"

UllmStatus UllmFileOpen(const char* path, UllmFile* file) {
  memset(file, 0, sizeof(UllmFile));

  struct stat st;
  if (stat(path, &st) != 0) {
    ULOGE("Failed to stat file '%s': %s (%d)", path, strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  file->fd = open(path, O_RDONLY);
  if (file->fd < 0) {
    ULOGE("Failed to open file '%s': %s (%d)", path, strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  file->size = st.st_size;
  ULOGI("Opened file '%s' with size %" PRIu64, path, file->size);
  return ULLM_STATUS_OK;
}

UllmStatus UllmFileRead(const UllmFile* file, void* dst, uint64_t size) {
  ssize_t bytes_read = read(file->fd, dst, size);
  if (bytes_read < 0) {
    ULOGE("Failed to read file: %s (%d)", strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  return ULLM_STATUS_OK;
}

UllmStatus UllmFileSeek(const UllmFile* file, uint64_t advance) {
  off_t result = lseek(file->fd, advance, SEEK_CUR);
  if (result < 0) {
    ULOGE("Failed to seek file: %s (%d)", strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  return ULLM_STATUS_OK;
}

UllmStatus UllmFileGetPos(const UllmFile* file, uint64_t* pos) {
  off_t result = lseek(file->fd, 0, SEEK_CUR);
  if (result < 0) {
    ULOGE("Failed to seek file for pos: %s (%d)", strerror(errno), errno);
    return ULLM_STATUS_IO_ERROR;
  }

  return ULLM_STATUS_OK;
}

void UllmFileClose(UllmFile* file) {
  if (file->fd >= 0) {
    close(file->fd);
  }
}
