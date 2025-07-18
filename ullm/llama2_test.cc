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

#include <gtest/gtest.h>

#include "llama2.h"

namespace ullm {
namespace {

void OutputHandler(const char* token, void* cookie) {
  std::string* test_output = static_cast<std::string*>(cookie);
  test_output->append(token);
}

TEST(UllmLlama2, Stories15M) {
  const std::string expected_test_output = R"(The birds chirp. Where do they go?
The birds flew around the sky, looking for something to do.
The birds saw a big tree and flew over to it.
The birds saw a big, red apple on the ground. It looked delicious.
The birds flew down and picked up the apple.
The birds flew back up to the tree and started to eat the apple.
The apples were so delicious!
The birds ate until they were full.
The birds flew away, happy and full.
)";

  std::string test_output;
  UllmLlama2RunConfig run_config;
  UllmLlama2RunConfigInit(&run_config);
  run_config.checkpoint_path = "ullm/tinystories15M.bin";
  run_config.tokenizer_path = "ullm/tokenizer.bin";
  run_config.prompt = "The birds chirp. Where do they go?";
  run_config.output_callback = OutputHandler;
  run_config.cookie = &test_output;

  UllmLlama2State state;
  UllmStatus status = UllmLlama2Init(&run_config, &state);
  ASSERT_EQ(status, ULLM_STATUS_OK);
  status = UllmLlama2Generate(&run_config, &state);
  EXPECT_EQ(status, ULLM_STATUS_OK);
  // TODO(aarossig): Add a test to verify that it can run repeatedly.
  UllmLlama2Deinit(&state);

  EXPECT_EQ(test_output, expected_test_output);
}

}  // namespace
}  // namespace ullm
