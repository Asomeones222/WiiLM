// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <malloc.h>
// #include <ogcsys.h>
// #include <gccore.h>
// #include <wiiuse/wpad.h>
// #include "c-flags.h"

// static u32 *xfb;
// static GXRModeObj *rmode;


// void Initialise() {
  
// 	VIDEO_Init();
// 	PAD_Init();
// 	// Dolphin compatibility
// 	WPAD_Init();
// 	// Dolphin compatibility
// 	WPAD_SetDataFormat(0, WPAD_FMT_BTNS_ACC_IR);
// 	rmode = VIDEO_GetPreferredMode(NULL);

// 	xfb = MEM_K0_TO_K1(SYS_AllocateFramebuffer(rmode));
// 	console_init(xfb,20,20,rmode->fbWidth,rmode->xfbHeight,rmode->fbWidth*VI_DISPLAY_PIX_SZ);
 
// 	VIDEO_Configure(rmode);
// 	VIDEO_SetNextFramebuffer(xfb);
// 	VIDEO_SetBlack(FALSE);
// 	VIDEO_Flush();
// 	VIDEO_WaitVSync();
// 	if(rmode->viTVMode&VI_NON_INTERLACE) VIDEO_WaitVSync();
// }


// int main() {
 
// 	Initialise();
 
//// 	printf("Hi!\n");
// 	while (1) {
//         WPAD_ScanPads();                           // read controller state
//         u32 pressed = WPAD_ButtonsDown(0);         // buttons pressed this frame
//         if (pressed & WPAD_BUTTON_A)               // check for A
//// 			printf("A clicked!\n");
// 		if (pressed & WPAD_BUTTON_B)               // check for B
// 			break;
//         VIDEO_WaitVSync();                         // throttle loop to video refresh
//     }
//// 	printf("Bye!\n");
// 	return 0;
// }

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

#include <stdio.h>
#include <string.h>

#include "c-flags.h"
#include "llama2.h"
#include "log.h"
#include "macros.h"
#include <stdlib.h>
#include <malloc.h>
#include <ogcsys.h>
#include <gccore.h>
#include <wiiuse/wpad.h>

#define ULLM_LOG_TAG "ullm"

// Args to forward to runners.
typedef struct {
  const char* prompt;
} Args;

void OutputHandler(const char* token, void* cookie) {
  // __fpurge(stdout);
 printf("%s", token);
//  printf("%c", token);
}

UllmStatus UllmRunLlama2(const Args* args) {
//  printf("Called UllmRunLlama2\n");
  UllmLlama2RunConfig run_config;
  UllmLlama2RunConfigInit(&run_config);
  run_config.prompt = args->prompt;
  run_config.output_callback = OutputHandler;

  UllmLlama2State state;
  UllmStatus status = ULLM_STATUS_OK;
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmLlama2Init(&run_config, &state));
  ULLM_GOTO_IF_ERROR(cleanup, status, UllmLlama2Generate(&run_config, &state));

cleanup:
//  printf("Called UllmRunLlama2 cleanup\n");
  UllmLlama2Deinit(&state);
  return status;
}

const struct {
  const char* name;
  UllmStatus (*run)(const Args* args);
} model_runners[] = {
  {
    .name = "llama2",
    .run = UllmRunLlama2,
  },
};

UllmStatus UllmRun(const Args* args) {
//  printf("Called UllmRun\n");
  return model_runners[0].run(args);
}

static u32 *xfb;
static GXRModeObj *rmode;

void Initialise() {

	VIDEO_Init();
	// PAD_Init();
	// Dolphin compatibility
	WPAD_Init();
	// Dolphin compatibility
	WPAD_SetDataFormat(0, WPAD_FMT_BTNS_ACC_IR);
	rmode = VIDEO_GetPreferredMode(NULL);

	xfb = MEM_K0_TO_K1(SYS_AllocateFramebuffer(rmode));

  VIDEO_Configure(rmode);
  VIDEO_SetNextFramebuffer(xfb);
  VIDEO_SetBlack(FALSE);
  VIDEO_Flush();
  VIDEO_WaitVSync();
  if (rmode->viTVMode & VI_NON_INTERLACE)
    VIDEO_WaitVSync();

  // CON_InitEx(rmode, 20, 20, rmode->fbWidth-40, rmode->xfbHeight-40);
  console_init(xfb, 0, 80, rmode->fbWidth, rmode->xfbHeight, rmode->fbWidth * VI_DISPLAY_PIX_SZ);
}

int main(int argc, char** argv) {
  Initialise();
//  printf("Hi!\n");

  char prompt[] = "Once upon a time there was a boy and a girl";
  Args args = {
    .prompt = prompt
  };

//  printf("Args:");
 printf(args.prompt);
 printf(" ");

  setbuf(stdout, NULL);
  UllmStatus status = UllmRun(&args);
  if (status == ULLM_STATUS_OK)
  {
    printf("\n\n Press A to exit...\n");
    while (1)
    {
      WPAD_ScanPads();
      WPAD_SetDataFormat(0, WPAD_FMT_BTNS_ACC_IR);
      u32 pressed = WPAD_ButtonsDown(0) | WPAD_ButtonsDown(1) | WPAD_ButtonsDown(2) | WPAD_ButtonsDown(3);
      if (pressed & WPAD_BUTTON_A)
      {
        break;
      }
      VIDEO_WaitVSync();
    }

    return 0;
  } else {
      ULOGE("Failed to run inference: %s", UllmStatusToString(status));
  }

  return -1;
}
