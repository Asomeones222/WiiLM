#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <ogcsys.h>
#include <gccore.h>
#include <wiiuse/wpad.h>
#include "c-flags.h"

static u32 *xfb;
static GXRModeObj *rmode;


void Initialise() {
  
	VIDEO_Init();
	PAD_Init();
	// Dolphin compatibility
	WPAD_Init();
	// Dolphin compatibility
	WPAD_SetDataFormat(0, WPAD_FMT_BTNS_ACC_IR);
	rmode = VIDEO_GetPreferredMode(NULL);

	xfb = MEM_K0_TO_K1(SYS_AllocateFramebuffer(rmode));
	console_init(xfb,20,20,rmode->fbWidth,rmode->xfbHeight,rmode->fbWidth*VI_DISPLAY_PIX_SZ);
 
	VIDEO_Configure(rmode);
	VIDEO_SetNextFramebuffer(xfb);
	VIDEO_SetBlack(FALSE);
	VIDEO_Flush();
	VIDEO_WaitVSync();
	if(rmode->viTVMode&VI_NON_INTERLACE) VIDEO_WaitVSync();
}


int main() {
 
	Initialise();
 
	printf("Hi!\n");
	while (1) {
        WPAD_ScanPads();                           // read controller state
        u32 pressed = WPAD_ButtonsDown(0);         // buttons pressed this frame
        if (pressed & WPAD_BUTTON_A)               // check for A
			printf("A clicked!\n");
		if (pressed & WPAD_BUTTON_B)               // check for B
			break;
        VIDEO_WaitVSync();                         // throttle loop to video refresh
    }
	printf("Bye!\n");
	return 0;
}
