#ifndef __SYSCALL_BREASKPOINT_H
#define __SYCALL_BREAKPOINT_H

#include "../include/types_bikaya.h"

void terminate_process(void* pid);
void syscall_breakpoint_handler(void);

#endif