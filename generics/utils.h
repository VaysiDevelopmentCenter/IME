#ifndef __UTILS_H
#define __UTILS_H

#include "../include/types_bikaya.h"

extern int IO_blocked[MULTIPLE_DEV_LINE + 1][DEV_PER_INT];

void wipe_Memory(void *memaddr, u_int size);
void initNewArea(memaddr handler, memaddr RRF_addr);
void setStatusReg(state_t *proc_state, process_option *option);
void setPC(state_t *process, memaddr function);
void setStackP(state_t *process, memaddr memLocation);
unsigned int getExCode(state_t *oldArea);
void cloneState(state_t *process_state, state_t *old_area, u_int size);
void init_time(time_t *process_time);
void update_time(u_int option, u_int current_time);
void loadCustomHandler(u_int exc_code, state_t *old_area);

#endif