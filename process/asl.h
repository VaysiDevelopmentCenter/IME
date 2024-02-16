#ifndef ASL_H
#define ASL_H

#include "../include/types_bikaya.h"

/* ASL handling functions */
semd_t* getSemd(int *key);
unsigned int emptyASL();
void initASL(void);

int insertBlocked(int *key,pcb_t* p);
pcb_t* removeBlocked(int *key);
pcb_t* outBlocked(pcb_t *p);
pcb_t* headBlocked(int *key);
void outChildBlocked(pcb_t *p);

#endif
