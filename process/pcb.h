#ifndef PCB_H
#define PCB_H
#include "../include/types_bikaya.h"

/* PCB HANDLING FUNCTIONS */

/* PCB free list handling functions */
void initPcbs(void);
void freePcb(pcb_t *p);
pcb_t *allocPcb(void);

/* PCB queue handling functions */
void mkEmptyProcQ(struct list_head *head);
int emptyProcQ(struct list_head *head);
void insertProcQ(struct list_head *head, pcb_t *p);
pcb_t *headProcQ(struct list_head *head);
pcb_t *removeProcQ(struct list_head *head);
pcb_t *outProcQ(struct list_head *head, pcb_t *p);

/* Tree view functions */
int emptyChild(pcb_t *this);
void insertChild(pcb_t *prnt, pcb_t *p);
pcb_t *removeChild(pcb_t *p);
pcb_t *outChild(pcb_t *p);
void populate_PCB_tree(pcb_t *dinasty_vec[], u_int lenght);

#endif
