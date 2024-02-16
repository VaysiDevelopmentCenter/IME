#ifndef _TYPES_BIKAYA_H
#define _TYPES_BIKAYA_H

#include "system_const.h"
#include "listx.h"


typedef unsigned int memaddr;
typedef unsigned int u_int;

// Custom exception handling for PCB (defined with specpassup syscall)
typedef struct handler_t {
    // Matrix of custom handlers
    state_t* handler_matrix [CSTM_HNDLRS] [HANDLER_AREAS];
    // Boolean vector for keep record of wich custom handler has been activated
    unsigned int has_custom[3];
} handler_t;



// Struct for process time accounting
typedef struct time_t {
    unsigned int usermode_time;
    unsigned int kernelmode_time;
    unsigned int activation_time;
    unsigned int last_update_time;
} time_t;



// Process Control Block (PCB) data structure 
typedef struct pcb_t {
    // Process queue fields 
    struct list_head p_next;

    // Process tree fields
    struct pcb_t *   p_parent;
    struct list_head p_child, p_sib;

    // Processor state, etc
    state_t p_s;
    // Process time analitics
    time_t p_time;

    // Process priority (aged by the scheduling process)
    int priority;
    // Original process priority
    int original_priority;

    // Key of the semaphore on which the process is eventually blocked
    int *p_semkey;

    // Set of possible custom exception handler for the process
    handler_t custom_handler;
 
} pcb_t;



// Semaphore Descriptor (SEMD) data structure
typedef struct semd_t {
    struct list_head s_next;

    // Semaphore key
    int *s_key;

    // Queue of PCBs blocked on the semaphore
    struct list_head s_procQ;
} semd_t;



// Auxiliary structure for option register setting in both architecture
// Used to setting better the option of a given process
#ifdef TARGET_UMPS
typedef struct {
    unsigned int interruptEnabled;
    unsigned int kernelMode;
    unsigned int interrupt_mask; // Require an additional btmask to mask individuale lines
    unsigned int virtualMemory;
    unsigned int timerEnabled; // This sets the PLT
} process_option;
#endif

#ifdef TARGET_UARM
typedef struct {
    unsigned int interruptEnabled;
    unsigned int kernelMode;
    unsigned int virtualMemory;
    unsigned int timerEnabled; // This sets the Interval Timer
} process_option;
#endif

#endif