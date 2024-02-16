#include "../devices/interval_timer_utils.h"
#include "../include/types_bikaya.h"
#include "../include/system_const.h"
#include "../generics/utils.h"
#include "scheduler.h"
#include "asl.h"
#include "pcb.h"


#ifdef TARGET_UMPS
#define IDLE_OPTION { ENABLE_INTERRUPT, KERNEL_MD_ON, ALL_INTRRPT_ENABLED, VIRT_MEM_OFF, PLT_DISABLED }
#endif
#ifdef TARGET_UARM
#define IDLE_OPTION { ENABLE_INTERRUPT, KERNEL_MD_ON, VIRT_MEM_OFF, TIMER_ENABLED }    
#endif


// Ready queue of the scheduler
struct list_head ready_queue;
// Current process selected to be executed
pcb_t *currentProcess = NULL;
// The idle state let the processor active
state_t idleState;



/*
    This function is called by the scheduler after a process is chosen
    for the execution and simply increment by one the priority of all the excluded

    return: void
*/
HIDDEN void aging(void) {
    struct list_head *tmp = NULL;

    list_for_each(tmp, &ready_queue) {
        pcb_t *currentPCB = container_of(tmp, pcb_t, p_next);
        currentPCB->priority++;
    }
}


HIDDEN void idle(void) { while(1) ; }


/*
    Prepares the ready queue and sets the scheduer to be exeuted, also handles
    the setup (with option, SP and PC) of the idle state that is loaded when 
    all process are waiting on their condition.

    return: void
*/
void scheduler_init(void) {
    initPcbs();
    initASL();
    currentProcess = NULL;
    mkEmptyProcQ(&ready_queue);

    // Sets the idle state option
    process_option idle_opt = IDLE_OPTION;
    setStatusReg(&idleState, &idle_opt);
    setStackP(&idleState, (memaddr)_RAMTOP);
    setPC(&idleState, (memaddr)idle);
}


/*
    Adds a new process to the scheduler, checks the arguments first

    p: the PCB pointer to be added to the scheduler
    return: void
*/
void scheduler_add(pcb_t *p) {
    if (p != NULL) {
        // Initialize the time_t struct if it's added for the first time
        init_time(&p->p_time);
        p->original_priority = p->priority;
        insertProcQ(&ready_queue, p);
    }
}


/*
    The scheduler main function, each time that is called put back the currentProc in
    the ready queue and the chose a new process to be executed. If the ready queue is 
    empty then HALT the system else load the chosen process but before ages the priority
    of the excluded and set the currentProc timeslice
*/
void scheduler(void) {
    // If there isn't process in ready_queue nor ASL then there's no process at all (shuts off)
    if (emptyProcQ(&ready_queue) && emptyASL() && currentProcess == NULL) {
        print_debug_terminal("No more process to be executed, shutting off!");
        HALT();
    }
    
    // If no process is ready then idle the process till one is (idle has all interrupt enabled)
     if (emptyProcQ(&ready_queue) && currentProcess == NULL)
       LDST(&idleState);
    
    else {
        // If a process executed before puts it back in the queue
        if (currentProcess != NULL) {
            update_time(USR_MD_TIME, TOD_LO);
            scheduler_add(currentProcess);
        }
        
        // Extracts a new process, restores its priority and ages all the excluded
        currentProcess = removeProcQ(&ready_queue);
        currentProcess->priority = currentProcess->original_priority;
        aging();

        //Set the new "time breakpoint"
        currentProcess->p_time.last_update_time = TOD_LO;

        // Loads the state and executes the chosen process but before sets the time slice
        setIntervalTimer();
        LDST(&currentProcess->p_s);
    }
}


// Returns a pointer to the ready queue
extern inline struct list_head* getReadyQ(void) {
    return(&ready_queue);
}

// Returns the current executing process
extern inline pcb_t* getCurrentProc(void) {
    return(currentProcess);
}


// Sets the current process (usually used to set it to NULL)
extern inline void setCurrentProc(pcb_t *proc) {
    currentProcess = proc;
}