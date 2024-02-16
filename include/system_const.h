#ifdef TARGET_UMPS
#include "uMPS/libumps.h"
#include "uMPS/arch.h"
#include "uMPS/types.h"
#include "uMPS/umps/cp0.h"
#endif
#ifdef TARGET_UARM
#include "uARM/uarm/libuarm.h"
#include "uARM/uarm/arch.h"
#include "uARM/uarm/uARMtypes.h"
#include "uARM/uarm/uARMconst.h"
#endif

#include "../devices/term_utils.h"

/**************************************************************************** 
 *
 * This header file contains the global constant & macro definitions
 * As well as target architechture includes for both uARM and uMPS
 * 
 ****************************************************************************/



/*=================== GENERIC & GLOBAL CONSTANT =====================*/
#define MAXPROC 20  // Max number of overall (eg, system, daemons, user) concurrent processes 
#define UPROCMAX 3  // Number of usermode processes (not including master proc and system daemons
#define DEFAULT_PRIORITY 1

#define	HIDDEN static
#define	TRUE 	1
#define	FALSE	0
#define ON      1
#define OFF 	0
#define EOS   '\0'
#define CR     0x0a   // Carriage return as returned by the terminal
#define TOD_LO     *((unsigned int *)BUS_REG_TOD_LO)

// Function to follow another flow of information on terminal 1
#define DEBUG_TERMINAL 1
#define print_debug_terminal(str) term_puts(str, DEBUG_TERMINAL)

// Generic info about the devices 
#define MAX_LINE 8
#define UNIQUE_DEV_LINE 3
#define MULTIPLE_DEV_LINE 5
#define WORDSIZE 4
#define DEV_PER_INT 8
#define DEV_REGISTER_SIZE 4
#define REGISTER_PER_DEV 4

#define TIME 3000
#define TIME_SLICE (TIME * TIME_SCALE)
#define USR_MD_TIME 0
#define KER_MD_TIME 1

#define OFFSET_INT 8

#ifndef NULL
        #define NULL ((void *) 0)
#endif



/* ======== CONSTANTS FOR STATE_T OPTION/PARAMETERS ============= */
#ifdef TARGET_UMPS
        // Status registrer bits for enabling/disabling interrupts in the given process
        #define DISABLE_INTERRUPT    0
        #define ENABLE_INTERRUPT     1 
        #define IEP_SHIFT 2
        #define INTERRUPT_MASK_SHIFT 8

        // Interrupt bitmask (only for uMPS ignored by uARM)
        #define ALL_INTRRPT_ENABLED 0xFF
        #define ALL_INTRRPT_DISABLED 0x00
        #define ONLY_TIMER_ENABLED 0x04

        // Status registrer bits for enabling/disabling kernel mode in the given process
        #define KERNEL_MD_ON    0
        #define USR_MD_ON       1
        #define KM_SHIFT        1

        // Status registrer bits for enabling/disabling virtual memory in the given process
        #define VIRT_MEM_ON      1
        #define VIRT_MEM_OFF     0
        #define VIRT_MEM_SHIFT   24

        // Status registrer bits for enabling/disabling timer in the given process
                // In uMPS this sets the PLT that is not used (always off)
        #define PLT_DISABLED  0
        #define PLT_SHIFT     27
#endif

#ifdef TARGET_UARM
        // Option to disable or enable all interrupt lines (is not possible to activite only some as in uMPS)
        #define DISABLE_INTERRUPT 0
        #define ENABLE_INTERRUPT  1

        // Status registrer bits for enabling/disabling kernel mode in the given process
        #define KERNEL_MD_ON 1
        #define USR_MD_ON    0

        // Status registrer bits for enabling/disabling virtual memory in the given process
        #define VIRT_MEM_ON      1
        #define VIRT_MEM_OFF     0

        // Status registrer bits for enabling/disabling Interval Timer in the given process
        #define TIMER_ENABLED  1
        #define TIMER_DISABLED 0
#endif



/* ==================== OLD/NEW AREAS AND RRF address =========================== */
#ifdef TARGET_UMPS
        // uMPS New/Old Areas address
        #define NEW_AREA_SYSCALL   0x200003d4
        #define OLD_AREA_SYSCALL   0x20000348

        #define NEW_AREA_TRAP      0x200002bc
        #define OLD_AREA_TRAP      0x20000230

        #define NEW_AREA_TLB       0x200001a4
        #define OLD_AREA_TLB       0x20000118

        #define NEW_AREA_INTERRUPT 0x2000008c
        #define OLD_AREA_INTERRUPT 0x20000000

        // uMPS's beginning address of RAM and size of a RAM page
        #define RAMBASE    *((unsigned int *)BUS_REG_RAM_BASE)
        #define RAMSIZE    *((unsigned int *)BUS_REG_RAM_SIZE)
        #define _RAMTOP     (RAMBASE + RAMSIZE)
        #define RAM_FRAMESIZE  4096

        //Time areas 
        #define INTERVAL_TIMER BUS_REG_TIMER
        #define TIME_SCALE     *((unsigned int *)BUS_REG_TIME_SCALE)
#endif

#ifdef TARGET_UARM
        // uARM New/Old Areas address
        #define NEW_AREA_SYSCALL   SYSBK_NEWAREA 
        #define OLD_AREA_SYSCALL   SYSBK_OLDAREA 

        #define NEW_AREA_TRAP      PGMTRAP_NEWAREA
        #define OLD_AREA_TRAP      PGMTRAP_OLDAREA

        #define NEW_AREA_TLB       TLB_NEWAREA
        #define OLD_AREA_TLB       TLB_OLDAREA

        #define NEW_AREA_INTERRUPT INT_NEWAREA
        #define OLD_AREA_INTERRUPT INT_OLDAREA

        // uARM's beginning address of RAM and size of a RAM page
        #define _RAMTOP RAM_TOP
        #define RAM_FRAMESIZE FRAME_SIZE

        // Time areas
        #define INTERAVAL_TIMER 0x000002E4
        #define TIME_SCALE *((unsigned int *)BUS_REG_TIME_SCALE)
#endif



/* ======================== EXCEPTION HANDLING MACROS ============================== */
#ifdef TARGET_UMPS
        // Get the exception code from the cause register
        #define CAUSE_GET_EXCCODE(x)    (((x) & CAUSE_EXCCODE_MASK) >> CAUSE_EXCCODE_BIT)

        #define SYSCALL_CODE            8
        #define BREAKPOINT_CODE         9
        #define INTERRUPT_CODE 0

        #define EXC_HANDLER_PROC_OPT { DISABLE_INTERRUPT, KERNEL_MD_ON, ALL_INTRRPT_DISABLED, VIRT_MEM_OFF, PLT_DISABLED }
#endif

#ifdef TARGET_UARM
        // Get the exception code from the cause register
        #define CAUSE_GET_EXCCODE(x)    ((x) & 0xFFFFFF)

        #define SYSCALL_CODE            SYSEXCEPTION
        #define BREAKPOINT_CODE         BPEXCEPTION
        #define INTERRUPT_CODE INTEXCEPTION

        #define EXC_HANDLER_PROC_OPT { DISABLE_INTERRUPT, KERNEL_MD_ON, VIRT_MEM_OFF, TIMER_DISABLED }
#endif



/* ========================= INTERRUPT HANDLING MACROS =============================== */
#ifdef TARGET_UMPS
        #define LINE_MASK 0xFF00
        #define LINE_OFFSET 8

        // Code for each interrupts line
        #define INTER_PROCESSOR 0
        #define PROCESSOR_LOCAL_TIMER 1
        #define BUS_INTERVAL_TIMER 2
        #define DISK_DEVICE 3
        #define TAPE_DEVICE 4
        #define NETWORK_DEVICE 5
        #define PRINTER_DEVICE 6
        #define TERMINAL_DEVICE 7

        #define INTER_DEVICES_BASE 0x1000003C
#endif

#ifdef TARGET_UARM
        // Code for each interrupts line (0,1,5 are not used in uARM)
        #define INTER_PROCESSOR 0
        #define PROCESSOR_LOCAL_TIMER 1
        #define BUS_INTERVAL_TIMER INT_TIMER
        #define DISK_DEVICE INT_DISK
        #define TAPE_DEVICE INT_TAPE
        #define NETWORK_DEVICE INT_UNUSED
        #define PRINTER_DEVICE INT_PRINTER
        #define TERMINAL_DEVICE INT_TERMINAL

        #define INTER_DEVICES_BASE 0x10006FE0
#endif

#define DTP_STATUS MASK 
#define DEV_STATUS_REG(dp) ((dp->status))
#define TERM_STATUS_MASK 0xFF
#define TRANSM_STATUS(tp) ((tp->transm_status) & TERM_STATUS_MASK)
#define RECV_STATUS(tp) ((tp->recv_status) & TERM_STATUS_MASK)

#define CMD_ACK 1
#define DVC_NOT_INSTALLED 0
#define DVC_BUSY 3

#define INTER_DEVICES(line) (INTER_DEVICES_BASE + (line - 3) * WS)



/* ========================= MACROS FOR REGISTER ACCESS =============================== */
#ifdef TARGET_UMPS
        #define STATUS_REG(state) state->status
        #define PC_REG(state)     state->pc_epc
        #define SP_REG(state)     state->reg_sp
        #define CAUSE_REG(state)  state->cause
        #define SYSCALL_NO(state) state->gpr[3]
        #define SYS_ARG_1(state) state->gpr[4]
        #define SYS_ARG_2(state) state->gpr[5]
        #define SYS_ARG_3(state) state->gpr[6]
        #define SYS_RETURN_VAL(state) state->gpr[1]
#endif

#ifdef TARGET_UARM
        #define STATUS_REG(state) state->cpsr
        #define PC_REG(state)     state->pc
        #define SP_REG(state)     state->sp
        #define CAUSE_REG(state)  state->CP15_Cause
        #define SYSCALL_NO(state) state->a1
        #define SYS_ARG_1(state) state->a2
        #define SYS_ARG_2(state) state->a3
        #define SYS_ARG_3(state) state->a4
        #define SYS_RETURN_VAL(state) state->a1
#endif



/* ========================= MACROS FOR SYSCALLS =============================== */
// Syscall's number and name
#define GETCPUTIME       1
#define CREATEPROCESS    2
#define TERMINATEPROCESS 3
#define VERHOGEN         4
#define PASSEREN         5
#define WAITIO           6
#define SPECPASSUP       7
#define GETPID           8

// Status code after syscall execution
#define FAILURE -1
#define SUCCESS 0



/* ============= MACROS FOR CUSTOM TLB, TRAP & SYSCALL HANDLER ============== */
// Code for specific exception access
#define SYS_BP_COSTUM      0
#define TLB_CUSTOM         1
#define TRAP_CUSTOM        2
// Vector dimension in handler_t struct
#define CSTM_HNDLRS        3
#define HANDLER_AREAS      2
// Vector dimension in handler_t stuct
#define CSTM_OLD_AREA      0
#define CSTM_NEW_AREA      1