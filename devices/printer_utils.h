#ifndef __DEVUTILS_H__
#define __DEVUTILS_H__

// LIST OF POSSIBLE ERROR CODE RETURNED BY THE STATUS REGISTRER
#define DEV_N_INSTALLED    0
#define DEV_READY          1
#define ILLEGAL_OPCODE     2
#define DEV_BUSY           3
#define PRINT_ERR          4

// LIIST OF THE POSSIBLE COMMAND INPUT TO dev_p->command registrer
#define RESET              0
#define CMD_ACK            1
#define PRINT_CHR          2

#define STATUS_MASK        0xFF

void send_printer(char* buffer);

#endif