#ifndef __TERMUTILS_H__
#define __TERMUTILS_H__

#include "../include/types_bikaya.h"

// LIST OF THE POSSIBLE ERROR CODE RETURNED FROM tp->*_status
#define TERM_N_INSTALLED   0
#define ST_READY           1
#define ILLEGAL_OPCODE     2
#define ST_BUSY            3
#define RECV_ERR           4
#define TRANSM_ERR         4
#define ST_TRANSMITTED     5
#define ST_RECEIVED        5

// LIST OF THE POSSIBLE COMMAND INPUT TO tp->*_command
#define RESET              0
#define CMD_ACK            1       // Interrupt to free the terminal and make it avaiable to other user
#define CMD_TRANSMIT       2
#define CMD_RECEIVE        2       // They both impose to the terminal to make an operation (transmit/receive)

#define CHAR_OFFSET        8       // The data transmitted/received are/shall be placed from (8 to 15 bit) used to shift
#define TERM_STATUS_MASK   0xFF    // 0.0.0.11111111 => 255. Used to mask the first 12 bit (most significant one)
#define DATA_MASK          0xFF00  // The mask to clean the data rcv'd => 0.0.11111111.0

void term_puts(const char *str, unsigned int subdevice);
void term_gets(char usr_input[], unsigned int STR_LENGHT, unsigned int subdevice);

#endif