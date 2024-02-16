#include "../include/system_const.h"
#include "../include/types_bikaya.h"
#include "term_utils.h"



// This function return the status of the terminal pointer given in input
static unsigned int trans_status(termreg_t *tp) {
    return ((tp->transm_status) & TERM_STATUS_MASK);
}


// Return staus of the terminal pointer for receiving pourpose
static unsigned int recv_status(termreg_t *tp) {
    return((tp->recv_status) & TERM_STATUS_MASK);
} 


/*
    This function takes a char and print it out in the terminal 0.

    return: -1 on failure or 0 on success.
*/
HIDDEN int term_putchar(char c, u_int terminal_subdevice) {
    // Assignation of the terminal file with the built-in macro
    termreg_t *term0_reg = (terminal_subdevice < DEV_PER_INT) ? (termreg_t *) DEV_REG_ADDR(IL_TERMINAL, terminal_subdevice) : NULL;
    // Check that terminal is ready, there are no errors
    unsigned int stat = trans_status(term0_reg);
    if (stat != ST_READY && stat != ST_TRANSMITTED)
        return -1;

    /*
    Shift the char to be transmitted of 8 bits and add the transmit command "opcode",
    the opcode is in the first 8 bit of the message but in next 8 bit there has to be
    the data to be transmitted
    */
    term0_reg->transm_command = ((c << CHAR_OFFSET) | CMD_TRANSMIT);

    // Wait if it's busy (busy waiting!)
    while ((stat = trans_status(term0_reg)) == ST_BUSY)
        ;

    // The acknowledgement makes the terminal avaiable to other once finished
    term0_reg->transm_command = CMD_ACK;

    // Error handler, return error code
    if (stat != ST_TRANSMITTED)
        return -1;
    else
        return 0;
}


/*
    Returns the integer representation of a char that has been read from terminal 0.
    If an error occured then return a opportune error code

    return: -1 on failure, the int ASCII representation of the char on success
*/
HIDDEN int term_getchar(u_int terminal_subdevice) {
    // Assignation of the terminal file with the built-in macro
    termreg_t *term0_reg = (terminal_subdevice < DEV_PER_INT) ? (termreg_t *) DEV_REG_ADDR(IL_TERMINAL, terminal_subdevice) : NULL;
    // Check that ther terminal is ready and there are no error
    unsigned int stat = recv_status(term0_reg); int char_read;
    if (stat != ST_READY && stat != ST_RECEIVED)
        return -1;
    
    // Ask the terminal to receive data
    term0_reg->recv_command = (CMD_RECEIVE);

    // Wait for the terminal to fetch and transfer the data requested 
    while ((stat = recv_status(term0_reg)) == ST_BUSY)
        ;

    /*
    Once exit from the cicle the data may/may not have been fetched but it's stored anyway
    some result in char_read, the data are in bit 8 to 15 (7 or less is the status bit) so
    we need to mask the status and then shift the data with an offset
    */
    char_read = ((term0_reg->recv_status & DATA_MASK) >> CHAR_OFFSET);

    // We now "free" the terminal with the ACK interrupt
    term0_reg->recv_command = CMD_ACK;

    // Checks for error and returns error code
    if (stat != ST_RECEIVED) 
        return -1;
    else
        return (char_read);
}


/*
    Given a NULL terminated string it prints it out on the terminal if not NULL terminated
    it will go to infinite loop.

    str: the string that has to be printed
    return: void
*/
void term_puts(const char *str, u_int subdevice) {
    /*
    The guards in this cicle is given by the if, the method usually returns 0 if everything went
    fine else return -1 that is casted to true by C cso it runs the then "returns" and terminates 
    */
    while (*str)
        if (term_putchar(*str++, subdevice))
            return;
}


/*
    Get a user's input from terminal zero, reading STR_LENGTH characters.
    If the input is shorter will substitute the \n with a \0. In both cases the 
    string returned is NULL terminated.

    usr_input: the char vector were we want to save the string
    STR_LENGHT: the lenght of the input vector
    return: void
*/
void term_gets(char usr_input[], u_int STR_LENGHT, u_int subdevice) {
    int i, letter;

    // Stops at the penultimate index and then puts str terminator at the end (last index)
    for (i = 0; i < STR_LENGHT-1; i = i + 1) {
        letter = term_getchar(subdevice);

        // Error handler, stops the execution and print an error message
        if (letter == -1) {
            term_puts("ERROR: reading from terminal\n", subdevice);
            usr_input[0] = '\0';
            return;
        }
        // If the usr has pressed ENTER then stop the execution and return the string
        else if ((char)letter == '\n') {
            usr_input[i] = 0;
            return;
        }
        else 
            usr_input[i] = (char)letter;
    }

    // Terminate the string with the 0, and return the readed string
    usr_input[i] = '\0';
}