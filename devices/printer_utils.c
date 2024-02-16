#include "../include/system_const.h"
#include "term_utils.h"
#include "printer_utils.h"


// The printer device that we're currently using (Printer 0)
static dtpreg_t *printer0_reg = (dtpreg_t *) DEV_REG_ADDR(IL_PRINTER, 0);

// Returns the status written in the registrer with the opportune bit mask
static unsigned int dev_status(dtpreg_t *dev_p) {
    return((dev_p->status) & STATUS_MASK);
}


/*
    Takes a character as input and after some device status check print the given character

    return: 0 on success, -1 on failure
*/
HIDDEN unsigned int print_char(char c) {
    unsigned int stat = dev_status(printer0_reg);
    // Check the if the device is ready/installed before using it
    if (stat != DEV_READY || stat == DEV_N_INSTALLED)
        return -1;

    // Insert the char to be printed in the corresponding registrer
    printer0_reg->data0 = c;

    // Then it sends to the device the PRINT command
    printer0_reg->command = PRINT_CHR;
    
    // Wait for the device to execute
    while((stat = dev_status(printer0_reg)) == DEV_BUSY)
        ;

    // After the operation it frees the device that can now be used by other
    printer0_reg->command = CMD_ACK;

    // Error handler, error code returned
    if(stat != DEV_READY)
        return -1;
    else 
        return 0;
}


/*
    Takes as input a pointer to a string/char array that MUST BE null terminated (\0),
    else it will go to infinite loop. If an error occured prints out an error message.

    buffer: the string to print (NULL terminated)
    return: void
*/
void send_printer (char *buffer) {
    while (1) 
        //If we have arrived at the null then we print the \n value
        if ((*buffer) == 0) {
            print_char('\n');
            return;
        }
        //This other statement print the next char and check also from possible errors
        else if (print_char(*buffer++)) {
            term_puts("ERROR: writing into printer device\n", DEBUG_TERMINAL);
            return;
        }           
}