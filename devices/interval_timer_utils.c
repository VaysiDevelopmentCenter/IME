#include "../include/system_const.h"
#include "../include/types_bikaya.h"
#include "interval_timer_utils.h"


/*
    Sets the Interval Timer in both architechture to the defined TIMESLICE

    return: void
*/
void setIntervalTimer(void) {
    #ifdef TARGET_UMPS
    memaddr *intervalTimer = (memaddr*) INTERVAL_TIMER;
    *intervalTimer = TIME_SLICE;
    #endif
    
    #ifdef TARGET_UARM
    setTIMER(TIME_SLICE);
    #endif
}


/*
    Set the Interval Timer in both architechture to a choosen timeslice
    that could be different from the default timeslice (3 milliseconds)

    time: the new interval that is desired to be set
    return: void
*/
void setTimerTo(u_int time) {
    // Timer setter on uMPS
    #ifdef TARGET_UMPS
    memaddr *intervalTimer = (memaddr*) INTERVAL_TIMER;
    *intervalTimer = time;
    #endif
    
    // Timer setter on uARM
    #ifdef TARGET_UARM
    setTIMER(time);
    #endif
}


/*
    Returns the current value in the Interval Timer as an unsigned int

    return: the current timer value
*/
u_int getIntervalTimer(void) {
    // Get the current elapsed time since the last timer ssetting uMPS
    #ifdef TARGET_UMPS
    memaddr *intervalTimer = (memaddr*) INTERVAL_TIMER;
    return (*intervalTimer);
    #endif
    
    // Get the current elapsed time since the last timer ssetting uARM
    #ifdef TARGET_UARM
    return (getTIMER());
    #endif
}