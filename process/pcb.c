#include "../include/system_const.h"
#include "../include/types_bikaya.h"
#include "../include/listx.h"
#include "../generics/utils.h"
#include "pcb.h"

HIDDEN pcb_t pcbTable[MAXPROC];
HIDDEN LIST_HEAD(pcbFree);



/*
    This function uses the array pcbTmp_arr and for every and each PCB it adds it to
    the pcbFree_queue. It simply initializes the free queue of PCBs

    param: void
    return: void
*/
void initPcbs(void) {
    for (u_int i = 0; i < MAXPROC; i++)
        list_add_tail(&(pcbTable[i].p_next), &pcbFree);
}

/*
    This function add the given PCB to the pcbFree list.

    WARNING: the PCB must be already removed from all the other list
    (siblings list, ready queue or semaphor queue, etc), this function doesn't
    control all this thing before adding the PCB, this could lead to inconsistence.

    p: the PCB wich has to be returned to the pcbFree_queue
    return: void
*/
void freePcb(pcb_t *p) {
    if (p != NULL)
        list_add_tail(&p->p_next, &pcbFree);
}

/*
    Function that removes a PCB from the pcbFree_queue if not already empty, 
    wipes the PCB, initialize some fields (p_child and p_sib)
    to empty list and returns it.

    return: the new allocated PCB or NULL if not avaiable
*/
pcb_t *allocPcb(void) {
    //Returns NULL if the pcbFree is empty (no free pcbs avaiable)
    struct list_head *tmp = list_next(&pcbFree); 

    //Error checking
    if (tmp == NULL)
        return (NULL);
    
    
    //Delete the pcb from the pcbFree_queue, obtain the pcb_t struct with "container_of" and return it
    pcb_t *newPcb = container_of(tmp, pcb_t, p_next);
    list_del(&newPcb->p_next);

    //Wipes the PCB and initialize his list to empty list
    wipe_Memory(newPcb, sizeof(pcb_t));
    INIT_LIST_HEAD(&newPcb->p_next);
    INIT_LIST_HEAD(&newPcb->p_child);
    INIT_LIST_HEAD(&newPcb->p_sib);

    return(newPcb);
}

/*

    This function simply initialize a new active PCB queue and setting the given pointer to be
    the dummy of such queue. Note that the given pointer must be only declared and not set
    to anything active (to avoid any type of bug), make sure that is always NULL.

    head: pointer to the "dummy" of pcbAtvive_queue
    return: void
*/
void mkEmptyProcQ(struct list_head *head) {
    INIT_LIST_HEAD(head);
}

/*
    Check if the given list_head pointer is an empty list/queue

    head: the dummy of the list/queue we want to check
    return: 1 if the list is empty, 0 else
*/
int emptyProcQ(struct list_head *head) {
    return(list_empty(head));
}

/*
    Insert the given PCB p to the pcbActive_queue, maintaining
    the sorting by priority of the queue, returns NULL if the args are invalid

    head: the pointer to the dummy of the queue
    p: the pointer to the pcb we want to add
*/
void insertProcQ(struct list_head *head, pcb_t *p) {
    struct list_head *tmp;
    pcb_t *last_examined_pcb;
    //Initial check that the arguments are correct
    if (head == NULL || p == NULL)
        return;

    //If the list is empty then it adds up directly
    else if (list_empty(head))
        list_add(&p->p_next, head);

    //Insert the element maintaining the sorting property of the queue
    else {
        list_for_each(tmp, head) {
            last_examined_pcb = container_of(tmp, pcb_t, p_next);

            //If the PCB has to stay in the middle of the queue, adds it in between and breaks
            if (p->priority > last_examined_pcb->priority) {
                list_add(&p->p_next, tmp->prev);
                return;
            }
        }

        //If the cicle loops til the end then the pcb has to be put in the queue tail (as last element)
        list_add_tail(&p->p_next, head);
    }
}

/*
    This function returns a reference the first element of the pcb_active_queue,
     so the first PCB in the priority queue (after checking for errors). 
    NOTE: that it doesn't remove the PCB from the queue (for that see removeProcQ() below).

    head: the pointer to the dummy of the queue
    return: the PCB pointer or NULL for errors
*/
pcb_t *headProcQ(struct list_head *head) {
    if (head == NULL || list_empty(head))
        return (NULL);

    return(container_of(list_next(head), pcb_t, p_next));
}

/*
    This function returns the first PCB in the queue as headProcQ but it
    removes it from the queue instead of only returning a reference to it.

    head: the pointer to the dummy of the queue
    return: the PCB pointer or NULL for errors
*/
pcb_t *removeProcQ(struct list_head *head) {
    pcb_t *toRemove = headProcQ(head);

    if (toRemove != NULL)
        list_del(&toRemove->p_next);
    
    return(toRemove);
}

/*
    This function looks in the pcbActive_queue for the pcb given as argument, and returns
    it once and if found. But before it removes it from the queue mentioned above.

    head: the pointer to the dummy of the queue
    p: the process we want to remove from the queue
    return: NULL if error, the requested PCB on success
*/
pcb_t *outProcQ(struct list_head *head, pcb_t *p) {
    struct list_head *tmp;

    if (head == NULL || list_empty(head) || p == NULL)
        return (NULL);

    list_for_each(tmp, head) {
        pcb_t *block = container_of(tmp, pcb_t, p_next);
        
        if (p == block) { //If there's a match returns the found/given pcb
            list_del(tmp);
            return (block);
        }
    }
    //If there's no match then returns NULL
    return (NULL);
}

/*
    This function check that the given PCB has no childs. If the argument
    is NULL then returnes FALSE.

    this: a pointer to the PCB we want to check
    return: 1 if the PCB has no child, 0 else (for errors as well)
*/
int emptyChild(pcb_t *this) {
    return (this != NULL && list_empty(&this->p_child));
}

/*
    This function insert the p PCB in the child list of the PCB prnt
    It doesn't check that p has another father when added so it must
    be checked NULL

    prnt: the PCB wich is the father
    p: the PCB wich has to be inserted
    return: void
*/
void insertChild(pcb_t *prnt, pcb_t *p) {
    p->p_parent = prnt;
    list_add_tail(&p->p_sib, &prnt->p_child);
}

/*
    This function removes the first child of the given PCB p
    
    p: the PCB of wich we want to obtain the first son in the list
    return: NULL if the PCB p has no child, the first son on success
*/
pcb_t *removeChild(pcb_t *p) {
    struct list_head *tmp = &p->p_child;

    if (p == NULL || list_empty(tmp))
        return (NULL);

    tmp = list_next(tmp);
    list_del(tmp);
    return(container_of(tmp, pcb_t, p_sib));
    
}

/*
    Removes the PCB p from his siblings list (also the child list of the 
    father), if p has no parent or the PCB isn't in the child list of the 
    father returns NULL

    p: the PCB we want to remove from the child list
    return: the given PCB if found, else NULL
*/
pcb_t *outChild(pcb_t *p) {
    if (p->p_parent == NULL)
        return(NULL);
    
    struct list_head *tmp, *siblingsList = &p->p_parent->p_child;

    list_for_each(tmp, siblingsList) {
        pcb_t *block = container_of(tmp, pcb_t, p_sib);
        
        if (p == block) { // If there's a match returns the found/given pcb
            list_del(tmp);
            return (block);
        }
    }
    //If there's no match then returns NULL
    return (NULL);
}

/*
    This function given an array of pcb pointer with the first [0] initialized to the root process,
    extrapolates all the radicated tree in dynasty_vec[0], then returnes. The program wouldn't go in
    buffer overflow but is strictly advised to use a MAXPROC vector lenght. 

    dynasty_vec: the vector in wich save the dynasty of vec[0], that has to be previously initialized
    length: a integer representing the lenght of the above vector
    retunr: void
*/
void populate_PCB_tree(pcb_t *dynasty_vec[], u_int lenght) {
    u_int first_free_space = 1;
    
    // For some reason the compiler doesn't set automatically to NULL all the cell
    for (int i = first_free_space; i < lenght; i++)
        dynasty_vec[i] = NULL;

    for (u_int i = 0; (i < lenght) && (first_free_space < lenght); i++) {
        struct list_head *tmp = NULL;
        pcb_t *current = dynasty_vec[i];
        
        // No more PCB to evaluate
        if (current == NULL)
           return ;

        // Insert the current process child as well in the vector
        list_for_each(tmp, &current->p_child) {
            dynasty_vec[first_free_space] = container_of(tmp, pcb_t, p_sib);
            first_free_space++;
        }
        
    }
}