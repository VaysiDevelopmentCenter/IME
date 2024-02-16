/* subset of the Linux Kernel source file: "include/linux/list.h"
CPLv2 */
#ifndef _LISTX_H
#define _LISTX_H

#ifndef NULL
#define NULL ((void *) 0)
#endif
typedef unsigned int    size_tt;

/*
    Macro che restituisce il puntatore all'istanza della struttura che contiene
    un certo list_head. Per farlo usa il puntatore al list_head, il tipo
    della struttura che lo contiene e il nome dato al campo che contiene la
    lista.
    __mptr non e' altro che una copia di ptr con il tipo del campo member; 
    l'indirizzo della struttura si ottiene poi sottraendo a __mptr l'offset
    del campo in questione, usando la macro offsetof.

    ptr: puntatore al list_head della struttura dati di cui si vuole ottenere
        il puntatore
    type: tipo della struttura che contiene list_head e di cui si vuole ottenere
        un puntatore
    member: nome della variabile di tipo list_head contenuta come campo nella
        struttura

    return: puntatore alla struttura dati che contiene il list_head puntato da 
        ptr
*/
#define container_of(ptr, type, member) ({			\
		const typeof( ((type *)0)->member ) *__mptr = (ptr);	\
		(type *)( (char *)__mptr - offsetof(type,member) );})

/* 
    Macro che restituisce l'offset in byte dall'inizio di una struttura a 
    uno dei suoi campi.
    Considera una struttura di tipo TYPE allocata a indirizzo 0 e trova
    l'offset guardando all'indirizzo del campo richiesto; se la struttura
    parte da 0 indirizzo e offset del campo coincidono.

    TYPE: tipo della struttura che contiene il campo di cui si vuole ottenere
        l'offset
    MEMBER: nome del campo di cui si vuole ottenere l'offset

    return: offset in byte dall'inizio della struttura TYPE al campo MEMBER
*/
#define offsetof(TYPE, MEMBER) ((size_tt) &((TYPE *)0)->MEMBER)

/*
    La struttura list_head e' una semplice coppia di puntatore, per implementare
    una lista bidirezionale. Per creare liste di strutture arbitrarie basta
    inserire al loro interno un campo di tipo list_head. 
*/
struct list_head {
	struct list_head *next, *prev;
};

/*
    Macro che definisce una lista vuota, inizializzando una list_head con 
    entrambi i campi che puntano alla lista stessa.

    name: nome della variabile lista da inizializzare

    return: struttura list_head inizializzata come vuota

    Esempio:
    struct list_head lista = LIST_HEAD_INIT(lista);
*/
#define LIST_HEAD_INIT(name) { &(name), &(name) }

/*
    Macro che dichiara e inizializza una nuova lista. Rispetto alla macro
    precedente si occupa anche di dichiarare la variabile.

    name: nome che si vuole dare alla variabile lista
*/
#define LIST_HEAD(name) \
	struct list_head name = LIST_HEAD_INIT(name)

/*
    Funzione inline che inizializza la lista list come vuota (entrambi i campi 
    che puntano a se stessa).
    Mentre LIST_HEAD_INIT crea una struttura anonima da assegnare a una 
    variabile, INIT_LIST_HEAD inizializza i campi di una struttura gia'
    esistente.

    list: lista da inizializzare
    
    return: void
*/
static inline void INIT_LIST_HEAD(struct list_head *list)
{
	list->next = list;
	list->prev = list;
}

/*
    Funzione che inserisce un nuovo elemento (new) tra prev e next

    new: nuovo elemento da inserire
    prev: elemento che deve precedere new
    next: elemento che deve seguire new
*/
static inline void __list_add(struct list_head *new,
		struct list_head *prev,
		struct list_head *next)
{
	next->prev = new;
	new->next = next;
	new->prev = prev;
	prev->next = new;
}

/*
    Funzione che inserisce un nuovo elemento (new) in testa alla lista head.

    new: nuovo elemento da inserire
    head: lista in cui inserire new

    return: void
*/
static inline void list_add(struct list_head *new, struct list_head *head)
{
	__list_add(new, head, head->next);
}

/*
    Come list_add, ma inserisce l'elemento new in coda

    new: nuovo elemento da inserire
    head: lista in cui inserire new

    return: void
*/
static inline void list_add_tail(struct list_head *new, struct list_head *head)
{
	__list_add(new, head->prev, head);
}

/*
    Rimuove gli elementi compresi tra prev e next, collegandoli direttamente.

    prev: punto di partenza del taglio
    next: punto di arrivo del taglio

    return: void
*/
static inline void __list_del(struct list_head * prev, struct list_head * next)
{
	next->prev = prev;
	prev->next = next;
}

/*
    Rimuove l'elemento entry dalla lista in cui e' contenuto.

    entry: elemento da rimuovere

    return: void
*/
static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
}

/*
    Funzione che controlla se la lista e' arrivata alla fine

    list: elemento della lista
    head: inizio della lista

    return: 0 se list non e' l'ultimo elemento, 1 altrimenti
*/
static inline int list_is_last(const struct list_head *list,
		const struct list_head *head)
{
	return list->next == head;
}

/*
    Funzione che controlla se la lista head e' vuota.

    head: lista da controllare

    return: 1 se la lista e' vuota, 0 altrimenti
*/
static inline int list_empty(const struct list_head *head)
{
	return head->next == head;
}

/*
    Funzione che restituisce l'elemento successivo a quello passato come
    parametro.

    current: elemento di cui si richiede il successivo

    return: current->next se la lista non e' vuota, NULL altrimenti
*/
static inline struct list_head *list_next(const struct list_head *current)
{
	if (list_empty(current))
		return NULL;
	else
		return current->next;
}

/*
    Funzione che restituisce l'elemento precedente a quello passato come
    parametro.

    current: elemento di cui si richiede il precedente

    return: current->prev se la lista non e' vuota, NULL altrimenti
*/
static inline struct list_head *list_prev(const struct list_head *current)
{
	if (list_empty(current))
		return NULL;
	else
		return current->prev;
}

/*
    Macro che costruisce un ciclo for per iterare su ogni elemento della lista
    che ha inizio in head. La variabile pos punta al campo struct list_head
    della lista.
    Ad ogni iterazione la variabile pos puntera' a uno degli elementi
    della lista, procedendo in ordine.

    Esempio:
    struct list_head* iter;
	list_for_each(iter,&head) {
		kitem_t* item=container_of(iter,kitem_t,list);
		printf("Elemento i-esimo %d \n",item->elem);
	}

    pos: puntatore da utilizzare per iterare sugli elementi
    head: inizio della lista (elemento sentinella)
*/
#define list_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); pos = pos->next)

/*
    Macro analoga a list_for_each ma che procede a ritroso.

    pos: puntatore da utilizzare per iterare sugli elementi
    head: inizio della lista (elemento sentinella)
*/
#define list_for_each_prev(pos, head) \
	for (pos = (head)->prev; pos != (head); pos = pos->prev)

/*
    Macro che costruisce un ciclo for per iterare sul contenuto di una lista;
    l'unica differenza dispetto a list_for_each e' che la variabile pos punta
     alla struttura che contiene il campo member anziche' al campo stesso.
    
    pos: puntatore da utilizzare per iterare sul contenuto della lista
    head: inizio della lista (elemento sentinella)
    member: nome del campo contenente la list_head
*/
#define list_for_each_entry(pos, head, member)                          \
	for (pos = container_of((head)->next, typeof(*pos), member);      \
	&pos->member != (head);        \
	pos = container_of(pos->member.next, typeof(*pos), member))

/*
    Macro analoga a list_for_each_entry ma che procede a ritroso

    pos: puntatore da utilizzare per iterare sul contenuto della lista
    head: inizio della lista (elemento sentinella)
    member: nome del campo contenente la list_head
*/
#define list_for_each_entry_reverse(pos, head, member)                  \
	for (pos = container_of((head)->prev, typeof(*pos), member);      \
	&pos->member != (head);        \
	pos = container_of(pos->member.prev, typeof(*pos), member))

#endif