e# Lab0: "Hello world!"
The goal of the lab is to get used with `aos-mini-linux`, learn how to write, load and unload a module. 
```c
#include <linux/module.h> // manages module
#include "linux/printk.h" // manages print function, it is actually not needed since module.h includes kernel.h which includes printk.h
#include "linux/stat.h" // manages file permission constants and macros used for file mode checking

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CLR");
MODULE_DESCRIPTION("hello world!");

// parameters can be declared and initialized in code
static int num = 5;
module_param(num, int, S_IRUGO);

// parameters can also be defined at module instantiation time with the command:
// insmod /modules/lab-1.... num=10
// S_IRUGO makes the parameter read-only from user space
// S_IWURS makes the paramter user-readable and user-writable!

// initialization of the module
// // run when the module is inserted (insmod)
static int __init my_module_init(void) {
  // the family of pr_*() functions allows printing from kernel to the log buffer
  pr_info("Hello world!\n");
  pr_info("Address of hello_init = %px\n", hello_init);
  return 0;
}

static void __exit my_module_exit(void) {
	pr_info("Module exit!\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
```

> [!NOTE] `static` keyword
> The use of keyword `static` limit the visibility of a function/variable to within the same source file.
> It is a best practice in Kernel coding.
> 
> Advantages:
> - **Encapsulation**: keeps helper functions/variables private to the file.    
> - **Avoid symbol collisions**.
> - **Enables compiler optimization**.
>   
>   Without `static` the function/variable becomes a global symbol, visible across the whole kernel.

# Lab1: high resolution timer
The goal of the lab is to understand how a `hr_timer` work.
```c
#include <linux/module.h> //manage modules
#include <linux/slab.h> //manage memory
#include <linux/hrtimer.h> //manage timers
#include <linux/prandom.h> //magage pseudo-random generator

MODULE_AUTHOR("CLR");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("A module that manipulates a list (im)properly.");

// define 1 second in nanoseconds
#define ONE_SEC 1000000000L

#define MAX_INVOCATIONS 10
static int count = 0;

// define a list structure
struct node {
	int data;
	struct node *next;
};

struct node *head;
struct hrtimer my_timer; // define a global hrtimer

// add a node to the top of the list
static void add_node(int data){
	struct node *new_node = kmalloc(sizeof(*new_node), GFP_KERNEL);
	new_node -> data = data;
	new_node -> next = head;
	head = new_node;
}

// define timer callback
enum hrtimer_restart my_timer_handler(struct hrtimer *timer){
	// ISO C90 forbids mixed declarations and code, so declarations must be on top
	// also called ANSI C standard
	struct node *curr = head; // take reference to top of the list
	int rand;
	
	pr_info("timer handler invoked\n");
	rand = prandom_u32(); // generate a pseudo-random number
	add_node(rand); // add the random value to the list
	pr_info("Added node with value %d at %px\n", rand, head);
	
	// the pointer value is actually replaced by '(____ptrval____)' when using %p
	// this is an intentional kernel behaviour for security
	// the setting 'CONFIG_KPTR_RESTRICT' determines whether a pointer can be printed
	// using %px prints the raw pointer
	
	// remove the last element of the list
	
	// uncomment this code the see the kernel panic
	//while(curr){ //BUG!!
	//	if (!curr->next->next){
	//		kfree(curr->next);
	//		curr->next = NULL;
	//	} else {
	//		curr = curr->next;
	//	}
	//}
	// BUG: in exploring the list there is NO check that curr->next is NULL
	// so curr->next->next will crash if curr->next is NULL.
	// this happens when the list has only 1 node. 
	
	// As expected the module triggers kernel panic due to NULL pointer deference
	
	// fixed code
	while (curr && curr->next) {
		if (curr->next->next == NULL) {
			kfree(curr->next);
			curr->next = NULL;
		}
		curr = curr->next;
	}
	
	if (++count >= MAX_INVOCATIONS) {
	    pr_info("Max timer invocations reached. Stopping.\n");
	    return HRTIMER_NORESTART; // tells the kernel to stop the timer
	}	
	// reschedules timer for another trigger in 1 second
	// it is alwyas RELATIVE 
	hrtimer_forward_now(timer, ns_to_ktime(ONE_SEC));
	return HRTIMER_RESTART; // tells the kernel to keep the timer going
}

static int __init my_module_init(void){
	// create a list 0 -> 1 -> 2
	int i;
	for(i=0; i<3; i++){
		add_node(i);
	}

	// initializes the timer
	// CLOCK_MONOTONIC: does not change with system time, generally used to misure intervals, 
	// 		    it avoid drift from system time changes
	// HRTIMER_MODE_REL: the timer is set to expire after a duration relative to current time
	hrtimer_init(&my_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	my_timer.function = &my_timer_handler; // set callback

	// start the timer
	pr_info("setting up timer to start in 3 seconds\n");
	hrtimer_start(&my_timer, ns_to_ktime(3 * ONE_SEC), HRTIMER_MODE_REL);

	return 0;
}

static void __exit my_module_exit(void) {
	struct node *curr = head;

	// free the linked list
	while (curr) {
		struct node *temp = curr;
		curr = curr->next;
		kfree(temp);
	}

	// cancel the timer
	hrtimer_cancel(&my_timer);
}

module_init(my_module_init);
module_exit(my_module_exit);
```
# Lab3: atomic operations
The goal of the lab is to understand why atomic operations are needed. 
We use implicit hardware-level locking via atomic instructions (lock-free synchronization).
Since we are using threads, we have to run QEMU using more than one processor (e.g. 4), otherwise the time-share between `add_thread` and `substract_thread` will not be truly concurrently, but one after another (interleaved).
```c
#include "linux/moduleparam.h" //manage passing parameters to the module, it includes also stat.h
#include <linux/kthread.h> //manage kernel treads
#include <linux/module.h> //manage modules
// atomic.h is already included into kernel.h which is included in module.h

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CLR");
MODULE_DESCRIPTION("Demonstrate concurrent access to a shared variable from two kernel threads under two scenarios: non-atomic version (broken) and atomic version.");

// handles paraneter to the module
static int variant = 0;
// insmod /modules/.... variant = 0 # runs non-atomic (broken) version
// insmod /modules/.... variant = 1 # runs atomic version
module_param(variant, int, S_IRUGO); // allows setting 'variant' when inserting the module

// define the mechanism to modify the `shared_variable`
volatile int64_t shared_variable = 0;
volatile int value = 10; // value subtracted/added at each iteration
volatile uint64_t iter = (1 << 20); //number of iterations for each thread (2^20)

static int add_thread(void* data){
	int64_t i; // since the number of iteration is very large, to be sure that it does not overflow
	for(i=0; i<iter; i++){
		// NO locking/atomic operation
		// race condition if multiple CPUs modify the variable at the same time
		shared_variable += value;
	}
	pr_info("[ADD] finished: %lld\n", shared_variable);
  	return 0;
}

static int subtract_thread(void* data){
	int64_t i;
	for(i=0; i<iter; i++){
		// NO locking/atomic operation
		shared_variable -= value;
	}
	pr_info("[SUB] finished: %lld\n", shared_variable);
  	return 0;
}

// creates and run two kernel threads
// this will lead to data corruption due to concurrent writes to 'shared_variables'
// we would expect 'shared_variable=0' at the end of substract thread, but this does not happen!
void non_atomic_version(void) {
  kthread_run(add_thread, NULL, "add_thread");
  kthread_run(subtract_thread, NULL, "subtract_thread");
}

atomic64_t atomic_shared_variable = ATOMIC_INIT(0);

static int atomic_add_thread(void *data) {
  uint64_t i;
  for (i = 0; i < iter; i++) {
    atomic64_add(value, &atomic_shared_variable); //uses atomic add to avoid race condition
  }
  pr_info("[ADD] finished: %lld\n", atomic_shared_variable.counter);
  return 0;
}

static int atomic_subtract_thread(void *data) {
  uint64_t i;
  for (i = 0; i < iter; i++) {
    atomic64_sub(value, &atomic_shared_variable); //uses atomic sub to avoid race condition
  }
  pr_info("[SUB] finished: %lld\n", atomic_shared_variable.counter);
  return 0;
}

// as expected 'atomic_shared_variable' is 0 at the end of the substract thread!
void atomic_version(void) {
	kthread_run(atomic_add_thread, NULL, "atomic_add_thread");
  	kthread_run(atomic_subtract_thread, NULL, "atomic_subtract_thread");
}

static int __init my_module_init(void){
	// chooses which variant to run based on the 'variant' parameter
	if(variant == 0){
		pr_info("Module loaded in non-atomic version!\n");	
		non_atomic_version();
	} else {
		pr_info("Module loaded in atomic version!\n");
		atomic_version();
	}
	
	return 0;
}

static void __exit my_module_exit(void){
	pr_info("Module unloaded!\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
```


> [!NOTE] `volatile` keyword
> The `volatile` keyword tells the compiler **not to optimize access** to a variable: every read/write must go **directly to memory**, not to a register or cache.
> 
> It is used when a variable can be changed outside the program's control, in this case in another thread.

# Lab3: RCU (Read-Copy-Update)
The goal of the lab is to understand how concurrent read works. 
```c
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kthread.h>
#include <linux/slab.h>
#include <linux/delay.h> // manages sleep functions

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CLR");
MODULE_DESCRIPTION("Demonstrates concurrent read to a shared linked list under two scenarios: using RCU (Read-Copy-Update) and not using it (unsafe access)");

// the goal is to show a cuncurrent READ without proper synchronization
// since writes are protected by spinlocks

// variant = 0 run the unsafe version
// variant = 1 run the safe version 
static int variant = 0;
module_param(variant, int, S_IRUGO);

// list element structure
struct list_element {
  int data;
  struct list_head list;
};

// define list and a spinlock
static LIST_HEAD(my_list);
static DEFINE_SPINLOCK(list_lock);

// read and print the list without locking 
// it generates kernel oops: general protection fault caused by accessing an invalid address
// the kernel thread is dereferencing a pointer that points to invalid/freed memory
static int read_list_unsafe(void *data){
	// kthread_should_stop check if the current thread has been asked to stop
	while(!kthread_should_stop()){
		struct list_element *entry;
		pr_info("[ ");
		list_for_each_entry(entry, &my_list, list){
			pr_info("%d ", entry -> data);
		}
		pr_info(" ]\n");
		msleep(100); // sleeps for 1 second
	}
	return 0;
}

// remove first element of the list (if any), increment its value and add it back
static int manipulate_list_unsafe(void *data){
	while(!kthread_should_stop()){
		struct list_element *entry, *temp;
		entry = kmalloc(sizeof(struct list_element), GFP_KERNEL);
		
		// use a spinlock for concurrent writes
		spin_lock(&list_lock);
		
		if (!list_empty(&my_list)) {
			temp = list_first_entry(&my_list, struct list_element, list);
			list_del(&temp->list);
			entry->data = temp->data + 1;
			kfree(temp);
		}
		list_add(&entry->list, &my_list);

		spin_unlock(&list_lock);
		msleep(200);
	}
	return 0;
}

// read and print the list with RCU locking 
// the execution does not stop until we unload the module
static int read_list_rcu(void *data){
	while(!kthread_should_stop()){
		struct list_element *entry;
		
		rcu_read_lock();
		
		pr_info("[ ");
		list_for_each_entry(entry, &my_list, list){
			pr_info("%d ", entry -> data);
		}
		pr_info(" ]\n");
		
		rcu_read_unlock();

		msleep(100); // sleeps for 1 second
	}
	return 0;
}

struct rcu_head pending_deletes;

// same as previous using RCU locking
// we use RCU-aware versions of the classic list manipulation functions 
static int manipulate_list_rcu(void *data){
	while(!kthread_should_stop()){
		struct list_element *entry, *temp;
		
		// use a spinlock for concurrent writes
		spin_lock(&list_lock);
		
		entry = kmalloc(sizeof(struct list_element), GFP_KERNEL);
		
		if (!list_empty(&my_list)) {
			temp = list_first_entry(&my_list, struct list_element, list);
			list_del_rcu(&temp->list);
			entry->data = temp->data + 1;
			
			synchronize_rcu();
			kfree(temp);
		}
		list_add_rcu(&entry->list, &my_list);

		spin_unlock(&list_lock);
		msleep(200);
	}
	return 0;
}

// thread pointers
static struct task_struct *read_thread;
static struct task_struct *manipulate_thread;

// NB: since 'read_list_unsafe' and 'manipulate_list_unsafe' are threads function they MUST accept one void* arguments and they MUST return an integer  (even if unused) 
void unsafe_list_manip(void) {
  read_thread = kthread_run(read_list_unsafe, NULL, "read_list_thread");
  manipulate_thread = kthread_run(manipulate_list_unsafe, NULL, "manipulate_list_thread");
}

//NB: same for 'read_list_rcu' and 'manipulate_list_rcu'
void rcu_list_manip(void) {
  read_thread = kthread_run(read_list_rcu, NULL, "read_list_thread");
  manipulate_thread = kthread_run(manipulate_list_rcu, NULL, "manipulate_list_thread");
}

static void initialize_list(void) {
  int i;
  for (i = 0; i < 10; i++) {
    struct list_element *entry = kmalloc(sizeof(struct list_element), GFP_KERNEL);
    entry->data = i;
    // initializes an empty list with both 'next' and 'prev' pointer of the list head point to itself
    INIT_LIST_HEAD(&entry->list);
    
    spin_lock(&list_lock);
    list_add_tail(&entry->list, &my_list);
    spin_unlock(&list_lock);
  }
}

static int __init my_module_init(void){
	pr_info("Module loaded!\n");
	initialize_list();
	if(variant == 0){
		unsafe_list_manip();
	} else {
		rcu_list_manip();
	}
	
	if (read_thread && manipulate_thread) {
		pr_info("Kernel threads created and started\n");
	} else {
		pr_info(KERN_ERR "Failed to create kernel threads\n");
		return -ENOMEM; // out of memory: Error NO MEMory
	}
	
	return 0;
}

static void __exit my_module_exit(void){
	struct list_element *entry, *temp;
	
	// stop the threads
	if (read_thread && manipulate_thread) {
		kthread_stop(read_thread);
		kthread_stop(manipulate_thread);
	}
	
	// free the list
	spin_lock(&list_lock);
	list_for_each_entry_safe(entry, temp, &my_list, list) {
		list_del(&entry->list);
		kfree(entry);
	}
	spin_unlock(&list_lock);
	
	pr_info("Module unloaded!\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
```

# Lab4: virtual memory allocation
```c
#include <linux/module.h>
#include <linux/slab.h>  

#include <asm/io.h>   // virt_to_phys
#include <linux/mm.h> // vmalloc_to_page etc..
#include <linux/random.h>
#include <linux/slub_def.h> // kmalloc
#include <linux/vmalloc.h>  //vmalloc


MODULE_AUTHOR("CLR");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Demonstrate Linux allocation techniques and kernel-level memory inspection.");

// global buffer pointer
static void *buffer;

// virtual page number MACRO
// converts an address to a virtual page number by shifting rigth by 'page_shift'
#define PN(x) ((void *)((unsigned long long)(x) >> PAGE_SHIFT))

// print NUMA memory zones
static int print_zones(void){
	int node_id = numa_node_id();
	// get current NUMA node id and its associated pglist_data (per node page allocator data)
	struct pglist_data *pgdat = NODE_DATA(node_id);
	int zone_id;

	pr_info("Memory zones for NUMA node %d: \n", node_id);
	for(zone_id = 0; zone_id < MAX_NR_ZONES; zone_id++){
		struct zone *zone = &pgdat -> node_zones[zone_id];
		
		if(zone->present_pages){
			unsigned long start_pfn = zone -> zone_start_pfn;
			unsigned long end_pfn = zone_end_pfn(zone);
			
			pr_info("Zone %d - Start PPN: 0x%lx, End PPN: 0x%lx\n", zone_id, start_pfn, end_pfn);
		}
	}	
	
	return 0;
}

// allocation with kmalloc
static int alloc_kmalloc(int n){
	size_t buffer_size = n * PAGE_SIZE;
	
	buffer = kmalloc(buffer_size, GFP_KERNEL);
	if(!buffer){
		pr_info(KERN_ERR "Failed to allocate the buffer\n");
		return -ENOMEM;
	}
	
	// translate virtual to physical page number with the macro
	pr_info("kmalloc - VPN: %px -> PPN: %px\n\n", PN(buffer), PN(virt_to_phys(buffer)));
	
	kfree(buffer);
	return 0;
}

// allocation with vmalloc
static int alloc_vmalloc(int n){
	size_t buffer_size = n * PAGE_SIZE;
	int i;
	
	buffer = vmalloc(buffer_size);
	
	for(i=0; i<n; i++){
		struct page *page = vmalloc_to_page(buffer+i*PAGE_SIZE);
		unsigned long ppn = page_to_pfn(page);
		
		// translate virtual to physical page number with the macro
		pr_info("vmalloc - VPN: %px -> PPN: %px\n", PN(buffer + i * PAGE_SIZE), (void *)ppn);
	}
	
	vfree(buffer);
	return 0;
}

//user-space random access probe
#define NR_TRIES 30
#define C_SIZE 700

// generates a random address in low 24-bit range (simulate user-space pointer)
static void *random_us_ptr(void) {
  u64 rand_val;
  get_random_bytes(&rand_val, sizeof(rand_val));
  rand_val = rand_val & 0x00ffffff;
  
  return (void *)(uintptr_t)rand_val;
}

// prints info about the current memory layout (VMAs)
// try accessing random user-space address safely using 'copy_from_user()'
static void print_proc_info(void){
	// pointers to process memory descriptor 'mm_struct' and memory areas 'vm_area_struct'
	struct mm_struct *mm;
	struct vm_area_struct *vma, *vmaf;
	int i;
	char to[C_SIZE]; // local buffer in the kernel, where the user-space memory will be copied
	
	pr_info("Current process %s\n",  current->comm);
	mm = current->mm;
	vmaf = mm->mmap;
	
	// print each VMA with read flag info 
	for(vma=vmaf; vma; vma = vma -> vm_next){
		vmaf = vma;
		pr_info("VMA: 0x%lx - 0x%lx %c\n", vma->vm_start, vma->vm_end, vma->vm_flags & VM_READ ? 'R' : 'N');
	}
	
	
	// attempt to read 700bytes from 30 random user-space addresses
	for (i = 0; i < NR_TRIES; i++) {
		int res;
		const void *f = random_us_ptr(); // random address in user-space
		
		// safely copies memory from user-space pointer f to a kernel space buffer to
		// safely fails without crashing the kernel
		res = copy_from_user(to, f, C_SIZE); 
		
		pr_info("We survived accessing %px, read %d bytes\n", f, C_SIZE - res);
	}
}

// define a custom struct with a spinlock
// the object we want to allocate multiple times
struct my_struct {
  u64 field1;
  u64 field2;
  u8 field3;
  spinlock_t lock;
} my_struct;

// number of times we want to allocate 'my_struct'
#define NUM_OBJ 19

static int howmany = 0;

// define a constructor to initilize memory and lock
// it is a function the kenrel runs every time a new object is allocated from the cache
static void my_struct_constructor(void *addr) {
	struct my_struct *p = (struct my_struct *)addr;
	// clears the memory by memset to 0
	memset(p, 0, sizeof(struct my_struct));
	// initialize the spinlock
	spin_lock_init(&p->lock);
	// increments global counter
	howmany++;
	pr_info("my_struct_constructor: %d \n", howmany);
}


// setup the slab cache
struct kmem_cache *cc;
typedef struct my_struct *my_struct_p;
static my_struct_p store[NUM_OBJ];

// store pointers to allocated objects in cache
static void build_and_fill_kmem_cache(void) {
	int i;
	// create a slab cache with constructor
	cc = kmem_cache_create("my_struct", sizeof(struct my_struct), 0, SLAB_HWCACHE_ALIGN, my_struct_constructor); 
	
	// allocate initialized 'my_struct' instances
	for (i = 0; i < NUM_OBJ; i++) {
		store[i] = kmem_cache_alloc(cc, GFP_KERNEL);
		pr_info("kmem_cache_alloc: %d \n", i);
	}
}

/*
             +------------------+
             | Slab Cache (cc)  | ← kmem_cache_create(...)
             +------------------+
                    ↓
         ┌────────────┬────────────┬────────────┐
         |  my_struct |  my_struct |  my_struct |   ← Preallocated objects
         └────────────┴────────────┴────────────┘
             ↑             ↑             ↑
         store[0]      store[1]      store[2]   ← Pointers to allocated objects

A slab cache is used for fast and efficient REPEATED allocation of the same fixed-size objects. A constructor esures each object is properly initialized.
*/

static int __init my_module_init(void){
	pr_info("Module allocated\n");
	print_zones();
	pr_info("Kernel logical base VPN: %px", PN(PAGE_OFFSET));
	pr_info("Kernel virtual range (VPN - VPN): %px - %px", PN(VMALLOC_START), PN(VMALLOC_END));
	alloc_kmalloc(4);
	alloc_vmalloc(4);
	print_proc_info();
	build_and_fill_kmem_cache();
	
	return 0;
}

static void __exit my_module_exit(void){
	// release the object allocated in the slub cache
	int i;
	for (i = 0; i < NUM_OBJ; i++) {
		kmem_cache_free(cc, store[i]);
		pr_info("kmem_cache_free: %d \n", i);
	}
	pr_info("Module unallocated\n");
}


module_init(my_module_init);
module_exit(my_module_exit);
```
# Lab5: UART character device driver
The goal of the lab is to learn how to write a Linux character device driver.
```c
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <asm/io.h> // provides inb() and outb() functions
#include <linux/fs.h> // manages fs operations
#include <linux/interrupt.h> // manages hardware interrupts
#include <linux/ioport.h> // used to reserve access to specific IO port ranges with 'request_region()' and 'release_region()'

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CLR");
MODULE_DESCRIPTION("Implementation of a UART driver for Linux.");
// UART stands for Universal Asynchronous Receiver-Transmitter

// define the base IO port address and IRQ line for the UART
#define PORT_BASE 0x2F8 // 2nd serial port of QEMU
#define PORT_SIZE 8 // size of the UART's IO registers
#define PORT_IRQ 3

// these are UART register addresses used to send/receive data, enable interrupts and check line status
#define RBR PORT_BASE + 0  // Receiver Buffer Register (read)
#define THR PORT_BASE + 0  // Transmitter Holding Register (write)
#define IER PORT_BASE + 1  // Interrupt Enable Register
#define IIR PORT_BASE + 2  // Interrupt Identification Register (read)
#define FCR PORT_BASE + 2  // FIFO Control Register (write)
#define LSR PORT_BASE + 5  // Line Status Register

static int major; // major number: returned when registering the device in /dev
static spinlock_t txLock, rxLock; // spinlocks for mutual exclusion in TX/RX
static const int bufsize = 64; 
static char rxbuffer[64]; // holds received characters
static int putpos = 0;
static int getpos = 0;
static volatile int numchar = 0;
static wait_queue_head_t waiting; // wait queue for blocking threads waiting on read when no data is available

// UART interrupt handler: it runs when an interrupt occurs on PORT_IRQ = 3
static irqreturn_t serialaos_irq(int irq, void *arg) {
	char c;
	switch (inb(IIR) & 0xf) { // reads the Interrupt Identification Register (mask lower 4 bits)

		case 0x6:   // Receiver Line Status (RLS) -> ERROR
			inb(LSR); // read LSR to clear interrupt
			inb(RBR); // read RBR to discard char that caused error
			return IRQ_HANDLED;

		case 0x4:       // Receiver Data Available (RDA) -> a char is waiting
			c = inb(RBR); // read RBR to get character and clear interrupt
			// locked sequence to safely access shared buffer
			spin_lock(&rxLock);
			// put char into buffer which is circular
			if (numchar < bufsize) {
				rxbuffer[putpos] = c;
				if (++putpos >= bufsize){
					putpos = 0;
				}
				numchar++;
			}
			// wakeup thread if any
			wake_up(&waiting);
			spin_unlock(&rxLock);
			return IRQ_HANDLED;

		default:
			return IRQ_NONE; // case in which the interrupt was not for UART
	}
}

// write implementation
static ssize_t serialaos_write(struct file *f, const char __user *buf, size_t size, loff_t *o) {
	int i;
	char c;
	
	// Simplest possible implementation, poor performance: serial ports are slow
	// compared to the CPU, so using polling to send data one chracter at a time
	// is wasteful. 
	// What is missing here is to set up the DMA to send the entire buffer 
	// in hardware and give us a DMA end of transfer interrupt
	// when the job is done. We are omitting DMA for simplicity.
	
	spin_lock(&txLock);

	// copy one bit at a time
	for (i = 0; i < size; i++) {
		if (copy_from_user(&c, buf + i, 1) != 0) {
			// if 'copy_from_user' fails, release the lock and return error (-1)
			spin_unlock(&txLock); 
			return -1;
		}

		// poll till bit 5 of LSR is 5
		// which means that Transmit Holding Register (THR) is empty
		while ((inb(LSR) & (1 << 5)) == 0); 
		// sends the character by writng it to the THR
		outb(c, THR);
	}
	spin_unlock(&txLock);
	return size;
}

// read implementation
static ssize_t serialaos_read(struct file *f, char __user *buf, size_t size, loff_t *o) {
	char c;
	int result;
	if (size < 1)
	return 0;

	// Simplest possible implementation, poor performance: this time we DO block
	// waiting for data instead of polling but we return after having read only
	// one character.
	// We should try to fill as many bytes of the buffer as possible, BUT also
	// return prematurely if no more chracter arrive. The piece that is missing
	// here is using the peripheral idle line detection, omitted for simplicity.

	spin_lock_irq(&rxLock);
	// sleeps until data is available or until we get interrupted by a signal
	result = wait_event_interruptible_lock_irq(waiting, numchar > 0, rxLock);
	
	// if result<0 we got interrupted by a signal
	if (result < 0){
		// release the lock and return
		spin_unlock_irq(&rxLock);
		return result;
	}

	// otherwise, we read one char from the buffer
	// update 'getpos' (circular buffer) and 'numchar'
	c = rxbuffer[getpos];
	if (++getpos >= bufsize){
		getpos = 0;
	}
	numchar--;
	
	spin_unlock_irq(&rxLock);

	if (copy_to_user(buf, &c, 1) != 0){
		// if 'copy_to_user' fails, we return -1
		return -1;
	}
	
	// we read one character
	return 1;
}

// specify how user-space applications can interact with this device
static const struct file_operations serialaos_fops = {
	.owner = THIS_MODULE,
	.write = serialaos_write,
	.read = serialaos_read,
};

static int __init my_init_module(void) {
	int result;

	// initialize waitqueues and spinlocks
	init_waitqueue_head(&waiting);
	spin_lock_init(&txLock);
	spin_lock_init(&rxLock);

	// reserve IO port space, if busy return error
	if (!request_region(PORT_BASE, PORT_SIZE, "serialaos")) {
		// cat /proc/ioports should find who's occupying our port
		pr_info("serialaos: can't access 0x%x\n", PORT_BASE);
		return -1;
	}

	// request IRQ line and assign 'serialaos_irq' handler
	result = request_irq(PORT_IRQ, serialaos_irq, IRQF_SHARED, "serialaos", THIS_MODULE);
	if (result < 0) {
		release_region(PORT_BASE, PORT_SIZE);
		pr_info("serialaos: can't claim IRQ %d: %d\n", PORT_IRQ, result);
		return result;
	}

	// set up serialaos
	outb(0x0, FCR); // disable hardware FIFO
	outb(0x5, IER); // enable RLS, RDA

	major = register_chrdev(0, "serialaos", &serialaos_fops);
	pr_info("serialaos: loaded\n");
	return 0;
}

static void __exit my_exit_module(void) {
	unregister_chrdev(major, "serialaos");
	release_region(PORT_BASE, PORT_SIZE); // free IO ports
	free_irq(PORT_IRQ, THIS_MODULE); // free IRQ
	pr_info("serialaos: unloaded\n");
}

module_init(my_init_module);
module_exit(my_exit_module);
```
Once the module is loaded, we have a new character device a `/dev/serialos`. 
It is possibile to send and receive characters via (respectively):
```zsh
echo hello > /dev/serialos
```
```zsh
cat /dev/serialos
```

- **Interrupt-driven input** (efficient): 
  `serialaos_read()` receive characters asynchronously and store them in the buffer. It reads only one character, and if there's no data, it blocks (sleeps) until at least one byte arrives.
  It is efficient since the CPU sleeps until data arrives.
  A better solution would read as many bytes as are available and return early if there is a pause in incoming data.
- **Polling-based output** (inefficient).
  `serialos_write()` sends one character at a time, using polling.
  It is inefficient since the CPU is sitting in a loop, checking repeatedly if the UART is ready to sent the next byte. 
  A better solution would be to use the DMA (Direct Memory Access): the DMA could transfer the whole buffer directly to the UART, getting an interrupt when the entire transmission is complete. 
# Workflow
1. Into modules, create a folder which MUST have the prefix `lab-` which will contains your module. 
   Note that this is the folder module. 
2. Create the module `module.c`. 
   Note that this is the name of the module (it can have arbitrary name).
3. Compile the modules by generating the kernel image with the command:
```zsh
make build sys
```
4. Run the kernel by starting QEMU:
```zsh
make run
```

```zsh
~aos/labs/course-labs-2425/stage/start-qemu.sh [OPTIONS]
```
where options can be:
- `--arch <arch>`: specify architecture (`amd64` or `aarch64`).
- `--smp <n>`: specify number of processors n (default 1).
- `--with-gui`: open QEMU gui.
- `--help`.
5. Now, once you are into `aos-mini-linux`, you can insert and remove modules with simple commands:
```zsh
insmod modules/<module folder>
rmmod <module name>
```

what is missing into the pdf: 
INIT_LIST_HEAD(&entry->list);
RCU-aware list manipulation