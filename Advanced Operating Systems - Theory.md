# Goal and techniques of OS
The goal of an OS are:
- **Resource Management**:
    - Ensures programs are created and run as if they had individual resource allocations.
    - Manages resources like CPU(s), memory, and disk.
    - Achieved through:
        - **Multiplexing the CPU**: increases CPU utilization and **reduces latency** by allowing running another program when one is blocked, using **preemption** and **context switches**.
        - **Process management**: manages process states, indicating what a process can or can't do and the resources it uses. 
        - **Scheduling** which is the decision of which process to run next. It should aim to balance several factors:
			- **Fairness**: ensure that no process is starving.
			- **Throughput**: aim for maximum process completion rate.
			- **Efficiency**: minimize the resources used by the scheduler itself and optimize CPU usage reducing context switching overhead.
			- **Priority**: relative importance or urgency of processes.
			- **Deadlines**: meet time constraints for time-sensitive operations like real-time tasks like multimedia playback or similar.
			Note that OS scheduling strategies are **balancing conflicting goals** like deadlines and fairness: there is NO universal scheduling policy. 
			- **General-Purpose OSs** (GPOS): balance throughput, fairness. They operates on best-effort basis, so there are no guarantees about fair resource allocation. High priority tasks usually get preference over low priority ones. They are designed to handle diverse workloads efficiently: sometimes deadlines might be missed.
			- **Real-Time OSs** (RTOS): weights more deadlines and priority. When a higher priority thread becomes available it is immediately given control. Rate Monotonic scheduling given higher priority to tasks with shorter periods, while Earliest Deadline First prioritizes tasks with the nearest deadline. 
- **Isolation and Protection**:
    - Regulates access rights to resources (e.g., memory) to prevent conflicts and unauthorized data access. This isolation is achieved through **memory virtualization**, where each program operates in its own virtual memory space but might share common portions in a protected way. 
      Virtual address space does NOT coincide with the actual physical memory usage: it does only contain the most recently used parts of it, while the remaining are stored in mass storage. 
      Paging is used to make physical memory appear abundant to processes. 
    - Enforces **data access rights** (e.g., file permissions).
    - Ensure **mutual exclusion** when needed thorough mechanism like locks and semaphores. 
- **Portability of Applications**:
    - Uses interface/implementation abstractions to hide HW complexity to applications (**facade pattern**, simpler interface to complex systems e.g., system calls). 
    - Ensures applications work on different systems with varying physical resources.
    - In Linux, this ensures that source code for user space application does not need updating when compiling for a newer kernel version. 
- **Extensibility**:
    - Creates uniform interfaces for lower layers, allowing reuse of upper layers (e.g., device drivers).
    - Hides complexity associated with different peripheral variants (**bridge pattern**, it avoid compile-time binding between an abstraction and its implementation, which is selected at run-time e.g., file system).
    - This allows programs to interact consistently with standard interface components, regardless of the underlying implementation. 

| Name                                                      | Target (Goal)                                                                                                                                                  |
| :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FIFO` - first in first out<br>`SJF` - shortest job first | Task turnaround time (difference between finishing time and arrival time of a task) for better system responsiveness                                           |
| `Round-robin`                                             | Minimizing response time by sharing CPU time among tasks of the same priority using time slices.                                                               |
| `CFS`                                                     | CPU fair share among all tasks.                                                                                                                                |
| `EEVDF` - Earlier Eligible Virtual Deadline First         | Earliest eligible deadline.                                                                                                                                    |
| `MLFQ` - Multi Level Feedback Queues                      | Optimize reponse time by allowing tasks to consume their entire time slice, improving fairness by boosting lower-priority task occasionally                    |
| `MQ-Deadline`                                             | Prioritizes read operations over write operations considering their deadline. It ensures that write operations are handled efficiently without causing delays. |

A **system call** is a way to ask for a privileged service: applications invoke kernel to do privileged operations (e.g., I/O operations). In Linux, each system call has a `syscall_number` (unique identifier) and `syscall_function()` (actual function that get executed).

Other patterns are commonly used in OSs:
- **Chain of responsibility**: pass request along chain of handlers, each handler decides either to process the request or pass it to the next one.
- **Command pattern**: turn a request into a standalone object that contains all information (e.g., I/O block request).
## Architectures of OSs 

The design of operating systems can encompass various architectural approaches, each with its unique characteristics:
- **No OS - Bare Metal Programming**:
	- Single purpose app that need high control on HW and respect high timing. 
    - Direct hardware manipulation without any OS layer (avoid overhead of processes and services).
    - Often used in very simple or highly specialized embedded systems which needs low power consumption. 
- **Monolithic**:
    - Single large kernel binary that incorporates all functionalities.
    - Device drivers and kernel code reside in the same memory area. Kernel source is very tightly coupled: faster performance due to the reduced layers involved.
    - They can become big and complex, which can affects security. They may be over general, leading to decisions that slow down performance. Coding is restricted to general-purpose libraries. 
    - Offer speed and simplicity at the cost of security and flexibility. 
- **Monolithic with modules**:
    - Minimal core kernel components.
    - External modules for additional services. Modules can be dynamically linked on demand, at run-time (without requiring system reboot). 
    - Easy extension of OS capabilities as new requirements emerge.
- **Microkernel**:
	- A single small kernel provides minimal process, memory management and communication via message passing.
	- All non essential components are implemented as processes in user-space. This imply easier maintenance and new software does not require rebooting.
	- A crash of a system process does not mean a crash of the entire system: stability and security are enhanced. 
-  **Hybrid**:
    - Combines micro-kernel design with additional in-kernel code to increase performance. 
    - Certain services (e.g., network stack, filesystem) run in kernel space while device drivers typically run in user space.
    - It allows on-demand capability without recompiling the whole kernel for new drives or sub-systems. 
    - The downside is that more interfaces increase the chance of bugs and security issues. 
- **Library OS** and **Unikernels**:
    - Services are provided via libraries compiled with the application and configuration code. 
    - An Unikernel is a specialized, single address space, machine image that can be deployed to cloud or embedded environments (RTOSes). There is no need for privilege transitions between user and kernel space, minimizing scheduler overhead, 

Each architecture offers different benefits and trade-offs, influencing the performance, stability, and complexity of the OS and the applications it supports. The choice of architecture depends on the specific requirements of the system it's designed to run on.

![[map.intro.jpg]]

# Processes
In Linux, a **task** refers to a single unit of computation while thread and processes are subcategories.
A task is made of a unique program counter, two stacks (one for user mode and one for kernel mode), a set of processor registers and an address space. 
 - **Threads**: grouped into **threads groups** which share a common user-mode address space but have separate stacks and program counters. Since they share address spaces, context switching between them is easier.
 - **Processes**: have isolated address space, providing memory protection. 

 The **Task Control Block** (TCB), also referred to as a **Process Control Block** (PCB) refer to the data structure that stores all the information about a task:
 - **Process Identifier (PID)**: unique identifier.
 - **Process State**: identified by macros:
	 - `TASK_RUNNING`: task either executing on a CPU or ready to run and waiting in the run-queue.
	 - `TASK_INTERRUPTIBLE`: task is sleeping but can be awakened by a signal (process is waiting on an event/resource).
	 - `TASK_UNINTERRUPTIBLE`: task is sleeping and won't wake up if it receives a signal. It is not killable.
	 - `TASK_DEAD`: task is not longer executed but remains in memory until its parent process retrieve its exit status and resources. It is referred as "zombie".
 - **Program Counter**: address of the next instruction to be executed.
 - **Virtual memory mappings**: memory allocated to the process, including pointer to structures like `mm_struct` and `fs_struct`.
 - Open files, including memory-mapped files.
 - Credentials (user and group IDs to enforce security).
 - Signal handling information (e.g. `preempt_count`). 

In the context of Linux, the PCB is implemented as the `task_struct`. Every task as an instance of `task_struct` which are organized into a task list. 
When a **context switch** occurs (i.e., the CPU switches from executing one thread to another), the kernel uses the information in the `task_struct` to save the state of the current thread and restore the state of the next thread to run.

![[task_state.png]]


A task enter the **waitqueue** with `wait_event()` and `wait_event_uninterruptible()`. 
Each waiting task is represented by a `wait_queue_entry` structure which contains details like flags, private field (connected to the `task_struct`), a callback function and a linked list entry. 
When the condition being waited on become true, `wake_up` function invoke the callback function changing the state of these tasks to running. 
The `WQ_FLAG_EXCLUSIVE` flag indicates that only one task (among those with the flag) should be woken up when the event occurs: this is needed to prevent the **Thundering Herd problem**, where all task woke up, only one reads the data while the rest go back to sleep.

`fork()` is a system call in Linux used to create a new process. It invokes `sys_clone()` which duplicates the current process, known as the **parent**, to create a **child** process. 
The new `task_struct` is a copy of the parent one, which only differs in the PID (which is unique), the PPID (parent PID) and certain resources (such as pending signals).
Rather than duplicate the address space, the parent and the child share a single copy using **Copy-On-Write** (COW) mechanism.

The `start_kernel()`, found in `init/main.c` is the entry point for the Linux kernel. It executes a series of early architecture specific setup routines (e.g., `setup_arch`, `mem_init`, `shed_init`, ..). 
Then `arch_call_rest_init()` is invoked, which transition to `rest_init()` which places the CPU into idle loop and creates a new kernel threads that run `kernel_init()` which is responsible for loading the initial RAM disk image using `initrd_load`.
Then `kernel_execve()` function executes `/bin/init` which is the **init task**, tagged with PID 1. It initializes long-running services, mounts hard drives, perform system cleanups, and more. The init process is special: if it exits or crashes, the system will panic or shut down. 

- **System V** init process stets tasks in motion sequentially during system startup, leading to longer startup times due to its single-threaded nature. It defines various levels (e.g., lvl1 for single-user mode, lvl3 for multi-user mode). The init task read from `/etc/inittab` action to take with their respective commands `id:rl:action:command`.
- **System D** provide a more efficient and parallelizable alternative to system V init process. It is based on units, plain text files that encode information about services, sockets, devices and mounts. The most common types of unit are `.services` files which encode details about processes that system D controls, and allow specifying dependencies to ensure everything start in the right order.  System D also offers **systemctl**, a command-line tool for querying and controlling the system D system and service manager. System D is a **declarative environment**.  
# Scheduling
The scheduler is responsible for determining the order in which tasks are executed. 
A **context switch** occurs when the CPU switches from executing one process or thread to another. This is a fundamental operation in multitasking systems, handled by the scheduler.
The `preempt_count` field in each task's `task_struct` tracks whether **preemption is currently allowed**.
- If `preempt_count > 0`, **preemption is disabled**.
- If `preempt_count == 0`, the task is **preemptible**.
The kernel **increments `preempt_count`** in critical sections (e.g., while holding a spinlock) to **avoid being preempted**, and decrements it afterward. Only when it reaches zero can a context switch occur.
The actual **low-level context switch** is performed by the architecture-specific macro or function `switch_to(prev, next)`.
It does:
1. **Saves the CPU context** of the current task (e.g., registers, stack pointer, ..).
2. **Loads the context** of the `next` task.
3. Returns control to the resumed task.

A scheduling class is an API that includes policy-specific code to update current task time statistics, pick the next task from queue, select the core on which the task must be enqueue and put the task to that queue. 
In Linux, individual policies are `SCHED_DEADLINE`, `SCHED_FIFO`, `SCHED_RR`, `SCHED_NORMAL`, `SCHED_BATCH` and `SCHED_IDLE`. 

| Scheduling Class   | Description                                                                    | Scheduling Algorithm              | Type of Target Processes |
| ------------------ | ------------------------------------------------------------------------------ | --------------------------------- | ------------------------ |
| **SCHED_DEADLINE** | Deadline-based                                                                 | (EDF) Earliest Deadline First     | Real-time                |
| **SCHED_FIFO**     | Soft real‑time processes, continue to run until higher priority task is ready. | First-In-First-Out                | Real-time                |
| **SCHED_RR**       | Share with timeslice                                                           | Round-Robin                       | Real-time                |
| **SCHED_NORMAL**   | Variable-priority                                                              | Completely Fair Scheduler (CFS)   | Not real-time            |
| **SCHED_BATCH**    | Low-priority                                                                   | CFS with idle task prioritization | Not real-time            |

Each process has a priority $\pi$.
- **Real-time processes**: $\pi \in[0,99]$. They belong to scheduling class `SCHED_FIFO` or `SCHED_RR` or `SCHED_DEADLINE`($\pi = 0$). The priority is known as `rt_priority`.
- **Non real-time processes**: $100 \leq \pi(v) \leq 139$ which depend on a **nice value** $v \in$ $[-20,+19]$ since $\pi(v)=120+v$. They belong to `SCHED_OTHER`, `SCHED_BATCH`($\pi \sim 139$) and `SCHED_IDLE`($\pi \sim 139$). The priority is known as `static_prio`.
![500](priority_value.png)

The `nice` value is only applicable to non-real-time processes, specifically those in the `SCHED_OTHER` (also known as `SCHED_NORMAL`) class. The `nice` values are used by the **Completely Fair Scheduler** (CFS) to adjust the share of CPU time that processes get, with lower `nice` values giving a process more priority, hence more CPU time.

It might seem counterintuitive, but within the Linux kernel's scheduling system, a **lower priority number** means a **higher priority for getting CPU time**. 

The central data structure of the core scheduler if the `runqueue`. Within the `runqueue` we have sub-queues like `cfs_rq`, `rt_rq` and `dl_rq` for different scheduling classes. 
Each CPU has its own `runqueue`, but it is possible to have multiple `runqueues` on a CPU by using task groups (**cgroups**).
Linux scheduler executes in a loop, iterating through scheduling classes: each class `pick_next_task()` method is called with the `runqueue` of the CPU it is running as its argument, In case no class yields a runnable task, the next pointer of the scheduling class is followed, stepping down to the next priority class.
## Completely Fair Scheduling (CFS)
CFS attempts to balance a process's virtual runtime with a simple rule: CFS **picks the process with the smallest `vruntime`**, which represents the time a task should run on the CPU.
CFS uses a red-black tree to find the task with the smallest `vruntime` efficiently.
If here're no runnable processes, CFS schedules the idle task. 

In Linux, the transition from the $O(1)$ scheduler to the CFS marked a significant evolution in process scheduling, emphasizing fairness and dynamic adaptability.
The $O(1)$ scheduler offered quick scheduling decisions but struggled with **fair CPU time distribution**, especially for long-running tasks. This was due to its reliance on fixed timeslices, which could lead to task starvation.

For each process $p$, its time-slice is computed as:
$$
\tau_{p}=f(\nu_{0},\ldots,\nu_{p},\ldots,\nu_{n-1},\bar{\tau},\mu)\sim max(\frac{\lambda_{p}\bar{\tau}}{\sum\lambda_{i}},\mu)
$$
where:
- $\lambda_p$ it the process **weight**, which depends on the nice value according to the exponential formula $\lambda_i=k\times b^{-\nu_i}$. The proportional weight is set against the total weight of all processes. `sched_prio_to_weights` table provide lambda values. 
- $\overline{\tau}$ is the schedule latency, a configurable parameter (default 6 ms) that represent the targeted time for each process to get a chance to run.
- $\mu$ is the **minimum granularity** (default 0.75 ms) to ensure a lower bound on the timeslice, preventing excessive preemption. 
The `sched_slice()` function can retrieve $\tau_p$.

CFS introduces the `vruntime` variable $\rho$ as an absolute measure of the dynamic priority of each process. It gives a way to compare process CPU usage when they have different priorities. Processes with lover $\rho$ are given priority. 
$\rho_p$ depends on the time $\epsilon_p$ the process has consumed, and it is inversely proportional to its weight $\lambda_p$.
$$
\forall p , \Delta \rho = \frac{\tau_p}{\lambda_p} = \frac{\overline{\tau}\lambda_p}{\lambda_p \sum \lambda_i} = \frac{\overline{\tau}}{\sum \lambda_i}
$$
$\Delta \rho_p$ at the end of the scheduling latency only depends on the total weight of all runnable process. Thus, any process can be fairly evaluated regardless of its individual priority. 

CFS ensure fair CPU time allocation for processes using the nice value, but it does not handle **latency requirements** well (some processes needs quick access to the CPU but do not require much time). Moreover, real-time processes are privileged.
## Enforcing additional fairness
### Cgroups
CFS alone is not enough to guarantee optimal CPU usage, especially when there are multiple threads from different user: for example, if user `A` with 2 threads and user `B` with $98$ threads, user `A` will only be given 2% of the CPU time, which is not ideal. Each user should be given an equal share of the CPU, which is then divided among their threads.
**Control groups** are a mechanism for guarantee fairness and optimal CPU usage when there are **multiple users**: it allocates CPU usage based on groups rather than individual threads. 
The idea is to treat both user as they were single task in a root runqueue. Each user group has its own local runqueue, where task are scheduled independently. 
This configuration allow fine-grained control over resources sharing and priority, improving overall system efficiency and responsiveness. 
### Load Balancing
When we have $n>1$ CPUs, **load balancing** is crucial: we cannot balance on the number of threads (otherwise high-priority threads will get the same share as low-priory one) but neither on the total load $\sum_i \lambda_{i, q}$ of each runqueue $q$ (if one queue holds a high-priority thread that often sleeps CPU will go idle often). 
The idea is to define $\gamma_{i, q}$ as the CPU usage of process $i$ on runqueue $q$ to balance on the total weighted load:
$$\Omega_p = \sum_i \lambda_{i, q} \times \gamma_{i,q}$$
In this way we account for both CPU usage and priority of threads.

**Load balancing** depends on the hierarchical layout of processor cores, caches and memory. In NUMA architecture, memory access cost differ based on the distance from the requesting node. For this reason, load balancers must ensure that threads and related memory access remains local to their domains within the hierarchy.
A **scheduling domain** collects processor units in groups that share certain HW properties. Each scheduling domain has a logic to determine when and where move a task to balance the load and a **designated core**, which performs load balancing (hierarchy is typically seen relative to this core): it periodically attempts to migrate tasks from overloaded CPUs or domains to underutilized ones.
## Deadline scheduler
It uses 3 parameters: runtime, period and deadline. Each task is given a specific runtime within every period: the runtime must be the worst-case scenario, commonly the period equals the deadline. The task must finish before its deadline. 

Constant bandwidth scheduling was added to ensure temporal isolation. If a task uses more time than needed, it is accelerated, preventing interference with other tasks. 

![[map.proc.jpg]]

# Linux Kernel Space Concurrency  
**Kernel concurrency** is a critical aspect of Linux kernel development, distinct from user space concurrency. It involves managing and synchronizing multiple threads within the **kernel space**. This includes:
- **Interrupts**: handling asynchronous events that can occur at almost any time.
- **Kernel Preemption**: allowing for the interruption of kernel tasks to enhance system responsiveness and multitasking capabilities.
- **Multiple Processors**: ensuring the kernel's ability to run on multi-processor systems, which introduces complexities in resource sharing and synchronization.

Regarding concurrency we can highlights:
- **Deadlock** between tasks: due to mutual exclusion, hold-and-wait, no preemption, and circular wait conditions.
- **Priority inversion** is a scheduling scenario where a high priority task is delayed by a lower priority task due to locking.

Interrupts are signals that cause the CPU to stops its current activities and execute a specific segment of code, disrupting the normal flow of execution. This asynchronous behavior can introduce concurrency challenges because interrupts handles tun independently of the main program flow. 

In a **preemptive kernel**, the kernel allows a process to be preempted while it is running in kernel mode: this means that the kernel can interrupt a process to give CPU time to another process.

This enables higher responsiveness as the system can switch tasks even when a process is executing in kernel mode. Widely used in RTOSes where response time is critical.

From Linux kernel version 2.6 onward, the kernel became optionally preemptive. **Preemption points** in the kernel include:
1. If a task in the kernel explicitly calls `schedule()` (**planned process switch**). It is always assumed that the code that explicitly calls `schedule()` knows it is safe to reschedule. 
2. When an interrupt handler exits, before returning to kernel-space, the kernel sets `TIF_NEED_RESCHED` flag in the current thread's descriptor to indicate that the scheduler needs to run (**forced process switch**). 

**Atomic context** refers to places in Linux source code where preeemption is not safe to do. Typical atomic contexts are when:
- the kernel is within an interrupt handler. 
- the kernel is holding a spinning lock, since an interrupt could lead to deadlocks or inconsistent state due to half-completed critical sections. 
- the kernel is modifying the per-CPU structures. 
- the kernel state cannot be restored completely with a context switch (e.g., when executing floating point instructions).

`preempt_count` variable is used to ensure a safe context switch that keeps track of the preemptions.
`preempt_count=0` when the process enters kernel mode, then it increase by 1 on lock acquisition (critical section) and on interrupts, then when lock get release or interrupt finished it get decreased by -1.  
As long as `preempt_count > 0` the kernel cannot switch.
To enable atomic context (i.e. disable preemption) we use `preempt_disable()` macro. To re-enable it, we use `preempt_enable()` macro. 
## The real-time patch - PREEMPT-RT
In the **PREEMPT-RT patched kernel**, critical sections and interrupt handlers become preemptible. This ensures that critical sections do not monopolize the CPU. 

**Threaded interrupt handlers** are kernel thread scheduled with `SCHED_FIFO`policy, which operates at a real-time priority level of 50. For real-time threads that must avoid interruption, set their `SCHED_FIFO` priority to a value higher than 50. 

The HW interrupt handler is designed to perform minimal tasks: it must activate the kernel thread responsible for processing the interrupt reducing the time the system spends in the interrupt context, thus improving response time. 

Synchronization mechanism in PREEMPT-RT context: 
- **Mutexes** & **semaphores**: used in process context for mutual exclusion.
- **Spinlocks**: used in interrupt handlers and atomic context to protect shared kernel data structures without sleeping. 
- **Read-write locks**: used when we have frequent read operations and fewer writes. 
- **Seqlocks**: used in situations with high read concurrency and infrequent writes, allowing readers to access data without blocking. 
# Linux Synchronization
## Locking
### Sleeping locks
Sleeping locks, also called semaphores, when a task operating inside the kernel tries to acquire the lock which is unavailable, the semaphore place the task on a waitqueue putting it to sleep.

Semaphores should only be picked up in process context, using `down()` or `down_interruptible()`, and can be released with `up()`.
### Spinning locks
On **Symmetric Multi-Processing** (SMP) machines the spinlock is the basic ingredient to ensure mutual exclusion between cores.
**Spinning locks** continuously poll the lock until it becomes available while a sleeping lock waits until it is notified that the lock is available (introducing more overhead).

- Useful when **lock for a short period of time**. It's wise to hold the spin locks for less than the duration of 2 context switches, or just try to hold the spin locks for as little time as possible.
- Used in interrupt handlers, whereas semaphores cannot be used because they sleep. (processes in kernel can't sleep!).

**IRQ variant** **disables interrupts locally**, not globally: it allows other processors to continue executing while ensuring exclusive access on the current processor. Disabling interrupts locally is sufficient since we only need to prevent the current CPU from being interrupts and potentially causing a deadlock if an interrupt handler tries to acquire the same spinlock. 

Spinlocks are often implemented exploiting machine-provided atomic instructions such as **atomic compare-and-swap** (CAS) or atomic-swap.
The CAS function only performs the swap if the current value matches an expected "old" value: it guarantees that only one thread can update the variable at a time. 

The spinning lock is based on a busy-waiting approach, which can waste CPU cycles if contention is high, leading to inefficiencies. 
Within spinlock preemption is disabled. 

Variants of spinning locks are: 
- **Readwrite locks**: distinguish between readers and writers, where multiple readers can access an object simultaneously, whereas only one writer is allowed at a time. It helps prevent race conditions and ensures data integrity. 
  The drawback is that many readers and few writers, writes can suffer **starvation**.
- **Seqlocks**: to prevent starvation of writers, a counter starting from 0 is used to track the number of writers holding the lock. Each writer increments the counter both at locking/unlocking phase. The counter permits to determine if any writes are currently in progress: if the counter is **even** **no writes** are taking place, if the counter is **odd** a **write** is taking place. 
  Similarly readers check the counter when trying to lock: if the counter is odd, it means busy wait. If even the reader does the work but before releasing, it checks if the counter changed (in case it reads again). 
  `jiffies` is the variable that stores a Linux machine's uptime, is frequently read but written rarely by the timer interrupt handler: a seqlock is used for machines that do not have atomic 64 bit read.

### Cache aware spinlocks - MCS locks
In **multi-processor context**, caching may introduce unwanted overhead in managing the lock causing the **cache ping-pong problem**. 
The problem works as follow:
- CPU1 gets the lock and it bring the cache block in its own cache.
- CPU2 tries to grab the lock, invalidating the cache line of CPU1, which means additional overhead in keeping the cache coherent. CPU2 cannot progress as the lock is taken.
- CPU3 tries to grab the lock, invalidating the cache line of CPU2,
- CPU2 re-spins trying to acquire again the lock invalidating CPU3 cache line, and so on. 

To solve this continuous invalidation problem, Mellor-Crummey and Scott proposed the **MCS lock** which solve the cache ping-pong problem by using two mechanisms:
- **Queue-based locking**: it maintain a queue of waiting threads. Each thread in the queue spins on a **local** variable in the **local cache**, reducing the need for frequent memory access, which is common in traditional spinlocks.
- **Localized spinning**: a thread waits for a signal from its predecessor in the queue. This means that **it only needs to monitor a local variable**, which is likely to be in its cache line. 

We have two primary structures:
- `qspinlock` structure which contains a boolean field `taken` and a pointer to another structure `msc_lock` called last.
- `msc_lock` structure for each CPU, which has a boolean field `locked` and a pointer to another `msc_lock` called next.
When CPU1 acquire the lock, it writes down the address of its own `mcs_lock` in the last field of `qspinlock` and set `taken = 1`.
CPU2, seeing that the lock is taken, put itself in line by writing the next pointer of the current last element (which is CPU1) towards its own `mcs_lock`, stores the address of its own `mcs_lock` in the last pointer of `qspinlock` and set its own `locked = 1` (it spins on this bit). 
When CPU1 finishes, it tries to put a null value in `qspinlock` last field, expecting to find its own structure but find the one of CPU2. So, it set CPU2 `locked = 0`. 

In Linux, queue spinlocks are more complex since the use of queues is minimized. 
CPU1 will grab the lock without even touching the queue. 
When CPU2 arrives, it turns `qspinlock` into a pending state using an additional bit `pending`, and will spin over the original lock value (instead of adding itself to the queue). 
Only when CPU3 arrives it will add itself to the queue. 
### Kernel Deadlocks
Linux kernel is impacted by the problem of deadlocks. 
If configured with `CONFIG_PROVE_LOCKING=y` the kernel uses a run-time mechanism for checking deadlocks. This mechanism is **lockdep**: it detects violations of locking rules keeping track of locking sequences through a graph and look for spinlocks acquired in interrupt handlers or in process context when interrupts are disabled.

Deadlock can be generated by the same class of locks
Lockdep keeps track of the state of the lock classes by tracing dependencies (which refers to the order in which the lock is acquired).

In general, we must ensure that the same lock is always acquired with the same interrupt state.
If a lock is taken during process context and an interrupt arrives and takes the same lock, it will result is a lockup (i.e. the interrupt will never finish): to solve this problem we should disable irqs during the first acquisition using `spin_lock_irqsave()` and `spin_lock_irqrestore()`.
## Lock-free
Lock-free algorithms allow synchronization and concurrent access to shared resources without the need for locks, which can cause contention and processing delays. 
### per-CPU variables
The most simple way is to reduce shared data with **per-CPU variables**: they consist of arrays where each element correspond to a CPU in the system, each CPU can access and modify its own element independently.
However, to prevent CPU switches during variable manipulation kernel preemption must be disabled when accessing per-CPU variables.
In this way, we localize data, minimizing conflicts that arises when multiple processors attempt concurrent modifications. 
### Atomics
In Linux kernel programming, atomic operation are crucial for ensuring thread safety without the overhead of locks. 
An atomic operation is executed entirely as a single, **indivisible step**, meaning that it will be completed in its entirety, or not at all.

 **Compare-and-Swap** (CAS) is a powerful synchronization primitive used in multi-threaded programming. 
 It is a single atomic operation that compares the contents of a memory location to a given value and, only if they are the same, modifies the contents of that memory location to a new given value. This atomicity is crucial to ensure that no other thread has altered the data between the steps of reading and writing. CAS is widely used in the implementation of lock-free data structures and algorithms. 

The **ABA problem** is a classical issue encountered in concurrent programming when using CAS operations. It occurs when a memory location is read (A), changed to another value (B), and then changed back to the original value (A) before a CAS operation checks it. The CAS operation sees that the value at the memory location hasn’t changed (still A) and proceeds, unaware that it was modified in between. This can lead to incorrect behaviour in some algorithms, as the change-and-reversion can have side effects or meanings that the CAS operation fails to recognize.
### Read Copy Update (RCU)
RCU is a mechanism that allows for lock-free read-side access to shared data while ensuring consistency with simultaneous write. 

Readers access data without any locks, which is particularly advantageous in high-concurrency environments due to minimal overhead. To do so, we can use `rcu_read_lock()`which stop preemption from happening. 
Writers update data by creating a new version and switching pointers atomically. 

The old data structure is not immediately freed: it is marked for **later reclamation** once all pre-existing readers that might be accessing it have finished. It is a **deferred reclamation** system, since memory can be freed safely only after all readers have released the references to it. 

The period between data removal and memory reclaim is called "**grace period**": it must be long enough that any readers have dropped their references i.e. they have entered **quiescent state** by calling `rcu_read_unlock()`. 
Since read section cannot be preempted (it is considered an atomic context), a context switch means quiescent state, thus the kernel can infer that a grace period elapsed when all other CPUs have executed a context switch. 
After a writer is done, we call `syncronize_rcu()` to put the writer in a wait state for the grace period to be elapsed. 

In Linux kernel, RCU is widely used (e.g., managing network routes, file system mounts) where the pattern of **frequent reads** and **infrequent writes** is common.
## Memory consistency models
Memory models define how memory operations in one thread are seen by others in a multiprocessor environment. Different models have varying levels of order enforcement and visibility, influencing how multi-threaded programs behave and are synchronized. 

The order in which memory accesses are seen by another thread might be different from the order in the issuing thread, due to **write buffering**, **speculation** and **cache coherencey**. 
Each model may have difference possible execution traces. A weaker model has more potential traces. 
## HW models
### Sequential consistency

> [!THEOREM] Sequential consistency
> A multiprocessor is called sequentially consistent if and only if for all pairs of instructions $(I_{p, i}, I_{p,j})$ you have that $I_{p, i} <_p I_{p,j} \implies I_{p, i} <_m I_{p, j}$.
> Where $<_p$ is the program order of the instructions in a single thread and $<_m$ is the order in which these are visible in the shared memory (also referred to as the **happens-before** relation).
> In practice, for every pair of instructions, the program order implies the same order in shared memory.  

Theoretically, it is the strongest model, but it is impractical. 
### Total Store Order (TSO)
Used in x86 intel processors, it uses a local **write queue** (also called **store buffer**, using FIFO, one each processor) to **hide memory latency** by temporarily storing writes. 
When a processor issue a read, it checks the most recent buffered write to that address, if there is none it access the shared memory. A processor's read cannot access other processors write queues, it will always view its own writes before others.
Once a write operation is completed and the data reaches the shared memory, it becomes visible to all processors. 

The system has instructions to flush the store buffer, and a set of atomic instructions implemented through the use of a global HW lock L (that automatically flush the buffer). 
### Partial Store Order (PSO)
Used in ARM processors. It is weaker than TSO. 
We can assume that each processor reads from and writes to its own complete copy of memory. Each write propagates to the other processors independently, with reordering allowed. 
This means that **a store can bypass another store**: stores can occur out of order.
However, writes to the same address are not allowed to bypass each other.

> [!TIP] Litmus test
>  Programs that tell us whether one system is either complying with one model. 

Since store can occur out-of-order, this complicates lock-free programming since the program behavior depends on thread timing. 
## Taming the HW
A **data-race** between threads is produced by at least one write and zero or more reads, which may produce inconsistent behavior. 

A **synchronization model** is a set of rules for how HW coordinates r/w to shared data. 

A **data-race-free** (DRF) **synchronization model** should provide memory operations to ensure happens-before relationship for memory r/w across different threads. Under this model, ordinary r/w can be reordered within threads, but cannot cross memory synchronization operations. 

A **data-race-free program** is such that two ordinary memory accesses to the same location from different threads are either both reads or else separated by synchronization operations forcing one to happen before the other. 

If the HW supports DRF synchronization model, a DRF program will appear as it is executed on a sequential consistent machine. 

**Fences** (**barriers**) are trivial synchronization instructions that allow to obtain DRF programs. They enforce program order $<_p$ into memory order $<_m$. It restricts processors and compilers from employing necessary optimizations, resulting in degraded performance. They enforce order both above and below. 

- **Performance consideration**: overuse of fences can lead to performance degradation as they restrict the processor's ability to reorder instructions for efficient execution.
- **Strategic placement**: it's crucial to strategically place fences where necessary, rather than using them indiscriminately, to balance correctness and performance.

The C compiler can introduce additional reordering of instruction that might appear as if the machine had a weaker memory model. This adds another layer of complexity to HW memory models. 
The primitives `READ_ONCE` and `WRITE_ONCE` prevent the compiler from reordering reads (or writes), omitting reads (for known values, i.e. fusing) or doing too many reads (when register spilling is needed, i.e. splitting). Without these primitives, compilers can reorder them, omit reads if value appear unchanged or insert excessive reads due to register spills. 

The Linux memory model is essentially PSO. 
It uses a bi-directional synchronization mechanism which establish a happens-before relationships across threads: 
`smp_load_acquire()`: ensures that all preceding reads and writes by the releasing threads become visible to other threads, so before it writes data to memory, all its prior operations are pushed to visibility for others. 
`smp_store_release()`: ensures that read operation sees the latest value before any other operation occur subsequently.


![[map.conc.jpg]]

# Virtual Address Space
Virtual memory creates the perception of a larger memory space, even when physical memory is limited: a computer can overcome memory shortages by temporarily moving data from RAM to disk storage. 
![[kernel_virtual_memory.png]]

## Process address space
Process Address Space is the range of addressable memory for each process which includes text (code), data, heap, and stack segments.
![[process_addr_space.png]]
**Virtual Memory Areas** (VMA) defines the range of virtual addresses utilized by a process thus they help determine whether a virtual page number is valid even if it's not currently mapped. They facilitate efficient data retrieval for pages not currently mapped. They maintain information about the corresponding data source, whether on disk or in existing memory pages. 
They ensure **secure access** through permission flag: readable, writable, executable.

VMAs can be **anonymous**, if they are not linked to any file (e.g., heap, stack, bss, ..), or **file-backed**. Anonymous areas initially map to a zero page and employ **Copy-On-Write** (COW) if written. The stack is unique with its `VMA_GROWSDOWN` flag. 
VMA can be created explicitly using `mmap()` with the backing store file descriptor and protection flag.

`smem` list various processes physical memory usage with their Unique Set Size (USS, indicates memory solely used by a process, not share), Proportional Set Size (PSS, indicates shared memory) and Resident Set Size (RSS, entire memory set in physical RAM).

**Paging** is a memory management scheme that eliminates the need for contiguous allocation of physical memory. It allows the physical address space of a process to be non-contiguous.
A **page fault** occurs when the CPU accesses a virtual address whose corresponding **Page Table Entry** (PTE) is invalid or does not allow the attempted access according to the access right of the page table. 
These rights are, in principle, derived from the access rights specified in the VMAs. However, translation of VMA access right to PTE's ones is not trivial due to:
- **Demand paging**: pages are loaded into memory only when they are accessed, not all at once.
- **COW mechanism**: optimizes memory usage by sharing pages between processes until a write occur.

The motivation behind those two mechanism is that processes do not address all the addresses in their address space. 
Segmentation faults occur when the kernel cannot handle a page fault, usually due to an invalid memory access (when address lacks proper mapping or permission).

Linux uses red-black tree structure to manage VMAs. VMAs are not overlapping. `find_vma()` navigates this structure using the `vm_end` values. 

- When a process tries to access an address that falls within a VMA but lacks a PTE, the entry is configured by `handle_mm_fault()`.
- This function delegates the work to lower-level functions like `handle_pte_fault()`, which checks the type of fault and decides how to handle it.
	- If the fault is due to a read or write to an anonymous page, it calls `do_anonymous_page()`, which allocates a new physical page and maps it into the process's page table.
	- If the fault is due to a write and the memory is writable according to the VMA but the page is marked as read-only, it calls `do_wp_page()` to create a copy of the page for the process. 
## Kernel address space
**Kernel Logical Addresses** are a subset of kernel addresses that are directly mapped to physical contiguous memory. They are used for memory that is frequently accessed or needs to be accessed quickly, such as memory used by DMA. 
Allocation is made using :
- `kmalloc` for small chucks of memory.
- `get_free_pages` for larger chunks: it may fail due to lack of available memory so we must ensure that deallocations are symmetric with allocations, over-allocation can degrade system responsiveness increasing the risk of DoS.
- `alloc_pages` when precise control is needed (e.g. on Non Uniform Memory Access (NUMA) nodes)

**Kernel Virtual Addresses** are not directly mapped to physical memory. They are helpful in situations where the kernel needs to allocate large buffers but can't find a continuous block of physical memory. Allocation is made using `vmalloc`.

In the kernel, fixed-size data structures are frequently allocated and released: when small objects are allocated from full pages, **internal fragmentation** becomes a significant concern.
The **buddy allocator** aims to reduce the need for splitting large free memory blocks when smaller requests are made. 
The buddy algorithm works as follow:
- It works by dividing memory into blocks of various sizes, which are powers of 2. When a request for memory is made, the buddy allocator finds the smallest block that will satisfy the request. If a block is larger than needed, it's split into "**buddies**."
- **Allocation**: the kernel checks the free lists starting from the requested block size up to larger sizes. If a free block of the requested size is available, it is allocated immediately. Otherwise, a larger free block is split into two equal “buddy” blocks; one half is allocated to satisfy the request, and the other half is returned to the appropriate free list.
- **Deallocation**: when a block is freed, the kernel checks if its adjacent “buddy” block of the same size is also free. If so, the two buddy blocks are merged into a single larger block. This merging process may continue recursively to coalesce bigger blocks, reducing fragmentation.

Allocating and freeing data structures is one of the most common operations inside any kernel. The buddy allocator, while efficient for larger allocations, is not ideal for these smaller structures due to potential internal fragmentation and the need for synchronization via locks, causing potential bottlenecks under heavy load. 
To address these limitations, Linux employs additional mechanisms:
- **Quicklist**: avoids unnecessary overhead in managing paging-related allocations caching frequently used pages. 
- **Slab allocator**: more efficient for fixed-size allocation patterns, reducing fragmentation and minimizing lock contention. 
  It pre-allocates memory in chunks (**slabs**). When an object of that type is needed, it can be quickly allocated from a pre-existing slab, reducing the overhead of frequent allocations. 
  It provides two main classes of caches:
		- **Dedicated**: for commonly used objects (e.g., `mm_struct`, `vm_area_struct` , ..). 
		- **Generic** (size-N and size-N(DMA)): general purpose caches, which in most cases are of sizes corresponding to powers of 2.
  It is not a replacement for the buddy allocator, but a higher-level memory manager that works on top of it: it uses the buddy allocator to get larger chunk or memory which are then subdivided into slabs. 
# Physical Address Space
In kernel mode, memory allocation is managed through either a slab object cache (e.g., kmalloc) or by directly interfacing with the buddy allocator.
User space request to the buddy are triggered by file operations, memory mapping or stack growth. 
Caches stores frequently accessed data in memory. The cached data is written back to disk either to a regular file or to swap space when free memory is low. 
## Zonal page allocation
Physical memory can be organized into **NUMA** banks or nodes.
Each node feature eight cores with a shared cache and memory bank slots. Some cores also share floating point unit. 
NUMA key characteristic is non-uniform memory access times, since time depends on the processor's location relative to the memory: NOT all processors can equally access all part of the memory.
To accessing data, a processor first checks its local cache. If the data is not there or in its local memory, it must reach out to remote memory in another node. 

![](NUMA.png)
**Zonal Page Allocation** is a memory management strategy which aims to allocate memory physically near the CPU requesting it. 
To support this, physical memory is divided into **zones**, which represent different ranges of physical addresses with distinct properties. They include the total number of pages and free lists organized by page order (used by the buddy allocator).
## User space page caching
Page cache is a set of physical page descriptors corresponding to pages that contain data read and written from regular FS files or associated with anonymous VMAs.
It is accessible through two maps:
- **Forward mapping (`file descriptor + offset -> struct page`):** enables direct access to the physical page containing a file's data at a specified offset. It is generally used for file-backed memory mappings.
  Anonymous pages are normally not forward-mappable. However, when they are **swapped out**, they are temporarily associated with the `swapper_space` pseudo-filesystem, allowing them to be managed within the swap cache and later swapped back in.
- **Backward (inverse/reverse) mapping (`struct page -> [VMA]`):** enables the kernel to determine which VMAs and virtual addresses are using a given physical page. This is crucial for operations like **page invalidation**, especially for **shared pages** (e.g., CoW, file-backed page of files of shared pages), where multiple processes may have mapped the same page. 
  The `struct page` describes **physical page** (where actual data is) attributes in memory, containing essential details like mappings, counters, and flags.

```c
struct page {
    atomic_t _mapcount;                // number of user space mapping
    struct address_space *mapping;     // pointer to the address space
    pgoff_t index;                     // offset within the address space
    
    unsigned long flags;               // status flags for the page
    atomic_t _refcount;                   
    // reference count: freshly allocated pages have count of 1, when count reach 0 it will be freed
    struct list_head lru;              // LRU list linkage
};
```

To manage memory pressure, Linux employs a **Page Frame Reclaim Algorithm** (PFRA) which organizes pages based on their activity into 'active' and 'inactive' lists within each memory zone.
PFRA is triggered under certain conditions based on a **zone-based watermark** model, based on the number of available free pages:
- High: when free pages fall below this level, `kswapd` periodically runs the PFRA.
- Low: when free pages hit this level, it triggers a deferred `kswapd` invocation.
- Min: at this critical level, the buddy allocator itself will invoke the PFRA to free up pages.

![](kswap.png)

PFRA is based on the **Corbato's clock** algorithm. It keeps a circular list of pages in memory.
- Each page has a **reference bit (R)**, which is set to 1 when the page is accessed.
- The algorithm uses a **"clock hand"** that scans the list of pages:
    - If the **R bit is 1**, the page has been recently used so the algorithm **clears the bit to 0** and moves the hand forward.
    - If the **R bit is 0**, the page is considered **not recently used**, and it is **evicted** (freed or swapped out).
This approach approximates **Least Recently Used** (LRU) behavior without the overhead of tracking exact access times. 

The actual Linux implementation is more complex, since it uses a two-list variant of the Clock algorithm.
Pages are organized in 2 circular lists:
- **Inactive** list: pages initially reside here and move to the 'active' list after two accesses.
- **Active** list: contains pages that have been referenced multiple times.
The kernel periodically scans the inactive list using a clock-like algorithm to identify pages for eviction based on their reference bits. 
It also periodically moves pages from the active list back to the inactive list to maintain a balance between the two, ensuring that the inactive list contains enough candidates for reclamation.

![[PFRA.png]]

The inactive list size determines the number of pages that can be held without being promoted to the active list. The ideal size is equal to the **working set** $w(t, \tau)$, which is the set of pages reference in the time interval $(t-\tau, t)$. 
If the inactive list is too small, pages may not get promoted, leading to **thrashing** (page repeatedly enter and exit the inactive list). This issue is solved by dynamically adjusting the size of the inactive list by reallocating space from the active list based on the concept of **refault distance**, which is the number of slots that would have been needed to keep that page in the inactive list.
The refault distance is calculated as the difference between activation/evictions counter during the page's eviction and its return. 
![[map.vm.jpg]]

# I/O
A **bus** is an hardware connection channel that facilitates communication between the CPU and other devices. Communication between the CPU and devices can happen over two buses:
- **Memory bus:** peripheral device registers are exposed into the CPU's address space. This setup allows the CPU to read, write and send commands to the peripherals using standard memory access instructions. It simplifies the interaction by treating peripheral I/O as regular memory reads and writes. It is called **memory-mapped I/O**. 
- **Port bus:** typically has its own address space separate from the main memory, necessitating specialized commands for data transfer and control. It is called **port-based I/O**.
## CPU to device communication
### Port-based I/O instructions
Each device is allocated a port number within the I/O address space.

UARTs are HW components that handles asynchronous serial communication. 
It takes bytes of data and broadcast the individual bits sequentially. It frames data between start and stop bits, ensuring that the timing is manages by the communication channel itself.
- When transmitting, it writes into the **Transmit Holding Buffer** (THB) register. 
- When receiving, it access the **Receiver Buffer Register** (RBR).
### Memory-mapped I/O instruction
I/O operations are done through loads and stores to **memory mapped registers**, since HW makes device register available as they where memory locations. 
## Device to CPU communication
- **Polling**: CPU constantly checks the status of devices. While simple, it is inefficient as it consumes significant CPU resources. It is inexpensive if the device is fast.
- **HW interrupts**: the OS issue a request to the device, putting the calling process to sleep and context switch to another task, when the device has finished it raise a hardware interrupt causing the CPU to jump at a predetermined **Interrupt Service Routine** (ISR) that is an interrupt handler. It is useful for slow devices. In fast devices can cause a livelock. 
- **Direct Memory Access** (DMA): involves using a DMA controller that independently manages data transfers between different devices and memory. It does so without needing the CPU's involvement, thereby **decoupling** it from the transfer process.
# Low-level I/O
Before reading and writing data to peripheral registers, Linux drivers needs to request access to the devices: `request_region` to access I/O port and `request_mem_region` to do memory-mapped I/O.
Port-based I/O is accessed through special instruction like `INB` and `OUTB`.
In memory-mapped I/O the address of peripherals must be mapped in the virtual address space in a uncacheable region using `IOREMAP`.
We can abstract by using an `iportmap` call and then using generic `ioread` and `iowrite` functions.
## Interrupts management
Each interrupt is associated with a specific number (vector). The **Interrupt Descriptor Table** (IDT) in Linux contains handlers (ISR) for all vectors.  
ISR is a low-level architecture specific entry point that saves the CPU state and then calls the generic `do_IRQ()` routine.
The routine identifies the source of interrupt and dispatches the appropriate device-specific interrupt handler.

In Linux, the concept of "**deferring work**" involves postponing the execution of a task until a later time. The idea is to move the non-critical management of interrupts to a later time (improve responsiveness and benefit from aggregation).
Interrupts are spitted into:
- Top half: minimal mandatory work, which works in a non-interruptible context and schedules the deferred work.
- Bottom half: finalize the work specified **reconciliation point**. 

> [!TIP] Reetrancy
> Reentrancy is a property of code that allows it to be **safely called again** before its previous execution is complete. 
> - Reetrant code does not rely on shared data, it uses local variables of enrues exclusive access to shared resources.
> - Non-reetrant code uses static or global variable without proper synchronization and relies on a state that may be altered.

Three methods are available for deferring work.
### SoftIRQs
They are **statically allocated** at compile time, they still work in kernel context (not in the limited hard interrupt context) thus **cannot sleep**. 
It is difficult to program them directly, mainly used only by networking and block devices directly in their interrupt handlers. 
Same type of SoftIRQs can run simultaneously on several processors and for this reason the code must be reentrant.

Once the last interrupt handler completes, the system invokes `do_softirq()`which handles any deferred work. 
### Tasklets
 They offer an easier interface: they are just a type of SoftIRQ with an interface that allow to create them **dynamically**. 
 Kernel ensures that NO more than one is running, so they ensure non-reetrancy. **One-shot** deferral scheme. 

A tasklet is a pointer to a function and some additional data. It is represented by a list of  `tasklet_struct`.
During reconciliation points, the kernel cheeks this list for any scheduled tasklets that are not currently running on another CPU, and switches them to a running state. 

Since tasklets **cannot sleep** (as SoftIRQs), they must manage locking data with spinlocks when accessing shared data. 
### Workqueues
It is a schedulable entity that runs in process context (so they can sleep). It is a general mechanism to submit work to a **worker kernel thread**.
They accommodate deferred tasks that may need to **sleep** or **wait** for resources, something that SoftIRQs and tasklets aren't capable of. 
# Linux Device Management
Linux categorizes devices based on their type and function into:

|               | Chatacter device                                                                                                                                                                                                  | Block device                                                                                                                           | Network device                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Functionality | Operate with a stream of characters, accessed one at a time, so they do **not use buffering**.<br>Interaction with these devices is immediate, making them ideal for hardware that requires prompt data transfer. | Organized in blocks for **random access**, employing **buffering** and **caching**. <br>Suitable for large data storage and retrieval. | Handle data packet transmission and reception over network interfaces, critical for network communication. |
| Access        | Via special files in `/dev/`.                                                                                                                                                                                     | Through special files in `/dev/`.<br>Any block can be access regardless of its location on the device.                                 | Managed through network configuration tools, not directly via `/dev/`.                                     |
| Examples      | Serial ports, keyboards, terminal devices.                                                                                                                                                                        | Hard drives, SSDs, USB drives.                                                                                                         | Ethernet adapters, wireless interfaces.                                                                    |

Linux integrates devices into the file system as special files: each device is assigned a path name, typically located in the `/dev` directory. 

Drivers bridge the gap between kernels and user space. 
When we read from a device node, the kernel moves the data stream collected by the driver into the application memory space. 
When we write to a device, the kernel takes the application data stream and places it into the driver's buffer.

Each driver is identified by two numbers: the **major** is used to identify the driver associated with the device, while the **minor** is used to identify multiple devices of the same type managed by a single driver.

- Initially, in Linux, devices (nodes) were created manually, but managing them was tricky. 
- `devfs` was created, a **Virtual File System** (VFS) managed by the kernel. It exposes information about devices and drivers, as well as the relationships and hierarchy among different components, to user space. It was an attempt to manage device nodes dynamically, however it lacked persistence between reboots and difficulties in changing permissions or ownership.
- `udev` is a user-space daemon handles the creation and removal of device nodes, and manages device permissions and symlinks dynamically. It merged `devfs` dynamics with `sysfs`, a pseudo FS that exports information about various kernel sub-systems, HW devices and associated drivers to user space. 
## Low-level device management
### Char devices
Operations are specified when registering the driver with `file_operations`, which is a set of functions. 

When a device is registered through `register_chrdev` the kernel finds a major number (or it can be specified).
`cdev` structure is used to map a char device major to their corresponding char device structure: this mapping is used by the kernel to find the correct device driver upon request. 
### Block devices
They are **random-access**, so we do not have to scan them sequentially. 
The **mapping layer** identifies the **segments** of a page (parts that correspond to **contiguous sectors** on disk). Internally, the mapping layer operates on chunk of data called **blocks**.

The VFS handles all systems calls (open, read, write, close). It verifies if the requested data is already in **page cache**: if not it read it from disk and put into cache. To do so, a mapping layer is necessary to identify the physical block address corresponding to the logical block address requested. 

A block I/O operation is represented by a `bio` structure. Each `bio` describes the operation (read/write), starting sector, memory pages involved (via an array of `bio_vec`) and I/O specific flags. 
A `bio` contains multiple `bio_vec`, each one representing a single **segment**, which is a continuous area in  physical memory (typically part of a page) where data resides or will be placed. 
In this way, a `bio` which links multiple non-contiguous memory, allowing efficient handling of scatter I/O operations.

![](block_devices.png)

Once bios are created, they are converted into actual request (optimizing by merge where possible) that can be sent to the device driver. 

`request_queue` structure represents the queue of pending I/O requests for a block device. It helps in scheduling and optimizing these requests before they are dispatched to the actual device driver: techniques such as merging adjacent requests are employed to enhance performance.
`queue_rq` function convert a sequence of abstract block I/O into actual actions. 

The rate at which request are transferred into the actual HW queue is regulated by a mechanism called **plugging**: under low conditions is allow delaying operations thus giving additional time to merges. 

We can have multiple HW queue, enabling concurrent and out-of-order completition. 

![](block_IO_request.png)

**I/O schedulers** are in charge of decide how `bio` request must be submitted to the HW device. Common goals are oriented at improving performance and ensure efficient and fair use of the device:
- Minimize time wasted by hard disk seeks (e.g., using algorithm like elevator which sorts request based on their disk block number).
- Prioritize a certain process I/O requests.
- Give a share of the disk bandwidth to each running process (e.g., proportionally based on process priority level).
- Guarantee that certain request will be issued before a deadline. 
There are several schedulers.
- Noop: recommended for flash-based storage where there are no mechanical parts. 
- Budget Fair Queueing (BFQ): divides disk bandwidth equally among all running processes. Recommended for systems with slower storage devices, like mechanical disks, or in multi-user environment. 
- Kyber: low-overhead scheduler that prioritize read over write. It allows to indirectly configure the size of HW queues by specifying a latency target. Ideal for dynamically changing environments. 
- MQ-deadline: prioritize read operations over write, considering deadlines. 
## High-level device management (The Device Model)
The Linux device model is an abstraction layer which aims to **maximize code reuse** between different platforms in the Linux kernel.
This is achieved by the kernel providing a framework and a set of APIs that enable consistent and efficient management of devices. 
The goals are: 
- **Device representation** which gives an unified view of a clear structure
- **Driver binding**: automatic matching. 
- **Power management**.
- **Hot plugging**: ensure proper detection, configuration and clean-up.
- **Sysfs interface**: expose device and drivers to user space for easier interaction. 
- **Event notifications**: to user space. 

The core components (logically) of this device model are: 
- Devices: represented by a `struct device`.
- Drivers: represented by a `struct device_driver`.
- Buses: represented by a `struct bus_type` which contains method for matching devices with drivers. 

Linux knows available bus from information passed at run-time by the system (ACPI) or through device trees. The bus handles the addition and removal of devices and drivers, and it matches drivers to devices. 
Drivers register themselves with a particular bus using `driver_register()`, broadcasting the handled devices.  
A device is initialized using the driver `probe()` method.

Linux kernel provides **driver frameworks** to promote best practices, provide consistency, minimizing code repetition and handling complex HW interactions. 
### Bus framework
They simplify interaction with complex bus systems. 
It is designed for non-discoverable devices often found in system-on-chip (SoC) or embedded systems. 
Unlike other buses, platform devices are known to the kernel at boot time. 
#### PCI bus framework
**Peripheral Component Interconnect** express (PCIe) is a high-speed serial computer expansion bus standard designed around point-to-point topology. 
It introduces a logical view: 
- Root complex: acts as a bridge between CPU and memory subsystem to PCIe devices. 
- Endpoints: terminal devices of a PCIe connection (e.g., GPU, SSDs) which initiate or respond to a PCIe transaction. 
- Bridges: connects different bus types or PCIe hierarchies. 

Linux identifies and prepare peripherals through **device enumeration**. 
A specific case is **PCI enumeration**: during system boot, the firmware interacts with each PCI peripherals to allocate safe space for memory mapper I/O and port I/O regions. 
`dsmeg` log provides a record of the PCI peripherals that Linux recognizes during booting. 
Each PCI device has a configuration space containing the device id, vendor id and six **Base Address Registers** (**BARS**, distinguish between memory-mapped and I/O-mapper resources). 
## Sysfs and kobjects
**kobjects** are the underlying foundation of the device model.

- They represent kernel objects such as bus, devices, drivers, and modules.
- They are able to emit **uevents**, notifications sent to user-space tools, signaling any changes in the kobjects. 
- They belong to **ksets** and form hierarchies. 

The **sysfs** exports information about these objects to user space, providing structured access to hardware details.

![[map.io.jpg]]

# Platform Configuration
When booting, the OS needs to be aware of: 
- Devices that are already present on the machine.
- How interrupts are managed.
- The number of processors available.

To address these needs, two standards have been developed to provide the kernel with all the necessary data: 
- **Advanced Configuration** and **Power Interface** (ACPI): primarily used on **general-purpose platforms**, particularly those utilizing Intel processors. 
- **Device trees**: mainly used on **embedded platforms**.
## ACPI
ACPI helps with discoverability, power management, and thermal management. 
- It provides an open standard for OS to discover and configure computer hardware components, eliminating the need for platform-specific drivers. 
- It enables power management features (e.g., putting unused hardware components to sleep).
- It supports auto-configuration (e.g., plug-and-play and hot swapping). 
- It facilitates status monitoring.
![](ACPI.png)

- **ACPI tables** contains static data structures provided by BIOS or UEFI firmware.
- **ACPI registers** are used for control tasks.
- ACPI ML (**AML**) **interpreter** reads and execute AML code, enabling **dynamic configuration**.
- **Operating System-directed configuration** and **Power Management** (**OSPM**) allows the kernel to set the appropriate state for a CPU when it's idle.
- User space interacts with the kernel's ACPI through an **ACPI daemon** (`acpid` in Linux) which listens for ACPI events and executes predefined scripts in response. 

The **Differentiated System Descriptor Table** (DSDP) is the main ACPI table which describe HW devices and configuration. It contains the **ACPI namespace**, a hierarchical data structure describing the underlying HW structure. 

Power management refers to the various tools and techniques used to consume the minimum amount of power necessary based on the system state, configuration, and use case. 
The goals are:
- Improvement in reliability: high temperature reduce the reliability of HW components.
- Improvement in battery lifetime (energy consumption).
- Ensure regulatory compliance.

**Thermal Design Power** (TDP) is the average value of physical power that a cooling system must be able to dissipate to ensure sustained reliability. 
Nowadays, TDP has reached a **plateau**, also called **power wall**, since higher values would require prohibitive cooling costs and put the product out of market. For this reason, the industry shifted towards heterogeneous cores to improve performance. 
The primary source of **power consumption** is related to the **frequency** and **voltage** of each system device.

Power management is complex due to the interdependence of multiple variables. To simplify this, ACPI uses **device states**: tables that describe attached devices, their power states and controls for putting them into a different power state. 

ACPI defines a hierarchical state machine to describe the power state of a system. There are five different power state types: 
- **G-type** states group together S-type states and provide a coarse-grained description of the system's behavior. They can indicate whether the system is working, appearing off but able to wake up on specific events, or completely off.
- **S-type** (**sleep**) states describe how the computer implements the corresponding G-type state. For example, the G1 state can be implemented in different ways, such as halting the CPU while keeping the RAM on, or completely powering off the CPU and RAM and moving memory content to disk.
- **C-type** (**CPU**) states allow the CPU to reduce its clock speed (C0) or even shut down completely when idle to save energy (C1-C3). They only make sense in S0 configuration (working state) and are usually invoked through a specific CPU instruction, such as MWAIT on Intel processors.
- **D-type** (**device**) states are the equivalent of C-states but for peripheral devices. 
- **P-type** (**performance**) allows the CPU to adjust its clock speed and voltage based on workload demands.They only make sense in the C0 configuration (working state).

When the scheduler identifies that more performance is required (a P-state change), it communicates with the CPUFreq module in OSPM, which then interacts with ACPI, which in turn triggers the change by modifying the Performance Control Machine register. 
## Device trees
It is a data structure describing HW: what peripheral devices are present, their memory addresses clock information, and so on. 
They can be in textual format (`.dts`, device tree source), which is human-readable, or in binary format (`.dtb`, device tree binary) which is the compiled version interpreted by the OS at boot. 
Nodes are organized in a hierarchy as a collection of property and value tokens. 
# Booting
General purpose PCs uses:
- BIOS: older firmware interface which has several constraints (slower and cannot boot from drivers larger than 2TB since it only support 32-bit logical block addressing).
- UEFI:  **Unified Extensible Firmware Interface** a modern replacement for BIOS.

Embedded devices uses:
- **UBOOT**: uses in Linux systems, it has broad support for various HW and provides same functionality as BIOS/UEFI.
- **Simpler bootloaders** for bare metal OSes, which may reside in a protected area of the processor flash memory and simply copying the application from non-volatile storage into RAM and executing it. 
## UEFI
- It is modular, it can be extended with drivers. For this reason it support multiple platforms.
- It takes control right after the system is powered on, it quickly takes control to initialize system hardware and load firmware settings into RAM. 
- It uses a dedicated FAT32 partition, known as the **EFI System Partition** (**ESP**), to store bootloader (startup) files for various OS. It is more secure than BIOS. 
- It uses **GUID Partition Table** (**GPT**) to overcome the size limitations of BIOS and allows more flexible partitioning​​ (support 64-bit logical block addressing). The GPT does not require a special boot code, unlike the older Master Boot Record (MBR) system. It is located at the beginning of the disk, starting from sector 1 as sector 0 is reserved for a protective MBR to ensure limited backward compatibility. Each partition entry holds significant information:
	- A globally unique identifier (GUID), which indicated the type of partition.
	- Starting and ending LBAs of the partition, defining its location and size on disk.
	- Flags for specific features or permission of the partition. 
	- Each partition has a descriptive name. 

![[map.boot.jpg]]

# Virtualization 
A Virtual Machine (VM) is an effective, isolated replication of an actual computer designed for running a specific operating system (OS). 

It's based on a **Virtual Machine Monitor** (VMM) or **hypervisor**. 
There are two types of hypervisors:
- **Type 1 Hypervisor**: native or **bare-metal** hypervisor, it operates **directly on the hardware** without an underlying host OS.
- **Type 2 Hypervisor**: runs on top of a **host OS** (e.g., KVM or VirtualBox).

A mention also goes to **paravirtualization**, which is when the guest OS is modified to work closely with the hypervisor, leading to improved performance and reduced virtualization overhead.
The VMM has total control over system resources, ensuring:

- **Fidelity**: VM behaves in the same way as the real machine.
- **Safety**: VM is restricted from bypassing the VMM's management of virtualized resources.
- **Efficiency**: programs running within the VM experience little to no significant drop in performance.

Reasons for using a VM:
- **Consolidation**: multiple VMs can run on a single physical machine, **maximizing hardware utilization** by running one machine at full capacity instead of multiple machines at partial capacity.
- **Adaptability**: VMs can quickly adjust to changing workloads.
- **Cost Reduction**: data centers can reduce both hardware and administrative expenses using VM.
- **Scalability**: VMs can be easily scaled horizontally to meet increased demands.
- **Standardization**: VMs provide a standardized infrastructure across different environments.
- **Security and Reliability**: VMs offer secure sandboxing for applications and can enhance fault tolerance.

Instruction can be **privileged**, which traps if executed in user mode, or **unprivileged**. 
A privileged instructions affect the state within the virtual CPU, represented in the Virtual Machine Control Block (VMCB).

**Virtualization-sensitive** instructions: 
- **Controls Sensitive**: Directly changes the machine status, like enabling or disabling interrupts.
- **Behavior Sensitive**: Operates differently depending on whether it's executed in user or supervisor mode, which can impact fidelity.

> [!THEOREM] Popek and Goldberg theorem
> For any conventional computer, a virtual machine monitor may be built if the **set of sensitive instructions** for that computer is a **subset** of the **set of privileged instructions**.

This theorem states that for any conventional computer, a VMM can control the execution of guest OS (by intercepting and emulating privileged instructions) only if all sensitive instructions are also privileged: so that any attempt by the guest OS to execute a sensitive instruction will cause a **trap** to the VMM, allowing it to **handle** the instruction appropriately. 

This theorem is a **sufficient condition**, not a necessary one as some mechanism exist(ed) to achieve virtualization for x86 processors not equipped with VT-x.

Virtualization can be achieved through:
1. **Software-Based** (de-privileging): 
	- Based on "**trap and emulate**" working.
	- Together with **shadowing**, a basic constituent of software virtualization is **ring deprivileging**: the process of reducing the privilege level of a virtual machine by moving it from a higher privilege ring to a lower one.
2. **Hardware-Assisted**: 
	  - Modern processors offer built-in support for virtualization, making it easier and more efficient. These processors have **additional modes for guest and hypervisor operations**.
	  - Some instructions aren't trapped, fetched, and executed by the hypervisor, but directly transformed into **emulation routines**.
	  - It has a control block where the state of the guest can be saved and resumed. 

![[virtualization-techniques.png]]

## SW-based virtualization
Deprivileging is when a virtual machine functions with lesser access to hardware and system resources than it would if it ran on a physical system.
It's all about to deprivilege an instruction from ring 0 to run in ring 1 (or less).

> [!TIP] Rings
> -  **Ring 0**: Kernel mode, full privileges.
> - **Ring 1**: (Rarely used) Driver mode, intermediate privileges.
> - **Ring 2**: (Rarely used) Driver mode, intermediate privileges.
> - **Ring 3**: User mode, least privileges.

If all virtualization sensitive instruction are privileged, deprivileging them enables the VMM to intervene when necessary, using:
- **Trap**: when a user program in VM tries to execute a privileged instruction, the CPU raises a trap (exception or interrupt).
- **Handle** (**emulate**): the OS kernel catches the trap and emulates the effect of that privileged instruction safely.

The VMM manages the guest OS virtual memory using **shadow page tables**: when the guest OS attempts to sets its own page table, the VMM builds a corresponding shadow page table (in the physical host) that map each guest virtual address into a host physical address. 
VMM marks the guest pages as read-only, so any attempt by the guest to modify them triggers a trap. 
Because of privilege separation within the guest, the VMM maintains **two separate shadow page tables**: one for guest user and one for guest supervisor, since a guest user reading a guest supervisor page does not produce a trap visible to the hypervisor. 

Deprivileging might pose a problem in terms of **fidelity** because, if not taken care of, a guest could understand that it is running in unprivileged mode. 
This is resolved through **Just In Time compilation**. 
JIT compiler detects sensitive or privileged instructions in the guest code that need special handling when the guest OS is deprivileged, and manages:
- **Translation**: instead of executing instructions directly, the JIT compiler translates them into a sequence of safe, unprivileged instructions that the host system can execute without compromising security or stability.
- **Caching**: translated instructions are cached so that subsequent executions of the same instructions can use the optimized versions without needing re-translation, improving performance.

## HW-based virtualization
Main goals are to avoid the problem of deprivileging, allow the guest state to be saved and resumed, reduce the number of traps and avoid shadow paging overhead. 

Virtualization is obtained with the introduction of privileges modes (no ring aliasing): it can be a third privileged mode specifically for hypervisor or a second dimension for VM (e.g., intel VT-x).

**Intel VT-x** introduces a dedicated **vmx root-mode** where hypervisor and host OS operate from, while VM function in **vmx non-root mode**. 
- The entire processor state is duplicated, allowing fast switches between guest and host without needing to save and restore all CPU state.
- Root-mode execution can be virtualized too (recursive virtualization).
Memory mapping is managed through **Extended Page Tables** (EPT), a HW component that adds a second level of address translation: 
- The guest OS maintains its own guest page table, mapping guest virtual addresses to guest physical addresses.
- The VMM sets up EPTs, which maps guest physical addresses to host physical addresses.
### KVM
**Kernel-based Virtual Machine** (KVM) is a virtualization technology integrated into the Linux kernel, which allows Linux to be a type 1 hypervisor.

KVM is a kernel module that can be accesses as a device (it can be found in `/dev/kvm`).
Each virtual CPU is manages as a regular Linux process.
The **Virtual Machine Control Structure** (VMCS) controls the behavior and state of each VM. Through VMCS, KVM can control when the VM traps.

- **Memory overcommitment**: since VMs are Linux processes, and Linux virtual memory allows to have each process consume all of its address space, we can do overcommitment by default. It allows to assign more memory to VM than the physically available one. When the actual memory usage approaches the physical one, techniques like ballooning are used to reclaim memory.
- **Balooning**: allows a hypervisor to dynamically adjust the amount of physical memory available to VMs. A **special driver** installed in the **guest OS** allocates memory to itself, forcing the guest OS to **swap or release** less frequently used memory pages back to the host, allowing the hypervisor to reallocate this memory to other VMs.
- **Kernel same-page merging** (page duplication): is a feature in Linux designed to save memory across multiple processes or VMs. It works by scanning for physical pages with identical content, remaps each duplicate virtual page to point to the same physical one (marking all virtual pages as copy-on-write) and release the extra physical page for re-use. 

Developing a VMM requires to handle all interactions with the HW (i.e. emulated devices and hardware attached to host also called pass-through).

QEMU is an emulator based on binary translation. It can be used as a VMM based on KVM (using `--enable-kvm`). It provides a solid ecosystem of existing **emulated devices**. 
- Every attempt to read/write HW registers triggers a VM-exit, the virtual CPU thread transition into KVM where the VMM (QEMU) perform the actual I/O operation using host resources. 
- **VIRTIO** interface creates a **shared** memory mechanism (**virtqueue**) for hypervisors and virtual devices to communicate without much overhead. Since the guest device driver uses VIRTIO, we are in a para-virtualization setting. 

**Xen paravirtualization** is based on the idea of replacing system-specific routines with **hypercalls**: this optimize virtualization performance by allowing the guest OS to be aware of its execution on a virtual environment. 
Xen is a type 1 hypervisor. Critical instruction that would normally require ring 0 privilege are replaced with hypercalls to the Xen hypervisor.
## Containerization 
Containers are a way to isolate a set of processes and make them think that they are the only ones running on the machine. They are not VM. 
- Processes running inside a container are normal processes running on the host kernel.
- There is no guest kernel running inside the container.
- We cannot run an arbitrary OS, since the kernel is shared with the host. 
It is referred as OS-level virtualization. Inside the container there is NO performance drop. 

Containers are based mainly on: 
- **Namespaces** partitions kernel resources so that a set of processes sees only a specific set of resources. It enables **process isolation**: each process has its own network namespace, PID namespace, user namespace and a few others. PID namespaces allow containerized application to have their own independent process trees. 
	- Unix Timeshared System (UTS) namespaces give each process tree its host and domain names.
	- Mount namespaces allows containers to have a custom filesystem layout.
	- Network namespaces allow process to have separate network stack, including IP routes and interfaces.
	- User namespaces let processes inside a container to have different user and group identifies than on the host. 
- **CGroups** (control groups).

![[map.kvm.jpg]]