# MPI Communications

## Communication between Processes

To optimize data exchange in distributed computing, MPI organizes processes into **communicators** (groups of processes), where each process receives a unique identifier called `rank`. The global communicator that includes all processes in the system is `MPI_COMM_WORLD`.

To improve performance and portability, MPI allows the definition of specific **datatypes**. These allow describing complex memory layouts, facilitating the rapid transfer of data between nodes.

Point-to-point communication also utilizes a `tag` (a user-chosen integer) to qualify the message. If the criteria of an `MPI_Recv` do not match the incoming message's tag, the function remains waiting, while the non-matching message is temporarily preserved in the network buffer. The wildcard `MPI_ANY_TAG` is often used to accept messages with any identifier, subsequently retrieving the actual tag via the `status` parameter.

Communications in MPI are divided into point-to-point and collective:

* **Point-to-point:** Involve pairs of processes. On supercomputers with a high number of nodes, using them for global communications entails the disadvantage of having to execute a huge number of function calls, generating a large overhead and scalability problems.
* **Collective:** Operate at a higher level of abstraction over entire groups of processes. They replace point-to-point communication loops, intrinsically optimizing data exchange and synchronization via network hardware.

## Point-to-Point Communications
Point-to-point communications are necessary for data exchange between individual nodes. However, when the same data must be distributed to multiple processes — or more generally when all processes participate in a coordinated exchange — repeatedly using MPI_Send/MPI_Recv is highly inefficient: it involves many function calls (high overhead), increases the risk of errors in rank management, and does not exploit the optimized algorithms (e.g., tree broadcast) that collective functions implement internally.
### MPI_Send

#### Definition

```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

#### Parameters

| Parameter  | Type            | Description                                        |
|------------|-----------------|----------------------------------------------------|
| `buf`      | `const void *`  | Pointer to the buffer containing the data to be sent |
| `count`    | `int`           | Number of elements to send                         |
| `datatype` | `MPI_Datatype`  | MPI type of the elements (e.g., `MPI_INT`, `MPI_DOUBLE`) |
| `dest`     | `int`           | `rank_id` of the receiving process                 |
| `tag`      | `int`           | Message tag (non-negative integer)                 |
| `comm`     | `MPI_Comm`      | MPI communicator (e.g., `MPI_COMM_WORLD`)          |


#### Blocking behavior

`MPI_Send` is a **blocking** function: it returns only when the `buf` buffer is **safe to reuse**, meaning when MPI has copied the data into its internal system (system buffer or transmission channel).

> ⚠️ **Warning:** the return of the function **does not guarantee** that the message has been delivered to the receiving process. Delivery can occur asynchronously, after the return.

#### MPI_PROC_NULL
If `dest` is set to `MPI_PROC_NULL`, the call **has no effect**: it is treated as a null operation and returns immediately.

### MPI_Recv

#### Definition

```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

#### Parameters

| Parameter  | Type            | Description                                              |
|------------|-----------------|----------------------------------------------------------|
| `buf`      | `void *`        | Pointer to the buffer where the received data will be written |
| `count`    | `int`           | **Maximum** number of elements the buffer can contain    |
| `datatype` | `MPI_Datatype`  | MPI type of the elements (e.g., `MPI_INT`, `MPI_DOUBLE`) |
| `source`   | `int`           | `rank_id` of the sender (or `MPI_ANY_SOURCE`)            |
| `tag`      | `int`           | Message tag (or `MPI_ANY_TAG`)                           |
| `comm`     | `MPI_Comm`      | MPI communicator (e.g., `MPI_COMM_WORLD`)                |
| `status`   | `MPI_Status *`  | Structure with information on the received message       |


#### Blocking behavior

`MPI_Recv` is a **blocking** function: it returns only when the data has been successfully written into the `buf` buffer and is ready to be used. To execute the rest of the code, the function waits for the data to be received. This behavior is useful because, typically, the expected data is necessary to proceed with the computation.

> `MPI_Recv` **guarantees** that the data is available in the buffer upon return.


#### MPI_Status

The `status` structure contains information about the received message:

| Field              | Description                          |
|--------------------|--------------------------------------|
| `MPI_SOURCE`       | Rank of the actual sender            |
| `MPI_TAG`          | Tag of the received message          |
| `MPI_ERROR`        | Error code                           |

Especially useful when using wildcards (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`), to know who actually sent the message and with what tag.  
If the information is not needed, `MPI_STATUS_IGNORE` can be passed.

#### Count and message size

`count` specifies the **maximum capacity** of the buffer, not the exact number of expected elements.

- If the received message contains **fewer elements** than `count`: no error, the buffer is partially filled.
- If the received message contains **more elements** than `count`: overflow error (`MPI_ERR_TRUNCATE`).

To find out how many elements were **actually received**, `MPI_Get_count()` is used:

```c
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
```

It reads the number of received elements from the `status` structure and writes it into `count`. It must be called **after** `MPI_Recv`, passing the same `datatype` used in the reception.

```c
int received;
MPI_Get_count(&status, MPI_INT, &received);
```

#### Wildcards

| Constant          | Used for  | Effect                                         |
|-------------------|-----------|------------------------------------------------|
| `MPI_ANY_SOURCE`  | `source`  | Accepts messages from any sender               |
| `MPI_ANY_TAG`     | `tag`     | Accepts messages with any tag                  |

### Deadlocks
 
When using `MPI_Send` and `MPI_Recv`, particular attention must be paid to possible deadlocks. Since both functions are **blocking**, if two processes wait for each other without either proceeding, the program blocks indefinitely.
 
#### Classic scenario
 
The most common case occurs when two processes want to exchange data and both call `MPI_Send` before calling `MPI_Recv`:
 
```
Process 0                   Process 1
──────────────────          ──────────────────
MPI_Send → P1   ────┐  ┌─── MPI_Send → P0
                    │  │
                (both blocked: waiting for
                 the other to call Recv)
                    │  │
MPI_Recv ← P1   ────┘  └─── MPI_Recv ← P0
         (never reached)      (never reached)
```
 
`MPI_Send` is blocking: process 0 remains stalled until the message is handled by the MPI system. If process 1 is also doing the same concurrently, neither ever calls `MPI_Recv` and the program blocks.
 
> ⚠️ The behavior depends on the MPI implementation: for **small** messages, some systems internally buffer the message and the Send returns immediately, masking the deadlock. For **large** messages, the buffer is exhausted and the deadlock manifests. Do not rely on this behavior.
 
#### Solution: invert the order on one process
 
It is enough for one of the two to call `MPI_Recv` before `MPI_Send`:
 
```
Process 0                   Process 1
──────────────────          ──────────────────
MPI_Send → P1   ──────────► MPI_Recv ← P0    ✓
MPI_Recv ← P1   ◄────────── MPI_Send → P0    ✓
```
 
In this way, process 1 is already listening when process 0 sends, and the exchange occurs without blocking. 

### MPI_Sendrecv — Simultaneous and safe exchange

A more robust alternative to avoid deadlocks when two processes need to exchange data simultaneously (as in the case of exchanging *ghost cells*) is the use of `MPI_Sendrecv`. This function performs a send and a receive operation in a single call, delegating the internal management of precedences and buffers to MPI to prevent any blocking.

#### Definition

```c
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status)
```

The parameters are simply the concatenation of those of an `MPI_Send` and an `MPI_Recv`.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `sendbuf` | `const void *` | Pointer to the send data buffer |
| `sendcount` | `int` | Number of elements to send |
| `sendtype` | `MPI_Datatype` | MPI type of the elements to send |
| `dest` | `int` | Rank of the receiving process |
| `sendtag` | `int` | Outgoing message tag |
| `recvbuf` | `void *` | Pointer to the buffer where to save the incoming data |
| `recvcount` | `int` | Maximum number of elements to receive |
| `recvtype` | `MPI_Datatype` | MPI type of the elements to receive |
| `source` | `int` | Rank of the sending process |
| `recvtag` | `int` | Incoming message tag |
| `comm` | `MPI_Comm` | MPI communicator |
| `status` | `MPI_Status *` | Receive status (or `MPI_STATUS_IGNORE`) |

#### Why use it?

* **Zero Deadlock:** Processes can call it simultaneously without worrying about execution order, because MPI handles bidirectional communication safely.
* **Cleaner code:** It replaces conditional blocks (e.g., `if (rank % 2 == 0) { send; receive; } else { receive; send; }`), making the code much more readable, compact, and less prone to human error.

> ⚠️ **Watch out for buffers:** The `sendbuf` and `recvbuf` buffers must be **disjoint** in memory. It is not possible to send and receive data by overwriting the same memory area (if this behavior is needed, there is a specific variant called `MPI_Sendrecv_replace`).

### MPI_Isend and MPI_Irecv — Non-blocking communications

`MPI_Isend` and `MPI_Irecv` are the **non-blocking** versions of `MPI_Send` and `MPI_Recv`: they initiate the communication operation and return **immediately**, without waiting for its completion.

#### Definitions

```c
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request)

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request)
```

The additional parameter compared to the blocking versions is `MPI_Request *request`: a handle that identifies the ongoing operation and is later used to verify its completion.

#### Completion: MPI_Wait and MPI_Test

Since the functions return before the communication is concluded, it is necessary to explicitly synchronize before accessing the buffer:

```c
// Blocks until the operation is completed
MPI_Wait(MPI_Request *request, MPI_Status *status)

// Non-blocking: checks if the operation is completed (flag = 1) or not (flag = 0)
MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
```

> ⚠️ **Fundamental rule:** the `buf` buffer **must not be read or modified** between the call to `MPI_Isend`/`MPI_Irecv` and the completion confirmed by `MPI_Wait` or `MPI_Test`. Doing so is undefined behavior.

#### Typical usage pattern

```c
MPI_Request req;

MPI_Isend(buf, count, MPI_INT, dest, tag, MPI_COMM_WORLD, &req);

// computation independent of communication...
do_work();

MPI_Wait(&req, MPI_STATUS_IGNORE); // waits for completion
```

In this way, communication occurs **in parallel** with `do_work()`, hiding the network latency.

#### Blocking vs non-blocking comparison

```
MPI_Send (blocking)
────────────────────────────────────────────────────
[  Send (wait)  ][  computation  ]

MPI_Isend (non-blocking)
────────────────────────────────────────────────────
[Isend][ computation  ][Wait]
         ↑
    communication ongoing in background
```

| Aspect                | MPI_Send / MPI_Recv       | MPI_Isend / MPI_Irecv              |
|-----------------------|---------------------------|------------------------------------|
| Return                | Only on completion        | Immediate                          |
| Buffer after call     | Immediately reusable      | Do not touch before MPI_Wait       |
| Deadlock risk         | High (critical order)     | Low (do not block each other)      |
| Comm/compute overlap  | ✗                         | ✓                                  |
| Code complexity       | Low                       | Higher                             |

#### Why MPI_Isend is very useful

In HPC applications, the time spent in communication is often the bottleneck. `MPI_Isend` allows **overlapping communication and computation**: while data travels over the network, the process can already work on the part of the computation that does not depend on that data. This pattern, called *latency hiding*, can lead to significant speedups compared to the blocking alternative, especially in applications with many exchanges between processes.

Furthermore, since `MPI_Isend` and `MPI_Irecv` do not block, two processes can both call them in the same order **without deadlock risk**, simplifying synchronization management compared to the blocking versions.

### MPI_Abort

The `MPI_Abort()` function is used in parallel programming with MPI (Message Passing Interface) to force the abnormal termination of all processes belonging to a given communicator (generally `MPI_COMM_WORLD`). 
The typical use case occurs when **a critical and unrecoverable error happens on only one of the nodes** or processes of the computational grid. In a parallel application, nodes are tightly interconnected via blocking communication operations (such as `MPI_Send`, `MPI_Recv`, or synchronization barriers like `MPI_Barrier`). If a single node were to fail or silently halt its execution without notifying the others, the entire application would enter a **deadlock** state (indefinite blocking), continuing to consume cluster resources without producing any result. By invoking `MPI_Abort(MPI_COMM_WORLD, error_code)`, the node that encountered the anomaly orders the MPI runtime environment to immediately and cleanly terminate all other processes, returning an error code (`error_code`) useful for debugging.
The use of a normal system function like `exit()` would solely terminate the local process where the error occurred. Since the nodes are distributed over a network infrastructure, reliably and automatically propagating this exit state to all other nodes would be extremely difficult. Consequently, the faulty node would shut down in isolation, while the rest of the cluster would continue execution, remaining fatally blocked waiting for communications (deadlock).

## Collective Communications in MPI

In parallel computing with **MPI (Message Passing Interface)**, collective communications (such as `MPI_Bcast`, `MPI_Reduce`, or `MPI_Gather`) coordinate data exchange among all processes in a communicator in a single operation, offering clear performance advantages over point-to-point communications (`MPI_Send`/`MPI_Recv`). Knowing how to use them correctly is therefore a key requirement for a programmer in distributed computing.

### Why Collectives are more efficient than Point-to-Point:

1. **Algorithmic Complexity $O(\log N)$**: If a master process sends data to $N$ nodes via point-to-point, it takes linear time $O(N)$. Collective functions instead use tree algorithms (e.g., binomial trees) where intermediate nodes redistribute the message, dropping the complexity to a logarithmic level $O(\log N)$.
2. **Topological Optimization (Hardware Awareness)**: MPI libraries recognize the underlying architecture. They exploit shared memory for processes on the same node and hardware multicast (e.g., InfiniBand) for network traffic, optimizing the data path transparently to the user.
3. **Reduction of Overhead and Deadlocks**: Manually managing dozens of point-to-point communications increases the risk of bottlenecks on the master and mutual blockages (*deadlocks*). Collectives offer implicit and optimized synchronization, reducing control overhead and simplifying code structure.

### MPI_Barrier
The `MPI_Barrier(comm)` function is a collective synchronization operation that acts on a group of processes associated with a specific communicator, such as `MPI_COMM_WORLD`. When a process reaches and executes this instruction, it blocks and waits until all other processes belonging to the group have also reached the same call. Only when all processes have arrived at the barrier are they unblocked and free to continue their execution.

#### Advantages
* **Safe phase management:** Represents an extremely simple and direct way to separate two distinct phases of a parallel computation, ensuring that messages generated in one phase do not interfere with those of the next phase.

#### Disadvantages
* **Performance overhead:** Being a global operation that requires the participation and waiting of all processes in the communicator, it can be very time-consuming and significantly slow down program execution.
* **Often avoidable:** In many cases, invoking `MPI_Barrier()` should be avoided, as synchronization between processes can be achieved more efficiently by correctly structuring the explicit addressing of communications (for example, by exploiting `tags`, the sender identifier `source`, or the isolation guaranteed by separate communicators).

### MPI_Bcast
The `MPI_Bcast()` (Broadcast) function allows a single process, defined as "root", to send an exact copy of the same data to all other processes belonging to a given group or communicator. Being a collective operation, the function must strictly be invoked by all processes in the group.

#### Function arguments
The basic syntax is `MPI_Bcast(buffer, count, datatype, root, comm)`:
* **`buffer`**: pointer to the memory area of the data. For the `root` process, it represents the address from which to read the data to send; for all other processes, it represents the address where to write the received data.
* **`count`**: number of elements that make up the message.
* **`datatype`**: data type of the transmitted elements (e.g., `MPI_INT`).
* **`root`**: the identifier (rank) of the sending process that holds the initial data.
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

#### Operating scheme
Here is an example of how the buffer propagates if the source process (`root`) is *Proc 1* and must send 3 elements (identified as 1, 2, 3) to a total of 4 processes.

**Before calling `MPI_Bcast()`** Proc 0: `[   |   |   ]`  
Proc 1: `[ 1 | 2 | 3 ]`  <-- ROOT  
Proc 2: `[   |   |   ]`  
Proc 3: `[   |   |   ]`  

**After calling `MPI_Bcast()`** Proc 0: `[ 1 | 2 | 3 ]`   
Proc 1: `[ 1 | 2 | 3 ]`  
Proc 2: `[ 1 | 2 | 3 ]`  
Proc 3: `[ 1 | 2 | 3 ]`  


### MPI_Scatter

The `MPI_Scatter()` function performs a data distribution operation (and is conceptually the inverse of `MPI_Gather()`). It takes a block of contiguous data residing on a single process (defined as "root"), divides it into segments of equal size, and sends a distinct segment to each process within the group (including the root itself).

#### Function arguments
The standard syntax is `MPI_Scatter(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm)`:
* **`sendbuf`**: pointer to the array of data to be divided (relevant and read only by the `root` process).
* **`sendcnt`**: number of elements (not bytes!) sent to *each individual process*, not the total elements of the array.
* **`sendtype`**: data type of the elements in the send buffer (e.g., `MPI_INT`).
* **`recvbuf`**: pointer to the memory area where the receiving process will save its portion.
* **`recvcnt`**: number of elements each process receives (usually coincides with `sendcnt`).
* **`recvtype`**: data type of the elements in the receive buffer.
* **`root`**: the identifier (rank) of the process that holds the initial data to be scattered.
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

> 💡 **Note on send and receive parameters:**
> The presence of distinct parameters for sending (`sendcnt`, `sendtype`) and receiving (`recvcnt`, `recvtype`) stems from the fact that `MPI_Scatter()` produces the same result as a series of `MPI_Send` calls executed by the root node and `MPI_Recv` calls executed by all other processes. This design is not redundant but explicitly allows that **data types and quantities can be different** between sender and receiver. In fact, MPI does not require communicating processes to use the same data representation. However, in the vast majority of cases, these values will coincide; indeed, changing the data type is strongly discouraged.

#### Operating scheme
Here is a scheme of how the array is "scattered". Let's assume that **Proc 0** is the `root` and holds an array of 12 elements to distribute to a total of 4 processes. In this case, `sendcnt` and `recvcnt` will be 3, since each process will receive 3 elements.

**Before calling `MPI_Scatter()`** Proc 0 (ROOT): `sendbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  
Proc 1:        `recvbuf = [   |   |   ]`  
Proc 2:        `recvbuf = [   |   |   ]`  
Proc 3:        `recvbuf = [   |   |   ]`  

**After calling `MPI_Scatter()`** Proc 0: `recvbuf = [ 1 |  2 |  3 ]`  
Proc 1: `recvbuf = [ 4 |  5 |  6 ]`  
Proc 2: `recvbuf = [ 7 |  8 |  9 ]`  
Proc 3: `recvbuf = [ 10| 11 | 12 ]`  


### MPI_Gather

The `MPI_Gather()` function implements a "many-to-one" (all-to-one) collective communication and is the exact inverse operation of `MPI_Scatter()`. It gathers distinct data from the send buffers of each process within the group (including the destination node) and concatenates them in rank order into the receive buffer of a single specified process, defined as "root".

#### Function arguments
The standard syntax is `MPI_Gather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm)`:
* **`sendbuf`**: pointer to the memory area containing the data that the individual process must send.
* **`sendcnt`**: number of elements sent by the individual process.
* **`sendtype`**: data type of the elements in the send buffer.
* **`recvbuf`**: pointer to the memory area where the `root` process will save the gathered data. It is relevant only for the `root` process, which must have previously allocated enough space to hold all incoming data.
* **`recvcnt`**: number of elements the root receives *from each individual process* (not the total).
* **`recvtype`**: data type of the expected elements in the receive buffer.
* **`root`**: the identifier (rank) of the destination process that will gather all messages.
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

> 💡 **Note on parameters:**
> For `MPI_Gather()` as well, the quantities and data types declared for sending and receiving can differ from each other, allowing network operations to handle conversion between heterogeneous processes or platforms. 

#### Operating scheme
Here is a scheme of how the data is gathered. Suppose that **Proc 1** is the destination process (`root`) and that all 4 processes in the communicator must send 3 elements each. In this case, `sendcnt` and `recvcnt` will be 3.

**Before calling `MPI_Gather()`** Proc 0:        `sendbuf = [  1 |  2 |  3 ]`  
Proc 1 (ROOT): `sendbuf = [  4 |  5 |  6 ]`  
Proc 2:        `sendbuf = [  7 |  8 |  9 ]`  
Proc 3:        `sendbuf = [ 10 | 11 | 12 ]`  

**After calling `MPI_Gather()`** Proc 0: (Data sent)  
Proc 1: `recvbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  
Proc 2: (Data sent)  
Proc 3: (Data sent)  

Here is a practical example combining the use of `MPI_Scatter` and `MPI_Gather` to solve a classic problem: the **sum of two vectors**. 

The idea behind this scheme is the "scatter-compute-gather" pattern. The "root" process (usually Proc 0) holds the two initial complete arrays. It uses Scatter to divide the data and send it to the various nodes. Each node performs the sum only on its small portion of data, generating a fragment of the result. Finally, the root uses Gather to recompose the fragments into the final result vector.


### Practical Example: Parallel Vector Sum

The combination of the `MPI_Scatter` and `MPI_Gather` functions is ideal for parallelizing operations on arrays such as the sum of two vectors, $x$ and $y$, to obtain a result vector $z$ (where $z_i = x_i + y_i$). 

The process is divided into three main phases:
1. **Distribution (Scatter):** The `root` process (Proc 0) invokes `MPI_Scatter` twice: once to fragment and distribute the vector $x$ and once for the vector $y$. Each process thus receives its own partial arrays (`local_x` and `local_y`).
2. **Local computation:** Each process, in parallel and independently, executes a normal `for` loop to sum its `local_x` and `local_y` element by element, saving the result in a `local_z` array.
3. **Gathering (Gather):** The `root` process invokes `MPI_Gather` to gather all the `local_z` from the various processes and concatenate them into the final vector $z$.

#### Operating scheme
![Descriptive image of a parallel vector sum.](../immagini/vector_sum.jpeg)


### MPI_Allgather()

The `MPI_Allgather()` function combines the gathering and global distribution of data in a single operation. Each process in the communicator sends its own portion of data, and at the end of the operation, **all** processes possess the entire concatenated dataset. Conceptually, it is equivalent to executing an `MPI_Gather()` (to gather data on one node) immediately followed by an `MPI_Bcast()` (to broadcast the result to all).

#### Function arguments
Since the result is distributed to all nodes, **the `root` argument is not present**. The standard syntax is `MPI_Allgather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, comm)`:
* **`sendbuf`**: pointer to the memory area containing the data that the individual process must send.
* **`sendcnt`**: number of elements sent by the individual process.
* **`sendtype`**: data type of the elements in the send buffer.
* **`recvbuf`**: pointer to the memory area where **every process** will save the total gathered data. *All* processes must have allocated enough space to contain the data coming from the entire group.
* **`recvcnt`**: number of elements each process receives *from each individual node* (not the total size of the final array).
* **`recvtype`**: data type of the elements in the receive buffer.
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

#### Operating scheme
Suppose we have 4 processes in the communicator. Each of them possesses a `sendbuf` array of 3 elements. Setting `sendcnt = 3` and `recvcnt = 3`, at the end of the operation all 4 processes will have an identical `recvbuf` array of 12 elements (ordered based on the rank of the sender).

**Before calling `MPI_Allgather()`** Proc 0: `sendbuf = [  1 |  2 |  3 ]`  
Proc 1: `sendbuf = [  4 |  5 |  6 ]`  
Proc 2: `sendbuf = [  7 |  8 |  9 ]`  
Proc 3: `sendbuf = [ 10 | 11 | 12 ]`  

**After calling `MPI_Allgather()`** Proc 0: `recvbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  
Proc 1: `recvbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  
Proc 2: `recvbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  
Proc 3: `recvbuf = [ 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 ]`  

### Irregular Collective Communications: MPI_Scatterv() and MPI_Gatherv()

The `MPI_Scatterv()` and `MPI_Gatherv()` functions (where the "v" stands for *vector*) are the extended versions of the normal scatter and gather operations. Unlike the basic variants, they allow managing **messages of irregular sizes**, leaving **empty spaces (gaps)** between data, and distributing or gathering array portions in **any order**.

> ⚠️ **WARNING: Highly Prone to Error**
> Having this flexibility comes at a very high cost in terms of code complexity. Using these functions requires the manual construction and management of additional arrays to calculate exactly how many elements to break off for each node and from which memory index to start them. Making a mistake in the calculation of the indices (offsets) or overlapping memory areas is extremely easy and leads to data corruption or program crashes. For this reason, **these functions should be used only if strictly necessary**, namely limited to those cases where the data is so intrinsically irregular that it cannot be handled otherwise.

#### Key arguments of the functions
Taking the signature of `MPI_Scatterv(sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm)` as an example:
* **`sendcnts`** (or `recvcnts` for Gatherv): is no longer a single integer, but an **array of integers**. It individually specifies the number of elements (attention: elements, not bytes!) destined for, or coming from, each process.
* **`displs`** (*displacements*): is an **array of integers**. It specifies the starting index (the offset, always measured in number of elements) within the source buffer from which to extract data for a given process. 

*(Note: For `MPI_Gatherv()`, the logic is mirrored but applied to the root's receive buffer, which will use a `recvcnts` array and a `displs` array to know where to fit the incoming data).*

#### Operating scheme (MPI_Scatterv Example)
Suppose that **Proc 0** (root) must distribute data of different lengths to 3 processes, deliberately ignoring some empty spaces in its starting array.
We must manually define:  
* `sendcnts` = `[ 2, 1, 3 ]` (Proc 0 receives 2 elements, Proc 1 receives 1, Proc 2 receives 3)  
* `displs`   = `[ 0, 3, 5 ]` (The starting indices in the `sendbuf` for each process)  

**Before the call** Proc 0 (ROOT): `sendbuf = [ A | B | - | C | - | D | E | F ]`   
*The indices are:* 0   1   2   3   4   5   6   7      

**After calling `MPI_Scatterv()`** Proc 0: `recvbuf = [ A | B ]`      *(2 elements starting from index 0)* Proc 1: `recvbuf = [ C ]`          *(1 element starting from index 3)* Proc 2: `recvbuf = [ D | E | F ]`  *(3 elements starting from index 5)* #### Code Example: MPI_Scatterv()
This C code snippet demonstrates how to define and pass the `sendcnts` and `displs` arrays for the root process (in this case we assume a total of 3 processes). 

This example is particularly useful for noticing the extreme flexibility of the function: the use of index arrays allows not only sending data out of order, but even overlapping the segments read from the source buffer [1].

```c
int sendbuf[] = {10, 11, 12, 13, 14, 15, 16}; /* Source data on master (Proc 0) */
int displs[] = {3, 0, 1};                     /* Array of starting indices */
int sendcnts[] = {3, 1, 4};                   /* Array of message lengths */
int recvbuf[2];                               /* Receive buffer for each process */

/* ... (MPI initialization code omitted) ... */

MPI_Scatterv(sendbuf, sendcnts, displs, MPI_INT, 
             recvbuf, 5, MPI_INT, 0, MPI_COMM_WORLD);
```

### MPI_Reduce

The `MPI_Reduce()` function (which implements the concept of *Reduction*) is unique because it combines a network communication operation with a computation operation. It gathers data from the send buffers of all processes within a communicator, applies a binary and associative mathematical or logical operation to them (such as a sum or finding a minimum), and saves the final result in the receive buffer of a single specified process, called "root". 

From an algorithmic point of view, this operation is highly optimized: the library does not send all the raw data to the root node to have it calculated there, but distributes the computation across the nodes typically requiring only $O(\log_2 P)$ communication steps for $P$ processes.

#### Function arguments
The standard syntax is `MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm)`:
* **`sendbuf`**: pointer to the memory area containing the process's local data to be combined.
* **`recvbuf`**: pointer to the memory area where the final result will be saved (relevant and written only for the `root` process).
* **`count`**: number of elements to process. **Important note:** if `count > 1`, the reduction does not produce a single scalar, but an array, and is executed **element by element**. The first element of all processes will produce the first element of the result, the second with the second, etc.
* **`datatype`**: data type of the elements.
* **`op`**: the operator to apply (e.g., `MPI_SUM`, `MPI_MAX`, `MPI_LAND`, etc.).
* **`root`**: the identifier (rank) of the process that will receive the final result.
* **`comm`**: the reference communicator.

#### Operating scheme (Element-by-element reduction)
Here is a scheme illustrating a vector reduction. Suppose we have 3 processes, that **Proc 0** is the `root`, and that each **`sendbuf` is an array of size 2** (so `count = 2`). The chosen operator is `MPI_SUM`.

**Before calling `MPI_Reduce()`** Proc 0 (ROOT): `sendbuf = [ 5 | 1 ]`  
Proc 1:        `sendbuf = [ 3 | 2 ]`  
Proc 2:        `sendbuf = [ 7 | 6 ]`  

*(Internal `MPI_SUM` calculation action)* Index 0: `5 + 3 + 7 = 15`  
Index 1: `1 + 2 + 6 = 9`  

**After calling `MPI_Reduce()`** Proc 0 (ROOT): `recvbuf = [ 15 |  9 ]`  
Proc 1:        (Data sent to global calculation)  
Proc 2:        (Data sent to global calculation)  

![Descriptive image of a parallel vector sum with reduction.](../immagini/reduction_mpi.png)


### MPI_Allreduce

The `MPI_Allreduce()` function combines in a single operation the calculation of a global reduction and the distribution of the final result to all participating nodes. Essentially, it performs a distributed calculation (exactly as `MPI_Reduce()` does) but ensures that, upon completion, **all processes receive the final result**, not just a single designated "root" node.

> 💡 **Why it is useful and why prefer it (Reduce + Bcast):**
> This function is extremely useful in scenarios (such as iterative algorithms) where **each process needs to know a global datum to decide how to proceed** or to perform its local computations (for example, calculating the maximum or total error to check if a global stopping criterion has been reached). 
> 
> Although its effect is logically and functionally equivalent to invoking an `MPI_Reduce()` on the root node first and immediately after an `MPI_Bcast()` to redistribute the calculated data, it is **always** advisable to use `MPI_Allreduce()`. In addition to making the code more compact and elegant, it allows the MPI library to optimize communication at the hardware level, resulting in significantly faster, safer, and more efficient execution than two separate network calls.

#### Function arguments
Since the result is automatically distributed to all nodes, **the `root` argument disappears** from the function signature. The standard syntax is `MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm)`:
* **`sendbuf`**: pointer to the memory area containing the process's local data to be combined.
* **`recvbuf`**: pointer to the memory area where **all processes** will save the global final result.
* **`count`**: number of elements to process (here too, if > 1 the reduction is executed element by element).
* **`datatype`**: data type of the elements (e.g., `MPI_INT`).
* **`op`**: the mathematical or logical operator to apply (e.g., `MPI_SUM`, `MPI_MAX`, etc.).
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

#### Operating scheme
Suppose we have 4 processes and want to perform a global sum (`MPI_SUM`) of a single element (`count = 1`).

**Before calling `MPI_Allreduce()`** Proc 0: `sendbuf = [ 0 ]`  
Proc 1: `sendbuf = [ 3 ]`  
Proc 2: `sendbuf = [ 2 ]`  
Proc 3: `sendbuf = [ 4 ]`  

*(Internal action: execution of the global `MPI_SUM` of 0 + 3 + 2 + 4 = 9 and subsequent transparent transmission of the result)* **After calling `MPI_Allreduce()`** Proc 0: `recvbuf = [ 9 ]`  
Proc 1: `recvbuf = [ 9 ]`  
Proc 2: `recvbuf = [ 9 ]`  
Proc 3: `recvbuf = [ 9 ]`  


### MPI_Alltoall

The `MPI_Alltoall()` function is a powerful data movement operation ("all-to-all" communication). It works such that **every process within the group performs its own distribution operation (`MPI_Scatter()`)** to all other processes. 

Essentially, each process divides its send buffer into as many blocks as there are processes in the communicator and sends the *i-th* block to the process with rank *i*. Upon execution completion, the receive buffer of each process will consist of the concatenation of the blocks received from all other nodes, ordered by the index (rank) of the sender.

#### Function arguments
Since there is no single "root" node (everyone sends and everyone receives), the `root` argument is absent. The standard syntax is `MPI_Alltoall(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, comm)`:
* **`sendbuf`**: pointer to the memory area containing the data that the process must send.
* **`sendcnt`**: number of elements that the process sends to *each individual node* (not the total size of the buffer).
* **`sendtype`**: data type of the elements in the send buffer.
* **`recvbuf`**: pointer to the memory area where the process will save the received blocks. It must be large enough to contain the data coming from all processes.
* **`recvcnt`**: number of elements that the process expects to receive *from each individual node*.
* **`recvtype`**: data type of the elements in the receive buffer.
* **`comm`**: the reference communicator (e.g., `MPI_COMM_WORLD`).

#### Operating scheme
Suppose we have 4 processes and that each process wants to send a block of 2 elements (`sendcnt = 2`, `recvcnt = 2`) to each other process. Each `sendbuf` will be 8 elements long in total (4 blocks of 2 elements).

**Before calling `MPI_Alltoall()`** Proc 0 will send the first block to itself, the second to Proc 1, etc.  
Proc 0: `sendbuf = [  1,  2 |  3,  4 |  5,  6 |  7,  8 ]`  
Proc 1: `sendbuf = [  9, 10 | 11, 12 | 13, 14 | 15, 16 ]`  
Proc 2: `sendbuf = [ 17, 18 | 19, 20 | 21, 22 | 23, 24 ]`  
Proc 3: `sendbuf = [ 25, 26 | 27, 28 | 29, 30 | 31, 32 ]`  

**After calling `MPI_Alltoall()`** The receive buffer of each process is filled by assembling, in rank order, the portion of data that each process had specifically prepared for it.  
Proc 0: `recvbuf = [  1,  2 |  9, 10 | 17, 18 | 25, 26 ]`  
Proc 1: `recvbuf = [  3,  4 | 11, 12 | 19, 20 | 27, 28 ]`  
Proc 2: `recvbuf = [  5,  6 | 13, 14 | 21, 22 | 29, 30 ]`  
Proc 3: `recvbuf = [  7,  8 | 15, 16 | 23, 24 | 31, 32 ]`  


### 💡 **Note on data divisibility and remainder management:** It is crucial to remember that basic collective functions like `MPI_Scatter()` and `MPI_Gather()` **do not independently handle the remainders** of data division, nor do they automatically assign the excess to the root process. 
By definition, these functions demand that the sizes of the routed messages be **strictly uniform**: each process must receive or send exactly the same amount of elements indicated in `sendcnt` or `recvcnt`. If the amount of data $N$ is not perfectly divisible by the number of processors $P$, the programmer must take charge of managing the excess. They can do this in two ways:
1. **Manual processing on the root ("Safe" but unbalanced approach):** The programmer calculates the base block size with integer division (`sendcnt = N / P`) and uses the normal `MPI_Scatter()`. The excess data (`N % P`) will not be passed to the collective function and will remain in the root process's memory. The latter, after having distributed the equal parts to everyone (including itself), must have a dedicated portion of code to process the remaining data locally.
 2. **Distribution of the remainder via vector variants (balanced approach):** To prevent the root from having to do extra work alone, the best practice in parallel computing is to distribute the extra load. As typically happens, the eventual imbalance means that some processes are assigned one more element than the others. To implement this at the network level, `MPI_Scatter()` cannot be used, but one must strictly resort to **`MPI_Scatterv()`** or **`MPI_Gatherv()`**. These functions allow specifying **irregular** message sizes via an array (`sendcnts` or `recvcnts`), telling MPI to send, for example, 4 elements to the first nodes and 3 elements to the remaining ones.



### MPI_Scan

The `MPI_Scan()` function performs an inclusive *scan* operation (also known as *prefix sum* or partial prefix sum) on the data distributed among the processes of a communicator. Like the reduction operation, it applies an associative mathematical or logical operator to the data, but the results are distributed incrementally. Upon execution completion, the receive buffer of the process with rank $j$ will contain the reduction (e.g., the sum) of only the data coming from processes with ranks from $0$ up to $j$ (inclusive).

#### Function arguments
Since each process obtains as output an accumulated result specific to its position, **the `root` argument is not present**. The standard syntax is `MPI_Scan(sendbuf, recvbuf, count, datatype, op, comm)`:
* **`sendbuf`**: pointer to the memory area containing the process's local data to be subjected to the *scan*.
* **`recvbuf`**: pointer to the memory area where the process will save the partial accumulation result.
* **`count`**: number of elements to process. If `count > 1`, the *scan* is calculated **element by element** in a parallel and independent manner (e.g., the first element of process $j$'s `recvbuf` will contain the accumulation of the first elements of the `sendbuf`s of processes $0$ through $j$).
* **`datatype`**: data type of the elements.
* **`op`**: the mathematical or logical operator to apply (e.g., `MPI_SUM`, `MPI_MAX`, etc.).
* **`comm`**: the reference communicator.

#### Operating scheme
Suppose we have 4 processes and want to perform an inclusive *scan* with the sum operator (`MPI_SUM`) on a single element (`count = 1`).  

**Before calling `MPI_Scan()`** Proc 0: `sendbuf = [ -2 ]`  
Proc 1: `sendbuf = [  3 ]`  
Proc 2: `sendbuf = [  9 ]`  
Proc 3: `sendbuf = [  6 ]`  

*(Internal action: Proc 0 keeps its data intact, Proc 1 calculates the sum of Proc 0 and 1's data, Proc 2 of Proc 0, 1, and 2's data, Proc 3 of Proc 0, 1, 2, and 3's data)*.  

**After calling `MPI_Scan()`**  
Proc 0: `recvbuf = [ -2 ]`  *(Only Proc 0's data)*   
Proc 1: `recvbuf = [  1 ]`  *(-2 + 3)*   
Proc 2: `recvbuf = [ 10 ]`  *(-2 + 3 + 9)*   
Proc 3: `recvbuf = [ 16 ]`  *(-2 + 3 + 9 + 6)*   

### Summary: The Power of Collective Communications

To conclude this overview, it is essential to understand why collective communications represent the beating heart of advanced MPI programming. These operations, which jointly involve **all processes** within a communicator to calculate or share a global result, are the foundation of the *bulk synchronous* pattern, which cyclically alternates phases of pure local calculation with phases of global state updating.

Using collective communications instead of classic point-to-point sends (`MPI_Send`/`MPI_Recv`) offers three irreplaceable advantages:

1. **Algorithmic Efficiency:** Replacing a send from the root to $N$ nodes with a loop of `MPI_Send` requires linear time $O(N)$. Collective functions instead internally implement tree algorithms (e.g., binomial trees) that drop the complexity to $O(\log_2 N)$, making the operations intrinsically faster and more scalable.
2. **Topological Optimization (Hardware Awareness):** Collective calls are not simple "macros" for send/recv loops. The MPI library recognizes the underlying architecture and routes data transparently, exploiting shared memory if the processes are on the same physical node, or hardware-level multicast protocols on the network.
3. **Code Safety and Cleanliness:** Manually managing message routing exponentially increases the risk of bottlenecks and *deadlocks*. Collectives guarantee implicit and optimized synchronization, reducing lines of code and control overhead.

In summary, all the functions we have explored are divided into three large macro-families:
* **Pure synchronization** (`MPI_Barrier`): to temporally align the nodes.
* **Data movement** (`MPI_Bcast`, `MPI_Scatter(v)`, `MPI_Gather(v)`, `MPI_Allgather`, `MPI_Alltoall`): to distribute, gather, or exchange information without altering it.
* **Computation and Reduction** (`MPI_Reduce`, `MPI_Allreduce`, `MPI_Scan`): to combine network aggregation with global mathematical and logical calculations.
