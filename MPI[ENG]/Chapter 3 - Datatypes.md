# MPI Datatypes

## MPI Datatypes

In MPI, any communication operation (send or receive) requires defining the data through a specific triplet: the starting memory address (`address`), the number of elements to transmit (`count`), and the data type (`datatype`). 

In addition to the predefined types that map standard C language types (like `MPI_INT` for `int`s or `MPI_DOUBLE` for `double`s), MPI offers the highly powerful feature of **Derived Datatypes**. This feature allows overcoming one of the physical limits of memory: the need to have contiguous data for network communications. MPI indeed allows defining custom data types formed by blocks spaced in a regular or irregular manner, or composed of heterogeneous types (structs).

To understand the vital need for this abstraction, let's analyze a classic parallel computing problem: the subdivision of a two-dimensional domain.

#### The Problem: Column Exchange (Ghost Cells)
Let's imagine a two-dimensional matrix divided into vertical blocks among various processes. At each iteration, a process must send its outermost column to the neighboring process. 
In the C language, matrices are stored by rows (*row-wise*). This means that, while the elements of a row are physically contiguous in memory, **the elements of a column are not**. How can we send this scattered data in a single block?

Here is the evolution of the solutions, from worst to best:

**1. The BAD solution** The most naive approach consists of sending every single element of the column via individual calls to the `MPI_Send` (or `MPI_Isend`) function. 
* *Why it is bad:* Executing tens or hundreds of network calls to send a single scalar each generates an enormous overhead (latency/start-up time), literally killing the performance of the parallel program.

**2. The UGLY solution** The programmer tries to remedy the network overhead by allocating a temporary array (buffer). The scattered data of the column is read one by one via a `for` loop and copied (packed) into this temporary array so that it becomes contiguous. At this point, a single `MPI_Send` of the buffer is made. The receiver does the exact opposite: it receives the buffer and unpacks it into the destination column.
* *Why it is ugly:* Although it reduces network calls, this solution wastes memory (to allocate temporary buffers) and CPU cycles (to manually copy and re-copy data from one memory area to another).

**3. The OPTIMAL solution** The elegant solution offered by the standard consists of **defining a new MPI datatype** that describes exactly the shape of the column in memory, and then passing it directly to a single `MPI_Send` call.
Using functions for building Derived Datatypes, such as `MPI_Type_vector()` (which allows defining vectors of elements separated by a constant stride), the programmer "teaches" MPI how to read the column.
* *Why it is the best solution:* There are no temporary buffers nor wasted memory. The MPI library, now knowing the exact structure of the data in memory, will handle extracting and transmitting the necessary bytes internally and in an optimized manner.

Here is the section dedicated to Derived Datatypes, written maintaining the dry and easy-to-consult style, ready to be added to your notes in Markdown:

### The 4 Derived Datatypes

The MPI library provides specific functions to overcome the limit of memory contiguity, allowing the construction of four main categories of custom datatypes starting from basic types (or from other derived types created previously).

#### 1. Contiguous
Creates a contiguous block composed of multiple copies of an existing datatype.
* **Signature:** `int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
* **Parameters:** * `count`: number of elements in the block.  
    * `oldtype`: the basic datatype of the elements.  
    * `newtype`: pointer to the newly created datatype.  

#### 2. Vector (Regularly spaced vector)
Defines an array of blocks separated by a constant stride. It is ideal for extracting columns from matrices stored by rows.
* **Signature:** `int MPI_Type_vector(int count, int blocklen, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
* **Parameters:** * `count`: total number of blocks.  
    * `blocklen`: number of elements contained in each block.  
    * `stride`: number of elements separating the start of one block from the start of the next.  
    * `oldtype`: the datatype of the elements.  

#### 3. Indexed (Irregular spacing)
Allows defining a set of blocks having different lengths and separated by irregular offsets (displacements).
* **Signature:** `int MPI_Type_indexed(int count, const int array_of_blklen[], const int array_of_displ[], MPI_Datatype oldtype, MPI_Datatype *newtype)`.  
* **Parameters:** * `count`: total number of blocks.  
    * `array_of_blklen`: array containing the specific length of each individual block.  
    * `array_of_displ`: array containing the displacements (measured in number of elements) of the start of each block relative to the start of the data structure.  

#### 4. Struct (Heterogeneous Structure)
It is the most general and flexible form. It allows defining blocks of different sizes, separated by irregular spacing and, above all, composed of **different datatypes** (replicating C `struct`s).
* **Signature:** `int MPI_Type_create_struct(int count, int *array_of_blklen, MPI_Aint *array_of_displ, MPI_Datatype *array_of_types, MPI_Datatype *newtype)`.  
* **Key parameters (differences with Indexed):** * `array_of_displ`: unlike Indexed, offsets here are strictly measured in **bytes** and require the use of the specific `MPI_Aint` type.  
    * `array_of_types`: array containing the specific datatypes (`MPI_Datatype`) for each block.  


### Initialization and Termination

The simple declaration or structural creation of a new `MPI_Datatype` is not sufficient to be able to use it within network communication functions. The lifecycle of the datatype requires the use of two fundamental functions to interface with the MPI system:

1.  **Commit (Initialization): `MPI_Type_commit(...)`** * **Signature:** `int MPI_Type_commit(MPI_Datatype *datatype)`.  
    * **Description:** This function formally "registers" the new derived type in the MPI runtime system, compiling its memory map. **It must strictly be invoked** before using the `datatype` in any communication operation (e.g., `MPI_Send` or `MPI_Recv`).  

2.  **Free (Termination): `MPI_Type_free(...)`** * **Signature:** `int MPI_Type_free(MPI_Datatype *datatype)`.  
    * **Description:** When the derived type is no longer needed, this function must be invoked to safely deallocate the internal memory resources that MPI had reserved to describe the object, preventing *memory leaks*.  


### MPI_Type_contiguous

The `MPI_Type_contiguous()` function is the simplest method to create a new derived datatype. It allows grouping a fixed number (`count`) of adjacent memory elements of an already existing type (`oldtype`) to form a single contiguous logical block. 

This construct is perfect for the C language, where matrices are allocated in memory by rows (*row-major order*), making the elements of a single row physically adjacent to each other. Instead of sending the individual elements or passing a high `count` to the network function, we can define a datatype that represents exactly the logical concept of a "row" and transmit it as a single entity.

#### Example: Sending a row of a matrix
Suppose we have a 4x4 square matrix `a` composed of `float`s and we want to send the entire third row (i.e., the one at index 2, starting from the element `a`).

```c
int count = 4; /* Number of contiguous elements (the row length) */
MPI_Datatype rowtype;

/* 1. Definition of the new type formed by 4 contiguous MPI_FLOATs */
MPI_Type_contiguous(count, MPI_FLOAT, &rowtype);

/* 2. Registration (Commit) of the type in the MPI system before use */
MPI_Type_commit(&rowtype);

/* 3. Sending the third row (index 2) */
/* We pass the starting address of the row &a and ask MPI 
   to send exactly 1 block of the new 'rowtype' type */
MPI_Send(&a[2][0], 1, rowtype, dest, tag, MPI_COMM_WORLD);

/* ... at the end of the program ... */
MPI_Type_free(&rowtype);
```

In this way, the send function knows that it must read exactly 4 elements of type `float` starting from the memory address `&a` and will transmit them in a single optimized block.

Here is the section dedicated to `MPI_Type_vector()`, with the detailed explanation and the matrix example taken directly from the slides, ready to be copied in Markdown format:


### MPI_Type_vector

The `MPI_Type_vector()` function allows creating a new datatype consisting of an array of blocks separated from each other by a regular spacing (stride). 

The classic and most important use case for this function in the C language is the **extraction of a column from a matrix**. Since in C matrices are stored by rows (*row-major order*), the elements of a row are contiguous, but those of a column are separated in memory by a number of elements equal to the width of the row itself. `MPI_Type_vector()` allows instructing MPI on how to "jump" from one row to another to read only the data of the desired column.

#### Key Parameters
The function signature is `MPI_Type_vector(count, blocklen, stride, oldtype, &newtype)`:  
* **`count`**: total number of blocks to read.  
* **`blocklen`**: number of elements that make up a single block.  
* **`stride`**: the "stride", i.e., the number of elements separating the start of one block from the start of the next.  

#### Example: Sending a column of a matrix
Let's consider a `float` matrix `a` and suppose we want to extract and send the **second column** (the one at index 1).

The matrix in memory has the following contiguous values:
`[ 1.0, 2.0, 3.0, 4.0 | 5.0, 6.0, 7.0, 8.0 | 9.0, 10.0, 11.0, 12.0 | 13.0, 14.0, 15.0, 16.0 ]`

We want to extract the column composed of the values: `2.0`, `6.0`, `10.0`, `14.0`.
To do this, we define the vector geometry:
* `count = 4` (we want 4 elements in total, one for each row).
* `blocklen = 1` (each element of the column is a single float).
* `stride = 4` (to go from `2.0` to `6.0` we have to make a jump of 4 positions in memory, equal to the row width).

```c
/* Definition of the parameters to extract a single column from a 4x4 matrix */
int count = 4;
int blocklen = 1;
int stride = 4;
MPI_Datatype columntype;

/* 1. Creation of the derived vector type */
MPI_Type_vector(count, blocklen, stride, MPI_FLOAT, &columntype);

/* 2. Registration (Commit) of the type in the MPI system */
MPI_Type_commit(&columntype);

/* 3. Sending the second column */
/* We pass a as the starting address, which is the value 2.0.
   From there, MPI will extract 1 block of type 'columntype', jumping by 4 */
MPI_Send(&a[0][1], 1, columntype, dest, tag, MPI_COMM_WORLD);

/* ... at the end of the program ... */
MPI_Type_free(&columntype);
```

In this way, with a single `MPI_Send` instruction and without having to copy the data into temporary buffers, the library will exactly extract and send the values `2.0, 6.0, 10.0, 14.0`.

### MPI_Type_indexed

The `MPI_Type_indexed()` function allows creating a new datatype consisting of a set of blocks of elements that have both irregular lengths and irregular offsets (displacements) between them.

Unlike `MPI_Type_vector`, where the block size and stride are constant, here the programmer must explicitly provide two arrays: one to specify the length of each block and another to indicate its position (displacement) relative to the start of the buffer. All displacements are measured in *number of elements* (not in bytes) of the original datatype.

#### Key Parameters
The function signature is `MPI_Type_indexed(int count, const int array_of_blklen[], const int array_of_displ[], MPI_Datatype oldtype, MPI_Datatype *newtype)`:
* **`count`**: total number of blocks making up the new type.
* **`array_of_blklen`**: array containing the specific length (number of elements) for each individual block.
* **`array_of_displ`**: array containing the offset (in number of elements) of each block relative to the starting address.

#### Example: Extracting irregular blocks from an array
Let's consider an array `a` of 16 `float`s with sequential values from `1.0` to `16.0`. We want to extract three specific and heterogeneous fragments:
1. A first block of **1 element** starting from index **2** (value `3.0`).
2. A second block of **3 elements** starting from index **5** (values `6.0, 7.0, 8.0`).
3. A third block of **4 elements** starting from index **12** (values `13.0, 14.0, 15.0, 16.0`).

```c
/* Definition of the parameters for the 3 irregular blocks */
int count = 3;
int blklens[] = {1, 3, 4};  /* Array of block lengths */
int displs[] = {2, 5, 12};  /* Array of starting indices */
MPI_Datatype newtype;

/* 1. Creation of the indexed derived type */
MPI_Type_indexed(count, blklens, displs, MPI_FLOAT, &newtype);

/* 2. Registration (Commit) of the type in the MPI system */
MPI_Type_commit(&newtype);

/* 3. Sending the data */
/* By passing the starting address &a, MPI will apply the displacements in displs[]
   and will exactingly extract the 3 blocks described in blklens[] */
MPI_Send(&a[0], 1, newtype, dest, tag, MPI_COMM_WORLD);

/* Reception on the destination side (the process will receive 8 contiguous elements in total) */
/* MPI_Recv(&b[0], 8, MPI_FLOAT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); */

/* ... at the end of the program ... */
MPI_Type_free(&newtype);
```

In this example, with a single `MPI_Send` instruction where we specify to send `1` single logical block of type `newtype`, the library will navigate the original array and extract exactly the 8 desired values for transmission (`3.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0`). The receiver will receive these 8 values "compacted" in its receive array.


### Combining Derived Types (Composability)

One of the most powerful features of MPI *Derived Datatypes* is their total composability. The `oldtype` parameter required by the creation functions (such as `MPI_Type_contiguous`, `MPI_Type_vector`, and `MPI_Type_indexed`) does not necessarily have to be a predefined type (e.g., `MPI_FLOAT` or `MPI_INT`), but can be **another derived type previously created by the programmer**. 

This "Russian doll" approach allows mapping and transmitting hierarchical and complex data structures with extreme elegance.

#### Example: A vector of vectors
In this example, we first create a logical vector (`vec`) composed of basic `MPI_FLOAT` elements, and then we use it in turn as a building block (`oldtype`) to create an even more complex type (`vecvec`).

```c
int count, blocklen, stride; 
MPI_Datatype vec, vecvec;

/* 1. Let's create the first derived type (a base vector of floats) */
count = 2; blocklen = 2; stride = 3;
MPI_Type_vector(count, blocklen, stride, MPI_FLOAT, &vec);
MPI_Type_commit(&vec);

/* 2. Let's use the newly created 'vec' as 'oldtype' for a new vector */
count = 2; blocklen = 1; stride = 3;
MPI_Type_vector(count, blocklen, stride, vec, &vecvec);
MPI_Type_commit(&vecvec);

/* At the end of use, remember to free both types */
/* MPI_Type_free(&vec); */
/* MPI_Type_free(&vecvec); */
```

### MPI_Type_create_struct

The `MPI_Type_create_struct()` function is the most general and flexible form for creating derived datatypes. It allows defining a new type composed of a set of irregularly spaced blocks and, above all, made up of **existing datatypes that are different from each other**.

This construct is fundamental because it directly maps C language **`struct`**s into MPI. In fact, it allows transmitting an entire heterogeneous data structure across the network in a single optimized communication operation, avoiding having to unpack the fields or send multiple messages.

#### Key Parameters
Unlike other functions (like `MPI_Type_indexed`), where the spacings are measured in "number of elements", here the blocks have base types of different sizes. Consequently, all offsets must be strictly measured in **bytes**. To do this, a special integer datatype offered by MPI is used: `MPI_Aint`.

The function signature is `int MPI_Type_create_struct(int count, int *array_of_blklen, MPI_Aint *array_of_displ, MPI_Datatype *array_of_types, MPI_Datatype *newtype)`:
* **`count`**: total number of blocks making up the structure.
* **`array_of_blklen`**: array containing the number of elements for each individual block.
* **`array_of_displ`**: array of type `MPI_Aint` containing the displacement (offset) **in bytes** of each block relative to the starting address.
* **`array_of_types`**: array containing the specific datatypes (`MPI_Datatype`) for each block.

#### Example: Sending a heterogeneous C struct
Let's consider a `particle_t` structure used to represent a particle in a simulation, composed of coordinates/velocity (4 `float`s) and integer metadata (2 `int`s):

```c
typedef struct {
    float x, y, z, v;
    int n, t;
} particle_t;
```

To teach MPI how to map this struct in memory, we must describe it as being formed by 2 distinct blocks:
1. A first block of **4 elements** of type `MPI_FLOAT` with a 0-byte offset.
2. A second block of **2 elements** of type `MPI_INT` whose byte offset will start exactly where the 4 floats end.

```c
/* Parameters to describe the geometry of the struct */
int count = 2;
int blklens[2];
MPI_Aint displs[2], lb, extent;
MPI_Datatype oldtypes[2], newtype;

/* Configuration of the 1st block: 4 floats (x, y, z, v) */
oldtypes[0] = MPI_FLOAT;
blklens[0] = 4;
displs[0] = 0; /* The first block starts from the beginning (offset 0) */

/* We ask MPI to calculate the extent in bytes of the float type */
MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);

/* Configuration of the 2nd block: 2 ints (n, t) */
oldtypes[1] = MPI_INT;
blklens[1] = 2;
/* The byte offset of the second block is equal to the extent of 4 floats */
displs[1] = 4 * extent;

/* 1. Creation of the struct derived type */
MPI_Type_create_struct(count, blklens, displs, oldtypes, &newtype);

/* 2. Registration (Commit) of the type in the MPI system */
MPI_Type_commit(&newtype);

/* 3. Use for sending */
/* Now the struct can be sent entirely with a single instruction */
/* particle_t my_particle; */
/* MPI_Send(&my_particle, 1, newtype, dest, tag, MPI_COMM_WORLD); */

/* ... at the end of the program ... */
MPI_Type_free(&newtype);
```

### Conclusion: Optimization and final considerations on MPI

To close the topic on **Derived Datatypes** and make a final assessment of MPI programming, we can summarize the key concepts into two major aspects: memory optimization and the general challenges of *message passing*.

#### The crucial role of Derived Types
*Derived Datatypes* (Contiguous, Vector, Indexed, and Struct) represent the "optimal" tool provided by the MPI standard to exchange non-contiguous data in memory. 
By exploiting them, the programmer manages to:
1. **Avoid network overhead:** By warding off the "bad" solution, i.e., sending tens of tiny messages for every single fragmented data point.
2. **Save CPU and Memory:** By avoiding the "ugly" solution, i.e., the waste of machine cycles needed to manually copy and pack/unpack data into temporary arrays before sending.
3. **Map complex structures:** By creating hierarchies of composable types that faithfully reflect the heterogeneous data structures of the C language.

## Final considerations on the MPI model
Broadening the gaze to the entire MPI ecosystem, it is important to recognize the nature of this standard. *Message passing* is a fundamental programming model, but of a **very low level and decidedly "heavyweight"**.

Writing MPI programs requires a considerable effort from the developer, and presents specific challenges:
* **Verbose code:** A large part of the development cost stems from the amount of local code needed to manage network logic. Very often, the code dedicated solely to communication exceeds half of the total lines of the program.
* **Design difficulty:** Designing an MPI application that is adaptable, flexible, and completely bug-free (like the dreaded *deadlocks*) is extremely complex (*tough to get right*).

However, despite these intrinsic development difficulties, MPI remains the **absolute programming model of choice for scalability**. Its immense spread and global adoption are guaranteed by its **portability** across different systems. Learning to master tools like *collective communications* and *Derived Datatypes* is the price to pay to be able to extract every last *flop* of computing power from modern distributed supercomputers.
