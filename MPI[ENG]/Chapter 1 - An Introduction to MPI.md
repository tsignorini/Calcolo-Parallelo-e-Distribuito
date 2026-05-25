# Introduction to MPI


## MPI 

> MPI is the standard used in distributed computing on supercomputers. It is based on the SPMD (*Single Program Multiple Data*) paradigm, in which the same program is executed simultaneously by distinct processes on different machines. Unlike OpenMP, it is not based on directives but is a software library that offers low-level control over system components, ensuring high optimization of computation. Its use is necessary when the size of the problem exceeds the capabilities of a single machine.
> 
> **Advantages:**
> * **Scalability:** It allows overcoming the memory limits of a single machine, therefore handling large-scale problems.
> * **Safety:** Absence of *data races*, given that each process has its own private memory space.
> 
> **Disadvantages:**
> * **Network Overhead:** Message exchange and synchronization occur via network subroutines, introducing latencies that can slow down execution.
> * **Communication Errors:** Since data must be sent, one is exposed to possible communication errors.
> * **Complexity:** The code becomes significantly longer, more complex, and harder to manage compared to the shared memory paradigm.

### Pseudo-Code


```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    // 1. Initializes the MPI environment, and parameters are passed because MPI adds others
    MPI_Init(&argc, &argv);

    // 2. Obtains the total number of launched processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3. Obtains the identifier (ID) of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Each process prints this message independently
    printf("Hello! I am process %d out of a total of %d\n", rank, size);

    // 4. Safely terminates the MPI environment
    MPI_Finalize();
    
    return 0;
}
```
