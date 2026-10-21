#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>

/* Vectorizable membership check (no early exit) */
int is_inside(double px, double py, const double* x, const double* y, const double* delta_x, const double* delta_y, int n) {
    int out_count = 0;

    // When compiled with OpenMP SIMD support (e.g. -fopenmp-simd), the
    // following pragma enables vectorization of the loop; without the flag
    // it is simply ignored by the compiler.
    #pragma omp simd reduction(+:out_count)
    for (int i = 0; i < n; i++) {
        // Use the precomputed deltas to avoid redundant subtractions on the vertices
        double cross_product = delta_x[i] * (py - y[i]) - delta_y[i] * (px - x[i]);
        if (cross_product < 0) {
            out_count++;
        }
    }

    // The point is inside only if it is on the left of every edge
    return (out_count == 0);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    long long M = 0; // Total number of points
    int N = 0;       // Number of vertices

    // Pointers for optimized memory management
    double *buffer = NULL;
    double *x = NULL;
    double *y = NULL;
    FILE *in = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0 && argc != 3) {
        fprintf(stderr, "Usage: %s <number_of_points> <input_file>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // All processes can read the number of points from argv[1]
    if (argc >= 3) {
        M = atoll(argv[1]);
    }

    // Only the Master (rank 0) opens the file and reads the number of vertices
    if (my_rank == 0) {
        in = fopen(argv[2], "r");
        if (!in || fscanf(in, "%d", &N) != 1 || N < 3) {
            fprintf(stderr, "ERROR: invalid file\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast N so that every process can allocate its buffer
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Single contiguous buffer of size 2N: x is the first half, y the second.
    // This allows sending the whole geometry with a single MPI_Bcast.
    buffer = malloc(2 * (size_t)N * sizeof(double));
    x = buffer; y = buffer + N;

    // Only the Master reads the vertex coordinates from the file
    if (my_rank == 0) {
        for (int i = 0; i < N; i++)
            if (fscanf(in, "%lf %lf", &x[i], &y[i]) != 2) {
                fprintf(stderr, "ERROR: unable to read vertex %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
             }
        fclose(in);
    }


    // Send all the vertices to every process with a single collective operation
    MPI_Bcast(buffer, 2 * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 6. ALL nodes compute the Bounding Box locally */
    double min_x = x[0], max_x = x[0];
    double min_y = y[0], max_y = y[0];
    for (int i = 1; i < N; i++) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }
    double area_bb = (max_x - min_x) * (max_y - min_y);

    /* 7. ALL nodes precompute the edge deltas locally */
    double *delta_x = (double*)malloc(N * sizeof(double));
    double *delta_y = (double*)malloc(N * sizeof(double));
    assert(delta_x != NULL && delta_y != NULL);

    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N; // Circular closure of the polygon
        delta_x[i] = x[j] - x[i];
        delta_y[i] = y[j] - y[i];
    }

    /* 8. Workload partitioning (MPI load balancing) */
    long long local_M = M / comm_sz;
    if (my_rank == 0) {
        local_M += M % comm_sz; // The Master takes care of the remainder
    }


    /* 9. Thread-safe random seed setup for each node */
    // The seed depends on the time and on the rank, so that each process
    // generates a different sequence
    unsigned short rng_state[3];
    int seed = time(NULL) + my_rank * 11;
    rng_state[0] = seed & 0xFFFF;
    rng_state[1] = (seed >> 16) & 0xFFFF;
    rng_state[2] = seed & 0xFFFF;

    long long local_K = 0;  // Points inside the polygon found by this process
    long long global_K = 0; // Total across all processes (valid on the Master)

    // Synchronize all processes so that the timing starts at the same moment
    MPI_Barrier(MPI_COMM_WORLD);
    const double tstart = MPI_Wtime();

    /* 10. Monte Carlo simulation loop */
    for (long long i = 0; i < local_M; i++) {
        double px = min_x + erand48(rng_state) * (max_x - min_x);
        double py = min_y + erand48(rng_state) * (max_y - min_y);

        if (is_inside(px, py, x, y, delta_x, delta_y, N)) {
            local_K++;
        }
    }

    /* 11. Final reduction: sum all the local_K into the Master's global_K */
    MPI_Reduce(&local_K, &global_K, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    /* 12. Print the results (Master only) */
    if (my_rank == 0) {
        double estimated_area = area_bb * ((double)global_K / M);
        const double elapsed = MPI_Wtime() - tstart;

        printf("Number of processes: %d\n", comm_sz);
        printf("Total points (M)   : %lld\n", M);
        printf("Points inside (K)  : %lld\n", global_K);
        printf("Bounding Box area  : %f\n", area_bb);
        printf("ESTIMATED AREA     : %f\n", estimated_area);
        printf("Execution time: %f seconds\n", elapsed);
    }

    /* 13. Memory cleanup */
    free(buffer); // Frees both x and y at once
    free(delta_x);
    free(delta_y);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
