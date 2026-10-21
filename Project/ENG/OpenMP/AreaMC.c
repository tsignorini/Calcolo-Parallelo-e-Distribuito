#define _XOPEN_SOURCE // Required to expose erand48() in the POSIX headers
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <omp.h>
#include <time.h>

// Structure representing a 2D point
typedef struct {
    double x;
    double y;
} Point;

/*
 * Checks whether point P lies inside the convex polygon.
 *
 * The polygon is stored in Structure-of-Arrays (SoA) form:
 *   vx, vy : coordinates of the N vertices (counterclockwise order)
 *   ex, ey : precomputed edge deltas, i.e. ex[i] = vx[i+1] - vx[i],
 *            ey[i] = vy[i+1] - vy[i] (indices taken modulo N)
 *
 * For each edge, the 2D cross product (edge x (P - vertex)) is computed.
 * It is >= 0 when P is on the left of (or on) the edge, and < 0 when P is on
 * the right. P is inside the polygon only if it is on the left of every edge.
 */
bool is_inside(Point P, const double* vx, const double* vy,
                             const double* ex, const double* ey, int N) {
    int i = 0;

    // Main loop: edges are processed in blocks of 16 so that the inner loop
    // can be vectorized with SIMD instructions (no branches inside the block)
    for (; i + 16 <= N; i += 16) {
        int out = 0; // Number of edges in this block that have P on their right
        #pragma omp simd reduction(+:out)
        for (int k = 0; k < 16; k++)
            out += (ex[i+k]*(P.y - vy[i+k]) - ey[i+k]*(P.x - vx[i+k])) < 0.0;

        // If P is on the wrong side of at least one edge, it is outside
        if (out) return false;
    }

    // Tail loop: remaining edges (fewer than 16), checked one at a time
    for (; i < N; i++)
        if (ex[i]*(P.y - vy[i]) - ey[i]*(P.x - vx[i]) < 0.0) return false;

    // P is on the left of all edges: it is inside the polygon
    return true;
}


int main(int argc, char* argv[]) {
    // Default workload (10 million points) if no argument is given
    long long M = 10000000;

    // If a number is passed on the command line, it overrides M
    if (argc > 1) {
        M = atoll(argv[1]);
    }


    printf("Total points to generate: %lld\n", M);

    // 1. Read the vertices from a text file
    // Expected format: first line N, then N lines with "x y" coordinates
    const char* filename = "input.txt";
    FILE* infile = fopen(filename, "r");
    if (!infile) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return 1;
    }

    // Read and validate the number of vertices
    int N;
    if (fscanf(infile, "%d", &N) != 1 || N < 4 || N > 1000) {
        fprintf(stderr, "Error: Invalid number of vertices (must be between 4 and 1000).\n");
        if (infile) fclose(infile);
        return 1;
    }

    // Dynamic allocation of the vertex and edge-delta arrays
    double *vx = (double*)malloc(N * sizeof(double));
    double *vy = (double*)malloc(N * sizeof(double));
    double *ex = (double*)malloc(N * sizeof(double));
    double *ey = (double*)malloc(N * sizeof(double));
    if (!vx || !vy || !ex || !ey) {
        fprintf(stderr, "Error: Unable to allocate memory for the vertices.\n");
        fclose(infile);
        return 1;
    }

    // Initialize the Bounding Box with the extreme values of double,
    // so that the first vertex read always updates the limits
    double x_min = DBL_MAX, x_max = -DBL_MAX;
    double y_min = DBL_MAX, y_max = -DBL_MAX;


    // Read the vertices and compute the Bounding Box on the fly
    for (int i = 0; i < N; i++) {
        if (fscanf(infile, "%lf %lf", &vx[i], &vy[i]) != 2) {
            fprintf(stderr, "Error in the format of vertex %d.\n", i);
            free(vx); free(vy); free(ex); free(ey);
            fclose(infile);
            return 1;
        }

        if (vx[i] < x_min) x_min = vx[i];
        if (vx[i] > x_max) x_max = vx[i];
        if (vy[i] < y_min) y_min = vy[i];
        if (vy[i] > y_max) y_max = vy[i];
    }
    fclose(infile);

    // Precompute the edge deltas (dx, dy): they are constant throughout the
    // simulation, so computing them once avoids repeating the subtraction
    // for every generated point. The last edge closes the polygon (j wraps to 0).
    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N;
        ex[i] = vx[j] - vx[i];
        ey[i] = vy[j] - vy[i];
    }

    // Area of the Bounding Box (the sampling domain)
    double area_box = (x_max - x_min) * (y_max - y_min);

    // 2. Monte Carlo simulation parameters
    long long K = 0;        // Counter of the points that fall inside the polygon

    //printf("Starting Monte Carlo simulation on %lld points...\n", M);

    // Get the current system time (used to seed the random generators)
    time_t current_time = time(NULL);

    // Time measurement (start)
    double start_time = omp_get_wtime();

    // 3. OpenMP parallel block
    // reduction(+:K): each thread keeps a private copy of K, and the copies
    // are summed at the end of the region (no data races on the counter)
    #pragma omp parallel reduction(+:K)
    {
        // Thread-safe random generator (erand48 needs a state of 3 unsigned shorts)
        // The state is declared inside the parallel region, so it is private to each thread
        unsigned short rng_state[3];
        int thread_id = omp_get_thread_num();



        // Initialize the seed by mixing the time and the thread ID,
        // so that each thread produces a different sequence
        rng_state[0] = (unsigned short)(current_time + thread_id);
        rng_state[1] = (unsigned short)(current_time ^ (thread_id * 8121));
        rng_state[2] = 0x330E;

        // Distribute the loop iterations among the CPU threads
        #pragma omp for
        for (long long i = 0; i < M; i++) {
            Point P;
            // erand48 returns a double in [0.0, 1.0): scale it to the Bounding Box
            P.x = x_min + erand48(rng_state) * (x_max - x_min);
            P.y = y_min + erand48(rng_state) * (y_max - y_min);

            // Check whether the generated point lies inside the polygon
            if (is_inside(P, vx, vy, ex, ey, N)) {
                K++;
            }
        }
    }

    // Time measurement (end)
    double end_time = omp_get_wtime();

    // 4. Area estimate: (fraction of points inside) * (Bounding Box area)
    double estimated_area = area_box * ((double)K / (double)M);

    // 5. Print the formatted results
    printf("------------------------------------------\n");
    printf("Number of vertices (N): %d\n", N);
    printf("Bounding Box          : [%.2f, %.2f] x [%.2f, %.2f]\n", x_min, x_max, y_min, y_max);
    printf("Bounding Box area     : %f\n", area_box);
    printf("Points inside (K)     : %lld / %lld\n", K, M);
    printf("ESTIMATED FIGURE AREA : %f\n", estimated_area);
    printf("Execution time        : %f seconds\n", end_time - start_time);
    printf("------------------------------------------\n");

    // Release the allocated memory
    free(vx); free(vy); free(ex); free(ey);
}
