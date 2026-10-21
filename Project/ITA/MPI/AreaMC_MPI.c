#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <mpi.h>

/* Funzione di controllo vettorializzabile (Senza Early Exit) */
int is_inside(double px, double py, const double* x, const double* y, const double* delta_x, const double* delta_y, int n) {
    int out_count = 0;
    
    // Se compili con flag OpenMP per un approccio ibrido MPI+OpenMP, 
    // puoi decommentare il pragma seguente:
    #pragma omp simd reduction(+:out_count)
    for (int i = 0; i < n; i++) {
        // Usa i delta precalcolati per evitare sottrazioni ridondanti sui vertici
        double cross_product = delta_x[i] * (py - y[i]) - delta_y[i] * (px - x[i]);
        if (cross_product < 0) {
            out_count++;
        }
    }

    return (out_count == 0);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    long long M = 0; // Punti totali
    int N = 0;       // Numero vertici

    // Puntatori per la gestione ottimizzata della memoria
    double *buffer = NULL; 
    double *x = NULL;
    double *y = NULL;
    FILE *in = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0 && argc != 3) {
        fprintf(stderr, "Uso: %s <numero_punti> <file_input>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Tutti i processi possono leggere il numero di punti da argv[1]
    if (argc >= 3) {
        M = atoll(argv[1]);
    }

    if (my_rank == 0) {
        in = fopen(argv[2], "r");
        if (!in || fscanf(in, "%d", &N) != 1 || N < 3) {
            fprintf(stderr, "ERRORE: file non valido\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    buffer = malloc(2 * (size_t)N * sizeof(double));
    x = buffer; y = buffer + N;

    if (my_rank == 0) {
        for (int i = 0; i < N; i++)
            if (fscanf(in, "%lf %lf", &x[i], &y[i]) != 2) { 
                fprintf(stderr, "ERRORE: impossibile leggere il vertice %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
             }
        fclose(in);
    }


    MPI_Bcast(buffer, 2 * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 6. TUTTI i nodi calcolano localmente il Bounding Box */
    double min_x = x[0], max_x = x[0];
    double min_y = y[0], max_y = y[0];
    for (int i = 1; i < N; i++) {
        if (x[i] < min_x) min_x = x[i];
        if (x[i] > max_x) max_x = x[i];
        if (y[i] < min_y) min_y = y[i];
        if (y[i] > max_y) max_y = y[i];
    }
    double area_bb = (max_x - min_x) * (max_y - min_y);

    /* 7. TUTTI i nodi precalcolano i delta dei lati localmente */
    double *delta_x = (double*)malloc(N * sizeof(double));
    double *delta_y = (double*)malloc(N * sizeof(double));
    assert(delta_x != NULL && delta_y != NULL);
    
    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N; // Chiusura circolare del poligono
        delta_x[i] = x[j] - x[i];
        delta_y[i] = y[j] - y[i];
    }

    /* 8. Ripartizione del carico (Load Balancing MPI) */
    long long local_M = M / comm_sz;
    if (my_rank == 0) {
        local_M += M % comm_sz; // Il master si fa carico del resto
    }
    

    /* 9. Setup Thread-Safe del seed random per nodo */
    unsigned short rng_state[3];
    int seed = time(NULL) + my_rank * 11; 
    rng_state[0] = seed & 0xFFFF;
    rng_state[1] = (seed >> 16) & 0xFFFF;
    rng_state[2] = seed & 0xFFFF;

    long long local_K = 0;
    long long global_K = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double tstart = MPI_Wtime();

    /* 10. Loop di simulazione Monte Carlo */
    for (long long i = 0; i < local_M; i++) {
        double px = min_x + erand48(rng_state) * (max_x - min_x);
        double py = min_y + erand48(rng_state) * (max_y - min_y);
        
        if (is_inside(px, py, x, y, delta_x, delta_y, N)) {
            local_K++;
        }
    }

    /* 11. Riduzione finale: Somma tutti i local_K nel global_K del Master */
    MPI_Reduce(&local_K, &global_K, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    /* 12. Stampa dei risultati (Solo Master) */
    if (my_rank == 0) {
        double area_stimata = area_bb * ((double)global_K / M);
        const double elapsed = MPI_Wtime() - tstart;
        
        printf("Numero di processi : %d\n", comm_sz);
        printf("Punti totali (M)   : %lld\n", M);
        printf("Punti interni (K)  : %lld\n", global_K);
        printf("Area Bounding Box  : %f\n", area_bb);
        printf("AREA STIMATA       : %f\n", area_stimata);
        printf("Tempo di esecuzione: %f secondi\n", elapsed);
    }

    /* 13. Pulizia della memoria */
    free(buffer); // Libera contemporaneamente x e y
    free(delta_x);
    free(delta_y);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
