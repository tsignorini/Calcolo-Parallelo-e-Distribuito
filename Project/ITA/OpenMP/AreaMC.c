#define _XOPEN_SOURCE // Necessario per esporre erand48() negli header POSIX
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <omp.h>
#include <time.h>

// Struttura per rappresentare un punto 2D
typedef struct {
    double x;
    double y;
} Point;

bool is_inside(Point P, const double* vx, const double* vy,
                             const double* ex, const double* ey, int N) {
    int i = 0;
    for (; i + 16 <= N; i += 16) {
        int out = 0;
        #pragma omp simd reduction(+:out)
        for (int k = 0; k < 16; k++)
            out += (ex[i+k]*(P.y - vy[i+k]) - ey[i+k]*(P.x - vx[i+k])) < 0.0;
        if (out) return false;
    }
    for (; i < N; i++)
        if (ex[i]*(P.y - vy[i]) - ey[i]*(P.x - vx[i]) < 0.0) return false;
    return true;
}


int main(int argc, char* argv[]) {
    // Carico base (es. 10 milioni di punti) se non passi argomenti
    long long M = 10000000; 

    // Se passi un numero da terminale, sovrascrive M
    if (argc > 1) {
        M = atoll(argv[1]);
    }
    

    printf("Punti totali da generare: %lld\n", M);
    // 1. Lettura dei vertici da file di testo
    const char* filename = "input.txt";
    FILE* infile = fopen(filename, "r");
    if (!infile) {
        fprintf(stderr, "Errore: Impossibile aprire il file %s\n", filename);
        return 1;
    }

    int N;
    if (fscanf(infile, "%d", &N) != 1 || N < 4 || N > 1000) {
        fprintf(stderr, "Errore: Numero di vertici non valido (deve essere tra 4 e 1000).\n");
        if (infile) fclose(infile);
        return 1;
    }

    // Allocazione dinamica dell'array dei vertici
    double *vx = (double*)malloc(N * sizeof(double));
    double *vy = (double*)malloc(N * sizeof(double));
    double *ex = (double*)malloc(N * sizeof(double));
    double *ey = (double*)malloc(N * sizeof(double));
    if (!vx || !vy || !ex || !ey) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per i vertici.\n");
        fclose(infile);
        return 1;
    }

    // Inizializzazione Bounding Box con i limiti massimi e minimi dei double
    double x_min = DBL_MAX, x_max = -DBL_MAX;
    double y_min = DBL_MAX, y_max = -DBL_MAX;


    for (int i = 0; i < N; i++) {
        if (fscanf(infile, "%lf %lf", &vx[i], &vy[i]) != 2) {
            fprintf(stderr, "Errore nella formattazione del vertice %d.\n", i);
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

    // Precalcolo dei delta dei lati (dx, dy), costanti per tutta la simulazione
    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N;
        ex[i] = vx[j] - vx[i];
        ey[i] = vy[j] - vy[i];
    }

    double area_box = (x_max - x_min) * (y_max - y_min);

    // 2. Parametri della simulazione Monte Carlo
    long long K = 0;        // Contatore dei punti "caduti" all'interno

    //printf("Avvio simulazione Monte Carlo su %lld punti...\n", M);

    // Otteniamo il tempo di sistema attuale
    time_t current_time = time(NULL);
    
    // Misura del tempo (Start)
    double start_time = omp_get_wtime();

    // 3. Blocco Parallelo OpenMP
    #pragma omp parallel reduction(+:K)
    {
        // Generatore casuale Thread-Safe (erand48 richiede uno stato di 3 unsigned short)
        unsigned short rng_state[3];
        int thread_id = omp_get_thread_num();
        
        

        // Inizializziamo il seed mescolando il tempo e l'ID del thread
        rng_state[0] = (unsigned short)(current_time + thread_id);
        rng_state[1] = (unsigned short)(current_time ^ (thread_id * 8121));
        rng_state[2] = 0x330E;

        // Distribuzione del ciclo for sui vari thread CPU
        #pragma omp for
        for (long long i = 0; i < M; i++) {
            Point P;
            // erand48 restituisce un double tra [0.0, 1.0)
            P.x = x_min + erand48(rng_state) * (x_max - x_min);
            P.y = y_min + erand48(rng_state) * (y_max - y_min);

            // Controllo del punto generato
            if (is_inside(P, vx, vy, ex, ey, N)) {
                K++;
            }
        }
    }

    // Misura del tempo (End)
    double end_time = omp_get_wtime();

    // 4. Stima dell'Area
    double estimated_area = area_box * ((double)K / (double)M);

    // 5. Stampa dei risultati formattati
    printf("------------------------------------------\n");
    printf("Numero di vertici (N) : %d\n", N);
    printf("Bounding Box          : [%.2f, %.2f] x [%.2f, %.2f]\n", x_min, x_max, y_min, y_max);
    printf("Area Bounding Box     : %f\n", area_box);
    printf("Punti interni (K)     : %lld / %lld\n", K, M);
    printf("AREA STIMATA FIGURA   : %f\n", estimated_area);
    printf("Tempo di esecuzione   : %f secondi\n", end_time - start_time);
    printf("------------------------------------------\n");

    free(vx); free(vy); free(ex); free(ey);
}
