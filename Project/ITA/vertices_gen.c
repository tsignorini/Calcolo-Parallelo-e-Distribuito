#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    int N = 300;
    double R = 10.0; // Raggio della circonferenza
    
    FILE *file = fopen("input.txt", "w");
    if (file == NULL) {
        printf("Errore nell'apertura del file.\n");
        return 1;
    }
    
    // Prima riga: numero di vertici
    fprintf(file, "%d\n", N);
    
    // Generazione dei vertici in senso antiorario
    for (int i = 0; i < N; i++) {
        double theta = (2.0 * M_PI * i) / N;
        double x = R * cos(theta);
        double y = R * sin(theta);
        fprintf(file, "%.6f %.6f\n", x, y);
    }
    
    fclose(file);
    printf("File 'input.txt' generato con successo con %d vertici.\n", N);
    return 0;
}