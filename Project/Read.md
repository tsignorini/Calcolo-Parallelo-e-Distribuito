# Instructions / Istruzioni

---

## 🇮🇹 Italiano

### 1. Generatore di Vertici
**Compilazione:**
```bash
gcc -O2 vertices_gen.c -o vertices_gen -lm
```

---

### 2. Versione OpenMP

**Compilazione:**
```bash
gcc-16 -O2 -fopenmp AreaMC.c -o AreaMC -lm
```

**Esecuzione:**
```bash
./AreaMC 50000000 OMP_NUM_THREADS=8
```

**Benchmark di Scaling:**
- **Strong Scaling:**
  ```bash
  ./strong_scaling.sh
  ```
- **Weak Scaling:**
  ```bash
  ./weak_scaling.sh
  ```

---

### 3. Versione MPI

**Compilazione:**
- Standard:
  ```bash
  mpicc -O2 AreaMC_MPI.c -o AreaMC_MPI
  ```
- Con ottimizzazioni SIMD / Vettorializzazione:
  ```bash
  mpicc -O2 -fopenmp-simd -march=native AreaMC_MPI.c -o AreaMC_MPI
  ```

**Esecuzione:**
```bash
mpirun -np 4 ./AreaMC_MPI 50000000 input.txt
```

**Benchmark di Scaling:**
- **Strong Scaling:**
  ```bash
  ./strong_scaling_MPI.sh
  ```
- **Weak Scaling:**
  ```bash
  ./weak_scaling_MPI.sh
  ```

---

## 🇬🇧 English

### 1. Vertex Generator
**Compilation:**
```bash
gcc -O2 vertices_gen.c -o vertices_gen -lm
```

---

### 2. OpenMP Version

**Compilation:**
```bash
gcc-16 -O2 -fopenmp AreaMC.c -o AreaMC -lm
```

**Execution:**
```bash
./AreaMC 50000000 OMP_NUM_THREADS=8
```

**Scaling Benchmarks:**
- **Strong Scaling:**
  ```bash
  ./strong_scaling.sh
  ```
- **Weak Scaling:**
  ```bash
  ./weak_scaling.sh
  ```

---

### 3. MPI Version

**Compilation:**
- Standard:
  ```bash
  mpicc -O2 AreaMC_MPI.c -o AreaMC_MPI
  ```
- With SIMD / Vectorization optimizations:
  ```bash
  mpicc -O2 -fopenmp-simd -march=native AreaMC_MPI.c -o AreaMC_MPI
  ```

**Execution:**
```bash
mpirun -np 4 ./AreaMC_MPI 50000000 input.txt
```

**Scaling Benchmarks:**
- **Strong Scaling:**
  ```bash
  ./strong_scaling_MPI.sh
  ```
- **Weak Scaling:**
  ```bash
  ./weak_scaling_MPI.sh
  ```
