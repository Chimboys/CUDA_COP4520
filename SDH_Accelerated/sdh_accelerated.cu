/* ==================================================================
   
  Programmer:  Akmal Kurbanov (akmal@usf.edu)
  Optimized version of the SDH algorithm for 3D data using shared memory
  for output privatization. This approach reduces global atomic contention by
  accumulating histogram counts locally (per-block) in shared memory and then
  merging them into the global histogram.
  
  To run: `/apps/GPU_course/runScript.sh {name.cu} {#of_samples} {bucket_width} {block_size}`

 ==================================================================*/
 #include <stdio.h>
 #include <stdlib.h>
 #include <ctype.h>
 #include <math.h>
 #include <sys/time.h>
 #include <cuda.h>
 
 #define BOX_SIZE    23000   /* size of the data box on one dimension */
 
 /* Structure for each data point.
    Note: x, y, and z are declared as doubles. */
 typedef struct atomdesc {
     double x_pos;
     double y_pos;
     double z_pos;
 } atom;
 
 typedef struct hist_entry {
     unsigned long long d_cnt;  /* count may be huge */
 } bucket;
 
 bucket *histogram;         /* CPU histogram */
 long long PDH_acnt;        /* total number of data points */
 int num_buckets;           /* number of histogram bins */
 double PDH_res;            /* bucket width */
 atom *atom_list;           /* list of data points */
 
 struct timezone Idunno;
 struct timeval startTime, endTime;
 
 /*-------------------------------------------------------------
     CPU Baseline: Brute-force SDH (for correctness comparison)
 -------------------------------------------------------------*/
 int PDH_baseline() {
     int i, j, h_pos;
     double dist;
     for (i = 0; i < PDH_acnt; i++) {
         for (j = i+1; j < PDH_acnt; j++){
             double dx = atom_list[i].x_pos - atom_list[j].x_pos;
             double dy = atom_list[i].y_pos - atom_list[j].y_pos;
             double dz = atom_list[i].z_pos - atom_list[j].z_pos;
             dist = sqrt(dx*dx + dy*dy + dz*dz);
             h_pos = (int)(dist / PDH_res);
             histogram[h_pos].d_cnt++;
         } 
     }
     return 0;
 }
 
 /*-------------------------------------------------------------
     Utility functions to print and compare histograms
 -------------------------------------------------------------*/
 void output_histogram(){
     int i; 
     long long total_cnt = 0;
     for(i = 0; i < num_buckets; i++) {
         if(i % 5 == 0)
             printf("\n%02d: ", i);
         printf("%15llu ", histogram[i].d_cnt);
         total_cnt += histogram[i].d_cnt;
         if(i == num_buckets - 1)    
             printf("\n T:%lld \n", total_cnt);
         else 
             printf("| ");
     }
     printf("\n");
 }
 
 void output_histogram_gpu(bucket *hist) {
     int i; 
     long long total_cnt = 0;
     for(i = 0; i < num_buckets; i++) {
         if(i % 5 == 0)
             printf("\n%02d: ", i);
         printf("%15llu ", hist[i].d_cnt);
         total_cnt += hist[i].d_cnt;
         if(i == num_buckets - 1)    
             printf("\n T:%lld \n", total_cnt);
         else 
             printf("| ");
     }
     printf("\n");
 }
 
 void compare_histograms(bucket *cpu_hist, bucket *gpu_hist) {
     printf("\nDifference between CPU and GPU histograms:\n");
     for (int i = 0; i < num_buckets; i++) {
         long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
         if (i % 5 == 0)
             printf("\n%02d: ", i);
         printf("%15lld ", diff);
         if (i != num_buckets - 1)
             printf("| ");
     }
     printf("\n");
 }
 

 /*-------------------------------------------------------------
     Optimized approach: GPU Kernel with Output Privatization
     
     This kernel computes pairwise distances between atoms and builds 
     a histogram of distances. The main optimizations include:
     
     1. Using Shared Memory for a Local Histogram:
        - A shared memory array (shared_histogram) is allocated dynamically.
        - Each block initializes its portion of shared_histogram to zero.
        - Threads within a block update the shared histogram using atomic 
          operations, which are faster than global memory atomics.
          
     2. Reducing Global Memory Contention:
        - After computing local counts in shared memory, threads synchronize.
        - The partial results are then accumulated into the global histogram 
          with fewer, coarser atomic updates.
     
     3. Efficient Work Distribution:
        - Each thread computes its unique global thread ID and handles 
          a subset of atom pairs.
        - This ensures balanced work across threads while minimizing redundant
          computations.
     
     The combination of these strategies leads to improved performance 
     by leveraging faster shared memory accesses and reducing the overhead 
     of global memory atomics.
-------------------------------------------------------------*/

 __global__ void PDH_kernel(atom *device_atom_list, bucket *device_histogram, long long atom_count, double histogram_resolution, int num_buckets)
 {
     // Get the thread index within the block
     int thread_index_in_block = threadIdx.x;
     
     // Calculate the global thread ID across all blocks
     int global_thread_id = blockIdx.x * blockDim.x + thread_index_in_block;
   
     // Declare shared memory for the histogram
     extern __shared__ int shared_histogram[];
   
     // Initialize shared_histogram to 0, each thread handles a different bucket
     for (int bucket_index = thread_index_in_block; bucket_index < num_buckets; bucket_index += blockDim.x)
     {
         shared_histogram[bucket_index] = 0;
     }
   
     // Ensure all threads have initialized shared_histogram before proceeding
     __syncthreads();
   
     // Iterate over all atom pairs to compute distances and update the histogram
     for (int atom_index = global_thread_id + 1; atom_index < atom_count; atom_index++)
     {
         // Calculate the distance between the current atom and the atom at global_thread_id
         double dx = device_atom_list[atom_index].x_pos - device_atom_list[global_thread_id].x_pos;
         double dy = device_atom_list[atom_index].y_pos - device_atom_list[global_thread_id].y_pos;
         double dz = device_atom_list[atom_index].z_pos - device_atom_list[global_thread_id].z_pos;
         
         // Compute the Euclidean distance
         double distance = sqrt(dx * dx + dy * dy + dz * dz);
         
         // Determine the bucket index for this distance
         int bucket_index = (int)(distance / histogram_resolution);
         
         // Atomically increment the corresponding bucket in shared_histogram
         atomicAdd(&(shared_histogram[bucket_index]), 1);
     }
   
     // Synchronize threads after histogram computation
     __syncthreads();
   
     // Accumulate the results from shared_histogram into the global histogram
     for (int bucket_index = thread_index_in_block; bucket_index < num_buckets; bucket_index += blockDim.x)
     {
         // Atomically add the value from shared_histogram to the global histogram
         atomicAdd(&(device_histogram[bucket_index].d_cnt), shared_histogram[bucket_index]);
     }
 }

 /*
 set a checkpoint and show the (natural) running time in seconds for the CPU version
 */
 void report_running_time() {
     long sec_diff, usec_diff;
     gettimeofday(&endTime, &Idunno);
     sec_diff = endTime.tv_sec  - startTime.tv_sec;
     usec_diff= endTime.tv_usec - startTime.tv_usec;
     if(usec_diff < 0) {
         sec_diff--;
         usec_diff += 1000000;
     }
     long total_usec = sec_diff * 1000000 + usec_diff;
     double total_ms = total_usec / 1000.0; // Convert to milliseconds
   
     printf("Running time for CPU version: %0.5f ms\n", total_ms);
 }

 /*-------------------------------------------------------------
     Main function: sets up inputs, runs CPU baseline and GPU kernel,
     and compares histograms.
 -------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int i;
    if (argc != 4) {
        printf("Usage: %s <num_points> <bucket_width> <block_size>\n", argv[0]);
        exit(1);
    }
    // In-place check for each argument to ensure that arguments contain only digits.
    for (int arg = 1; arg < 4; arg++) {
        for (int j = 0; argv[arg][j] != '\0'; j++) {
            if (!isdigit(argv[arg][j])) {
                printf("Error: Argument %d ('%s') must be a positive integer with no signs or decimals.\n", arg, argv[arg]);
                exit(1);
            }
        }
    }

    PDH_acnt = atoi(argv[1]);
    PDH_res  = atof(argv[2]);
    int threadsPerBlock = atoi(argv[3]);
    if (threadsPerBlock <= 0) {
        printf("Error: <block_size> must be a positive integer.\n");
        exit(1);
    }

    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);
    // Initialize histogram counts to zero.
    for (i = 0; i < num_buckets; i++) {
        histogram[i].d_cnt = 0;
    }

    srand(1);
    for (i = 0; i < PDH_acnt; i++) {
        atom_list[i].x_pos = ((double)rand()/RAND_MAX) * BOX_SIZE;
        atom_list[i].y_pos = ((double)rand()/RAND_MAX) * BOX_SIZE;
        atom_list[i].z_pos = ((double)rand()/RAND_MAX) * BOX_SIZE;
    }
    
    // --------- CPU Computation ---------
    printf("\nCPU-based version\n");
    gettimeofday(&startTime, &Idunno);
    PDH_baseline();
    report_running_time();
    output_histogram();

    // --------- GPU Computation using Output Privatization (Algorithm 3) ---------
    atom *d_atom_list;
    bucket *d_histogram;
    bucket *histogram_from_gpu = (bucket *)malloc(sizeof(bucket) * num_buckets);

    cudaMalloc((void**)&d_atom_list, sizeof(atom) * PDH_acnt);
    cudaMalloc((void**)&d_histogram, sizeof(bucket) * num_buckets);
    cudaMemset(d_histogram, 0, sizeof(bucket) * num_buckets);
    cudaMemcpy(d_atom_list, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);

    // For the optimized kernel, grid dimension = number of tiles = ceil(PDH_acnt / block_size)
    int M = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;
    // Dynamic shared memory size = (num_buckets * sizeof(unsigned long long)) + (threadsPerBlock * sizeof(atom))
    size_t sharedMemSize = num_buckets * sizeof(unsigned long long) + threadsPerBlock * sizeof(atom);

    // ---------------- CUDA Event Timing ----------------
    // Create CUDA event objects "start" and "stop" to measure the kernel's execution time.
    // cudaEventCreate() allocates resources for the events.
    // cudaEventRecord() marks the start and stop times around the kernel launch.
    // cudaEventSynchronize() ensures that the stop event is completed before measuring elapsed time.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch the optimized kernel.
    PDH_kernel<<<M, threadsPerBlock, sharedMemSize>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);

    // Record the stop event and wait for the kernel to finish.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    // Calculate elapsed time between the two events in milliseconds.
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("********  Total  Running  Time  of  Kernel  =  %0.5f ms *******\n", elapsedTime);
    // Destroy the CUDA events to free resources.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // ----------------------------------------------------

    // Copy the GPU histogram back to host.
    cudaMemcpy(histogram_from_gpu, d_histogram, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);
    printf("GPU-based version \n");
    output_histogram_gpu(histogram_from_gpu);
    compare_histograms(histogram, histogram_from_gpu);
    
    // Cleanup
    free(atom_list);
    free(histogram);
    free(histogram_from_gpu);
    cudaFree(d_atom_list);
    cudaFree(d_histogram);

    return 0;
}
