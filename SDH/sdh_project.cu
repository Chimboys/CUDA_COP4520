/* ==================================================================
	Programmer: Akmal Kurbanov (akmal@usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the GAIVI machines
	(never complied on my local machine)
	To run: /apps/GPU_course/runScript.sh proj1-akmal.cu <num_of_data_points> <width_of_each_bucket>
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
	// made the count unsigned long long for several reasons
	// count cannot be negative, so unsigned is more appropriate
	// to statisfy atomicAdd function signature 
	// Signature: "unsigned long long atomicAdd(unsigned long long *address, unsigned long long val)"
} bucket;

bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
//included device variables in the main when we allocate memory on the GPU

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

// No need to allocate memory on GPU for PDH_anct and PDH_res since they are shared by value
__global__ void PDH_kernel(atom *d_atom_list, bucket *d_histogram, long long PDH_acnt, double PDH_res)
{
	// Each thread computes the pairwise distances between a specific atom (i) and all subsequent atoms (j).
	// This parallelization allows multiple threads to process different atoms simultaneously, significantly
	// improving performance compared to a sequential approach. The computed distances are then used to update
	// the corresponding bin in the histogram safely using atomic operations.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < PDH_acnt){
		for(int j = i+1; j <PDH_acnt; j++){
			double dx = d_atom_list[i].x_pos - d_atom_list[j].x_pos;
			double dy = d_atom_list[i].y_pos - d_atom_list[j].y_pos;
			double dz = d_atom_list[i].z_pos - d_atom_list[j].z_pos;
			double dist = sqrt(dx*dx + dy*dy + dz*dz);
			int hist_pos = (int) (dist / PDH_res);
			//using anatomicAdd to update the histogram specific entry by 1 safely
			atomicAdd(&(d_histogram[hist_pos].d_cnt), (unsigned long long)1);

		}
	}


}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	//just added a new line to make the output more readable
	printf("\n");
}

// print the count in all buckets of the histogram for the GPU
void output_histogram_gpu(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	//just added a new line to make the output more readable
	printf("\n");
}

// print the difference between the CPU and GPU histograms
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



int main(int argc, char **argv)
{
	int i;
	if(argc != 3) {
		printf("need 2 arguments like: <num_points> <res>\n");
		exit(1);
	}
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);


	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	
	// ~~~~~ CPU calculations starts here ~~~~~~~~
    printf("\n");
	printf("CPU-based version\n");
	gettimeofday(&startTime, &Idunno);
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	/* check the total running time */ 
	report_running_time();
	/* print out the histogram */
	output_histogram();

	
	// ~~~~~ GPU calculations starts here ~~~~~~~~
	
	// declaring pointers for the atom list and the histogram on the GPU
	atom *d_atom_list;
	bucket *d_histogram;

	//allocating memory for the histogram on the CPU to store GPU results
	bucket *histogram_from_gpu_to_cpu = (bucket *)malloc(sizeof(bucket)*num_buckets);

	//declaring number of threads per block
	int threadsPerBlock = 256;
	//declaring number of blocks per grid make sure that the there is enough blocks to cover all the atoms
	int blocksPerGrid = (PDH_acnt + threadsPerBlock - 1) / threadsPerBlock;

	// allocating memory on the GPU for the atom list and the histogram
	cudaMalloc((void**)&d_histogram, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);

	// initialize the GPU histogram to zero
	cudaMemset(d_histogram, 0, sizeof(bucket) * num_buckets);

	// copying the atom list and the histogram to the GPU
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	

	// calling the kernel function
	PDH_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);

	//Wait for Kernel to finish
	cudaDeviceSynchronize();
	//copyting the histogram from the GPU to the CPU
	cudaMemcpy(histogram_from_gpu_to_cpu, d_histogram, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	//printng the GPU histogram
    printf("GPU-based version\n");
	output_histogram_gpu(histogram_from_gpu_to_cpu);
	//printing the difference between the CPU and GPU histograms
	compare_histograms(histogram, histogram_from_gpu_to_cpu);
	

	// free the memory on the GPU and CPU

	free(atom_list);
    free(histogram);
    free(histogram_from_gpu_to_cpu);
    cudaFree(d_atom_list);
    cudaFree(d_histogram);

	return 0;
}

