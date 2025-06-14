/*
 ===============================================================================
 Project: CUDA‑Accelerated Bloom Filter with SipHash
 Author : Akmal Kurbanov <akmal@usf.edu>
 Reference:
    SipHash implementation from https://github.com/veorq/SipHash/tree/master

 Description:
    This program implements a Bloom filter for string membership using the
    SipHash cryptographic hash function. It includes both:
      • A CPU baseline (single‑threaded SipHash + byte‑array updates)
      • A GPU‑accelerated version (one thread per string, __device__ SipHash,
        atomicOr bit‑sets for concurrency)

 Usage:
    /apps/GPU_course/runScript.sh <source.cu> <num_elements> <false_positive_rate> <threads_per_block>

 Arguments:
    <source.cu>              CUDA source filename containing the filter kernels
    <num_elements>           Number of random strings to insert and then check
    <false_positive_rate>    Desired Bloom filter false‑positive probability (0 < p < 1)
    <threads_per_block>      CUDA block size (threads per block), e.g. 128, 256, 512

 CPU vs GPU Approaches:
    • CPU: Processes each string sequentially, calls siphash_cpu() k times,
      sets/checks bytes in a host array. Work = O(n·k), fully serial.
    • GPU: Launches up to millions of threads, each thread:
        – Computes k SipHash values in registers via inline SIPROUND
        – Uses atomicOr() on 32‑bit filter words to set bits without locks
        – Reads bits non‑atomically for membership tests
      Achieves massive concurrency, hides DRAM and compute latency
      via high occupancy and coalesced memory access.

 Key Optimizations:
    • __device__ SipHash-2-4 runs entirely in registers, no spills
    • One thread per string for balanced load and minimal divergence
    • atomicOr on 32‑bit words for safe, fine‑grained updates
    • Coalesced global reads/writes of filter bit‑array & position table
    • Early exit in query kernel on first zero bit to reduce work
    • Tunable block size for occupancy vs. resource trade‑off
 ===============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>     // For CPU timing
#include <cuda_runtime.h> // For CUDA API
#include <assert.h>
#include <stddef.h>
#include <inttypes.h>
#include <string.h>

/* SipHash implementation starts here */
// Constants for SipHash rounds (default SipHash-2-4)
#ifndef cROUNDS
#define cROUNDS 2
#endif
#ifndef dROUNDS
#define dROUNDS 4
#endif

// Rotate Left operation (essential for SipHash)
// though SipHash GPU implementation will be separate.
#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

// Little-Endian conversions for 32-bit and 64-bit integers
#define U32TO8_LE(p, v)            \
    (p)[0] = (uint8_t)((v));       \
    (p)[1] = (uint8_t)((v) >> 8);  \
    (p)[2] = (uint8_t)((v) >> 16); \
    (p)[3] = (uint8_t)((v) >> 24);

#define U64TO8_LE(p, v)              \
    U32TO8_LE((p), (uint32_t)((v))); \
    U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U8TO64_LE(p)                                           \
    (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) |        \
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) | \
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) | \
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

// Core SipHash round operation
#define SIPROUND           \
    do                     \
    {                      \
        v0 += v1;          \
        v1 = ROTL(v1, 13); \
        v1 ^= v0;          \
        v0 = ROTL(v0, 32); \
        v2 += v3;          \
        v3 = ROTL(v3, 16); \
        v3 ^= v2;          \
        v0 += v3;          \
        v3 = ROTL(v3, 21); \
        v3 ^= v0;          \
        v2 += v1;          \
        v1 = ROTL(v1, 17); \
        v1 ^= v2;          \
        v2 = ROTL(v2, 32); \
    } while (0)
#define TRACE

/* Computes a SipHash value (CPU version)
    ======================================
    *in: pointer to input data (read-only)
    inlen: input data length in bytes (any size_t value)
    *k: pointer to the key data (read-only), must be 16 bytes
    *out: pointer to output data (write-only), outlen bytes must be allocated
    outlen: length of the output in bytes, must be 8 or 16
*/
int siphash_cpu(const void *in, const size_t inlen, const void *k, uint8_t *out,
                const size_t outlen)
{

    const unsigned char *ni = (const unsigned char *)in;
    const unsigned char *kk = (const unsigned char *)k;
    // Ensure output length is valid (8 or 16 bytes)
    assert((outlen == 8) || (outlen == 16));
    // Initialize SipHash state variables (magic constants)
    uint64_t v0 = UINT64_C(0x736f6d6570736575);
    uint64_t v1 = UINT64_C(0x646f72616e646f6d);
    uint64_t v2 = UINT64_C(0x6c7967656e657261);
    uint64_t v3 = UINT64_C(0x7465646279746573);
    // Load the 16-byte key into two 64-bit integers
    uint64_t k0 = U8TO64_LE(kk);
    uint64_t k1 = U8TO64_LE(kk + 8);
    uint64_t m;                                                         // Temporary variable for message blocks
    int i;                                                              // Loop counter
    const unsigned char *end = ni + inlen - (inlen % sizeof(uint64_t)); // Pointer to end of full 8-byte blocks
    const int left = inlen & 7;                                         // Number of remaining bytes (0-7)
    uint64_t b = ((uint64_t)inlen) << 56;                               // Message length block, shifted to the high byte

    // Initialize state by XORing with the key
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

    // Modify state slightly if 128-bit output is requested
    if (outlen == 16)
        v1 ^= 0xee;

    // Process message in 8-byte blocks
    for (; ni != end; ni += 8)
    {
        m = U8TO64_LE(ni); // Load 8 bytes into m
        v3 ^= m;           // XOR message block into state
        TRACE;
        // Apply cROUNDS SipHash rounds
        for (i = 0; i < cROUNDS; ++i)
            SIPROUND;
        v0 ^= m; // XOR message block into state
    }

    // Process remaining bytes (0-7)
    // Build the last block 'b' byte by byte
    switch (left)
    {
    case 7:
        b |= ((uint64_t)ni[6]) << 48; // Fallthrough intended
    case 6:
        b |= ((uint64_t)ni[5]) << 40;
    case 5:
        b |= ((uint64_t)ni[4]) << 32;
    case 4:
        b |= ((uint64_t)ni[3]) << 24;
    case 3:
        b |= ((uint64_t)ni[2]) << 16;
    case 2:
        b |= ((uint64_t)ni[1]) << 8;
    case 1:
        b |= ((uint64_t)ni[0]);
        break;
    case 0:
        break; // No remaining bytes
    }

    // Finalize: XOR the last block (including length) into state
    v3 ^= b;
    TRACE;
    // Apply cROUNDS SipHash rounds
    for (i = 0; i < cROUNDS; ++i)
        SIPROUND;
    v0 ^= b;

    // Finalize: Modify state based on output length
    if (outlen == 16)
        v2 ^= 0xee; // Different finalization for 128-bit output
    else
        v2 ^= 0xff; // Standard finalization for 64-bit output

    TRACE;
    // Apply dROUNDS SipHash rounds for final mixing
    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;

    // Compute the first 8 bytes of the output hash
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out, b);

    if (outlen == 8)
        return 0;

    // Additional finalization step for 128-bit output
    v1 ^= 0xdd;
    TRACE;
    // Apply dROUNDS SipHash rounds again
    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;
    // Compute the second 8 bytes of the output hash
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out + 8, b);

    return 0;
}

/* Computes a SipHash value (GPU device version)
    ===========================================
    Similar logic to the CPU version but callable from CUDA kernels.
    Uses __device__ qualifier.
*/
__device__ int siphash_gpu(const unsigned char *in, const size_t inlen, const uint8_t *k, uint8_t *out,
                           const size_t outlen)
{
    // Cast key to unsigned char pointer (already uint8_t, so this is fine)
    const unsigned char *kk = (const unsigned char *)k;
    // No assert on device, assume valid outlen (should be 8 in this project)
    // assert((outlen == 8) || (outlen == 16)); // Cannot use assert in device code easily

    // Initialize SipHash state variables
    uint64_t v0 = UINT64_C(0x736f6d6570736575);
    uint64_t v1 = UINT64_C(0x646f72616e646f6d);
    uint64_t v2 = UINT64_C(0x6c7967656e657261);
    uint64_t v3 = UINT64_C(0x7465646279746573);
    uint64_t k0 = U8TO64_LE(kk);
    uint64_t k1 = U8TO64_LE(kk + 8);
    uint64_t m;
    int i;
    const unsigned char *end = in + inlen - (inlen % sizeof(uint64_t));
    const int left = inlen & 7;
    uint64_t b = ((uint64_t)inlen) << 56;

    // Initialize state by XORing with the key
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

    if (outlen == 16)
        v1 ^= 0xee;

    // Process message in 8-byte blocks
    for (; in != end; in += 8)
    {
        m = U8TO64_LE(in);
        v3 ^= m;
        for (i = 0; i < cROUNDS; ++i)
            SIPROUND;
        v0 ^= m;
    }

    // Process remaining bytes (0-7)
    switch (left)
    {
    case 7:
        b |= ((uint64_t)in[6]) << 48;
    case 6:
        b |= ((uint64_t)in[5]) << 40;
    case 5:
        b |= ((uint64_t)in[4]) << 32;
    case 4:
        b |= ((uint64_t)in[3]) << 24;
    case 3:
        b |= ((uint64_t)in[2]) << 16;
    case 2:
        b |= ((uint64_t)in[1]) << 8;
    case 1:
        b |= ((uint64_t)in[0]);
        break;
    case 0:
        break;
    }

    // Finalize: XOR the last block (including length)
    v3 ^= b;
    for (i = 0; i < cROUNDS; ++i)
        SIPROUND;
    v0 ^= b;

    // Finalize: Modify state based on output length
    if (outlen == 16)
        v2 ^= 0xee;
    else
        v2 ^= 0xff; // Standard finalization for 64-bit output 

    // Apply dROUNDS SipHash rounds for final mixing
    for (i = 0; i < dROUNDS; ++i)
        SIPROUND;

    // Compute the 8 bytes of the output hash
    b = v0 ^ v1 ^ v2 ^ v3;
    U64TO8_LE(out, b); // Write the 8-byte hash output

    // The CPU implementation always requests an 8-byte hash, 
    // so only the 8-byte output path is needed on the GPU.

    return 0;
}
/* SipHash implementation ends here */

// CUDA Error Checking Utility
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file,
                line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

typedef unsigned long long int uint128_t; // used for readability equivalent to long long int

#define MAX_STRING_LENGTH 20 // max length of each string generated by generate_flattened_string

// Bloom Filter structure to hold parameters
struct bloom_filter
{
    uint8_t num_hashes;     // number of hash functions (k)
    double error;           // desired probability for false positives (p)
    uint128_t num_bits;     // number of bits in array (m)
    uint128_t num_elements; // expected number of elements to be inserted (n)
    int misses;             // counter for false negatives during checks
};

// Global variables for parameters
double ERROR_RATE;           // false positivity rate (from command line)
uint64_t NUMBER_OF_ELEMENTS; // total number of elements (from command line)
uint64_t STRINGS_ADDED;      // number of strings actually added (usually same as NUMBER_OF_ELEMENTS)
int BLOCK_SIZE;              // CUDA block size (from command line)

struct bloom_filter bf_h; // bloom filter parameters (used by both CPU and GPU logic)
char *strings_h;          // Host: 1D-char array holding all strings flattened
int *positions_h;         // Host: 1D-int array storing start index of each string in strings_h
uint8_t *byte_array_h;    // Host: Bloom filter bit/byte array

float cpu_elapsed_time;             // For storing CPU execution time
struct timeval start_cpu, stop_cpu; // For CPU timing

/* Starts the CPU timer */
void start_timer()
{
    gettimeofday(&start_cpu, NULL);
}

/* Stops the CPU timer and calculates elapsed time in ms */
void stop_timer()
{
    gettimeofday(&stop_cpu, NULL);
    cpu_elapsed_time = (stop_cpu.tv_sec - start_cpu.tv_sec) * 1000.0;
    cpu_elapsed_time += (stop_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;
}

/* Initalize the filter parameters (k and m)
    =======================================
    *bloom: pointer to bloom filter struct
    elements: expected number of elements (n)
    error: desired false positivity rate (p)

    Calculates optimal number of bits (m) and hash functions (k)
    Reference: https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions
*/
void init_filter(struct bloom_filter *bloom, uint64_t elements, double error)
{
    bloom->error = error;
    bloom->num_elements = elements;

    // Calculate m (number of bits)
    bloom->num_bits = ceil((elements * log(error)) / log(1.0 / pow(2.0, log(2.0))));
    // Calculate k (number of hash functions)
    bloom->num_hashes = round(((double)bloom->num_bits / elements) * log(2.0));
    // Ensure at least one hash function
    if (bloom->num_hashes == 0)
        bloom->num_hashes = 1;
    bloom->misses = 0; // Initialize misses
}

/* Adds a string to the bloom filter (CPU version)
    =======================================
    *bloom: pointer to bloom filter struct
    *byte_array: pointer to host byte array
    *str: char array (string to add)
*/
void add_to_filter_cpu(struct bloom_filter *bloom, uint8_t *byte_array, const char *str)
{

    uint64_t hash;   // To store the 64-bit hash result
    uint8_t out[8];  // Output buffer for siphash (only need 8 bytes)
    uint8_t key[16]; // 16-byte key for siphash, initialized and modified per hash

    // Initialize the key (can be anything, just needs to be consistent)
    memset(key, 0, 16); // Start with a zero key for simplicity
    key[0] = 1;         // Make it non-zero

    // Find the string length
    size_t len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
    if (len == 0)
        return; // Skip empty strings

    // Generate and add bloom->num_hashes hashes for the string
    for (uint8_t i = 0; i < bloom->num_hashes; i++)
    {
        // Compute the SipHash (using the current key)
        siphash_cpu(str, len, key, out, 8);
        // Copy the 8-byte hash result into the uint64_t variable
        memcpy(&hash, out, sizeof(uint64_t));
        // Calculate the index in the bloom filter array
        uint128_t index = hash % bloom->num_bits;
        // Set the byte at that index to 1
        byte_array[index] = 1;

        // IMPORTANT: Generate a new key based on the previous hash to get different hash functions
        // This is a simple way to derive multiple keys/hashes from a base key and the input.
        // XORing the previous hash back into the key works.
        for (size_t j = 0; j < 8; j++)
        {                     // Modify the first 8 bytes of the key
            key[j] ^= out[j]; // XOR with the hash output
        }
        key[8] ^= (uint8_t)(hash >> 56);
        key[9] ^= (uint8_t)(hash >> 48);
    }
}

/* Checks if a string potentially exists in the bloom filter (CPU version)
    =======================================
    *bloom: pointer to bloom filter struct
    *byte_array: pointer to host byte array
    *str: char array (string to check)
    Returns: 1 if the string *might* be in the filter, 0 if it's *definitely not*.
*/
int check_filter_cpu(struct bloom_filter *bloom, uint8_t *byte_array, const char *str)
{

    uint64_t hash;
    uint8_t out[8];
    uint8_t key[16];

    memset(key, 0, 16);
    key[0] = 1; // Use the same initial key as in add_to_filter

    size_t len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
    if (len == 0)
        return 0; // Empty strings are not added

    // Generate and check the required number of hashes
    for (uint8_t i = 0; i < bloom->num_hashes; i++)
    {
        siphash_cpu(str, len, key, out, 8);
        memcpy(&hash, out, sizeof(uint64_t));

        // Calculate the index
        uint128_t index = hash % bloom->num_bits;

        // If the byte at this index is 0, the string is definitely NOT in the filter
        if (byte_array[index] == 0)
        {
            return 0;
        }

        // Regenerate the key for the next hash function, exactly as in add_to_filter
        for (size_t j = 0; j < 8; j++)
        {
            key[j] ^= out[j];
        }
        key[8] ^= (uint8_t)(hash >> 56);
        key[9] ^= (uint8_t)(hash >> 48);
    }

    // If all checked bytes were 1, the string might be in the filter (or it's a false positive)
    return 1;
}

// Returns a random alphanumeric character
char get_random_character()
{
    // Reduced charset for slightly faster generation if needed, includes space.
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    // Make sure rand() is seeded before first call.
    return charset[rand() % (sizeof(charset) - 1)]; // -1 excludes the null terminator
}

/* Generate a flattened 1D char array containing multiple random strings,
   and an array storing the starting position of each string.
   =======================================
   num_strings: number of strings to generate
   min_length, max_length: range for random string length
   **out_buffer: output pointer to the char buffer (allocated within)
   **out_offsets: output pointer to the int offsets array (allocated within)
   Returns: The total size (bytes) of the buffer including null terminators,
            or 0 on failure.
*/
/* Generate a flattened 1D char array containing multiple random strings,
   and an array storing the starting position of each string.
   =======================================
   num_strings: number of strings to generate
   min_length, max_length: range for random string length
   **out_buffer: output pointer to the char buffer (allocated within)
   **out_offsets: output pointer to the int offsets array (allocated within)
   Returns: The total size (bytes) of the buffer including null terminators,
            or 0 on failure.
*/
size_t generate_flattened_strings(int num_strings,
                                  int min_length,
                                  int max_length,
                                  char **out_buffer,
                                  int **out_offsets)
{
    if (min_length <= 0)
        min_length = 1;
    if (max_length < min_length)
        max_length = min_length;
    int range_length = max_length - min_length + 1;

    *out_offsets = (int *)malloc(num_strings * sizeof(int));
    if (!*out_offsets)
    {
        printf("Error: offsets array allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Estimate memory needed for flattened strings (upper bound)
    size_t buffer_capacity = (size_t)num_strings * (max_length + 1); // +1 for null terminator
    *out_buffer = (char *)malloc(buffer_capacity * sizeof(char));
    if (!*out_buffer)
    {
        printf("Error: string buffer allocation failed\n");
        free(*out_offsets);
        exit(EXIT_FAILURE);
    }

    int write_index = 0;
    for (int i = 0; i < num_strings; i++)
    {
        // Generate a random length within the specified range
        int str_len = min_length + rand() % range_length;
        (*out_offsets)[i] = write_index; // Store starting position

        // Generate random characters for the string
        for (int j = 0; j < str_len; j++)
        {
            (*out_buffer)[write_index++] = get_random_character();
        }
        // Null-terminate the string
        (*out_buffer)[write_index++] = '\0';
    }

    // Reallocate flattened array to the exact size used
    char *shrunk = (char *)realloc(*out_buffer, write_index * sizeof(char));
    if (!shrunk && write_index > 0)
    { // Realloc failure handling
        printf("Error: buffer reallocation failed\n");
        // The previous, potentially larger buffer is preserved if realloc fails and valid data remains intact.
        // If write_index is 0, *out_buffer might be NULL or invalid from failed malloc
        if (*out_buffer)
            return buffer_capacity; // Return original estimate as best guess
        else
            return 0; // Indicate failure if original malloc also failed
    }
    if (shrunk)
        *out_buffer = shrunk; // Update pointer if realloc succeeded or returned same ptr

    printf("Generated %d strings, total size %d bytes.\n", num_strings, write_index);

    return write_index; // Return the actual size used
}

// =====================================================================
// CUDA KERNELS
// =====================================================================

/**
 * GPU Bloom‑filter insertion kernel (one thread per string)
 *
 * Each CUDA thread:
 *   - Computes its global thread ID:
 *         uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 *   - Exits immediately if tid ≥ num_elements_to_add (bounds check).
 *   - Loads its assigned string by indexing into the flattened char array with positions[tid].
 *   - Measures string length in a simple loop (avoids device‑unsupported strlen).
 *   - Invokes siphash_gpu() k times, each call running entirely in registers via inline SIPROUND:
 *       • No global memory touched during hash rounds.
 *   - Maps the 64‑bit hash to a bit index in [0, num_bits):
 *         word_idx = bit_index >> 5;  // divides by 32
 *         mask     = 1u << (bit_index & 31);
 *   - Sets the bit using atomicOr(&filter_array_uint[word_idx], mask):
 *       • Uses 32‑bit atomic operations to prevent data races.
 *       • Contention is localized to single words and spread across banks.
 *   - Derives the next hash’s key by XOR‑mixing the previous 8‑byte output:
 *       • No extra memory allocations; key lives in registers.
 *
 * Optimizations:
 *   • Thread‑to‑data mapping is one‑string‑per‑thread for minimal control overhead.
 *   • Branch divergence only occurs on the tid≥N guard and zero‑length checks.
 *   • All SipHash computations use registers only (no spills), yielding high compute throughput.
 *   • Atomic writes are coalesced per 32‑bit word, maximizing DRAM bank utilization.
 *   • No shared memory required, reducing per‑block resource use and boosting occupancy.
 *   • Approx. 44 registers/thread ensures thousands of active warps for latency hiding.
 */
__global__ void add_kernel(unsigned int *filter_array_uint, // Bloom filter array (as uint array)
                           const char *strings,             // Flattened string data
                           const int *positions,            // Start position of each string
                           uint8_t num_hashes,              // k
                           uint128_t num_bits,              // m (original bit size)
                           uint64_t num_elements_to_add)    // How many strings this grid processes
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure thread ID is within the number of strings to add
    if (tid >= num_elements_to_add)
    {
        return;
    }

    // Get pointer to the start of the string for this thread
    const char *str = strings + positions[tid];

    // Find string length (cannot use strlen in device code)
    size_t len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
    // Skip empty strings if any were generated (unlikely with generator logic)
    if (len == 0)
        return;

    // Cast string pointer to unsigned char for siphash_gpu
    const unsigned char *ustr = (const unsigned char *)str;

    uint64_t hash;   // To store the 64-bit hash result
    uint8_t out[8];  // Output buffer for siphash_gpu
    uint8_t key[16]; // 16-byte key for siphash_gpu

    // Initialize the key consistently across threads and hashes
    // Needs to match the logic in add_to_filter_cpu and check_filter_cpu/gpu
    for (int k = 0; k < 16; ++k)
        key[k] = 0; // Reset key for each string
    key[0] = 1;     // Initial key state

    // Generate num_hashes hashes for the string
    for (uint8_t i = 0; i < num_hashes; i++)
    {
        // Compute the SipHash
        siphash_gpu(ustr, len, key, out, 8);
        // Directly interpret the 8-byte output as a 64-bit hash value
        hash = U8TO64_LE(out);

        // Calculate the index in the original bit array
        uint128_t bit_index = hash % num_bits;

        // --- Atomic Operation on the uint array ---
        // Calculate the index in the uint array
        uint128_t uint_index = bit_index / 32; // Each uint holds 32 bits
        // Calculate the bit position within that uint (0-31)
        unsigned int bit_pos = bit_index % 32;
        // Create a mask with only that bit set
        unsigned int mask = (1 << bit_pos);

        // Atomically OR the mask into the uint array element
        // This ensures that the bit is set to 1, even if multiple threads target the same uint concurrently.
        atomicOr(&filter_array_uint[uint_index], mask);
        // --- End Atomic Operation ---

        // Regenerate the key based on the previous hash (must match CPU logic)
        for (size_t j = 0; j < 8; j++)
        {
            key[j] ^= out[j];
        }
        key[8] ^= (uint8_t)(hash >> 56);
        key[9] ^= (uint8_t)(hash >> 48);
    }
}

/**
 * GPU Bloom‑filter query kernel (one thread per string)
 *
 * Each CUDA thread:
 *   - Computes global thread ID and exits if tid ≥ num_elements_to_check.
 *   - Locates its string pointer via positions[tid] in the flattened array.
 *   - Measures string length manually in a loop.
 *   - Runs siphash_gpu() k times in registers (same SIPROUND logic as insertion).
 *   - For each hash:
 *       • Computes bit_index and extracts word_idx + mask.
 *       • Performs a 32‑bit global read: value = filter_array_uint[word_idx].
 *       • Tests (value & mask)==0; on the first zero, sets result to 0 and breaks early.
 *   - If all k bits were set, writes result=1, else result=0, into results[tid].
 *
 * Optimizations:
 *   • One‑thread‑per‑string mapping keeps control simple and balanced.
 *   • Early‑exit on bit‑test failure saves work for negative cases.
 *   • Hashing is register‑only, fully unrolled inline via SIPROUND.
 *   • 32‑bit reads across threads in a warp coalesce naturally, reducing DRAM transactions.
 *   • No atomic needed for reads, eliminating serialization in the query path.
 *   • Branch divergence limited to bounds and early‑exit paths.
 *   • Uses only ~44 registers/thread, maintaining high SM occupancy and hiding memory/hash latency.
 */
__global__ void check_kernel(const unsigned int *filter_array_uint, // Bloom filter array (as uint array)
                             const char *strings,                   // Flattened string data
                             const int *positions,                  // Start position of each string
                             int *results,                          // Output array for results (0 or 1)
                             uint8_t num_hashes,                    // k
                             uint128_t num_bits,                    // m (original bit size)
                             uint64_t num_elements_to_check)        // How many strings this grid processes
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (tid >= num_elements_to_check)
    {
        return;
    }

    // Get pointer to the string for this thread
    const char *str = strings + positions[tid];

    // Find string length
    size_t len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
    // Handle empty strings (should not be considered 'present')
    if (len == 0)
    {
        results[tid] = 0;
        return;
    }

    // Cast for siphash_gpu
    const unsigned char *ustr = (const unsigned char *)str;

    uint64_t hash;
    uint8_t out[8];
    uint8_t key[16];

    // Initialize key (must match add functions)
    for (int k = 0; k < 16; ++k)
        key[k] = 0;
    key[0] = 1;

    int potentially_present = 1; // Assume present initially

    // Check all hash locations for this string
    for (uint8_t i = 0; i < num_hashes; i++)
    {
        siphash_gpu(ustr, len, key, out, 8);
        hash = U8TO64_LE(out);

        // Calculate the original bit index
        uint128_t bit_index = hash % num_bits;

        // --- Check the bit in the uint array ---
        // Calculate the index in the uint array
        uint128_t uint_index = bit_index / 32;
        // Calculate the bit position within that uint
        unsigned int bit_pos = bit_index % 32;
        // Create a mask for that bit
        unsigned int mask = (1 << bit_pos);

        // Read the uint value (no atomic needed for read)
        // Check if the specific bit is set using bitwise AND
        if ((filter_array_uint[uint_index] & mask) == 0)
        {
            // If the bit is 0, the string is definitely NOT present
            potentially_present = 0;
            break; // No need to check further hashes
        }
        // --- End Bit Check ---

        // Regenerate key (must match add functions)
        for (size_t j = 0; j < 8; j++)
        {
            key[j] ^= out[j];
        }
        key[8] ^= (uint8_t)(hash >> 56);
        key[9] ^= (uint8_t)(hash >> 48);
        // ...
    }

    // Write the final result (1 if all bits were set, 0 otherwise)
    results[tid] = potentially_present;
}

// =====================================================================
// MAIN FUNCTION
// =====================================================================
int main(int argc, char **argv)
{

    /* _____ COMMAND LINE ARGUMENTS _________________________________________________________________________ */

    // Expecting 3 arguments: num_elements, error_rate, block_size
    if (argc != 4)
    {
        fprintf(stderr, "Invalid usage. Requires 3 arguments:\n");
        fprintf(stderr, "./proj3 {# of elements} {desired %% error} {block size}\n");
        fprintf(stderr, "Example: ./proj3 10000 0.1 512\n");
        return -1;
    }

    // Parse arguments
    NUMBER_OF_ELEMENTS = atoi(argv[1]);
    ERROR_RATE = atof(argv[2]);
    BLOCK_SIZE = atoi(argv[3]);

    // Validate arguments
    if (NUMBER_OF_ELEMENTS == 0)
    {
        fprintf(stderr, "Invalid number of elements: %s\n", argv[1]);
        return -1;
    }
    if ((ERROR_RATE <= 0.0) || (ERROR_RATE >= 1.0))
    {
        fprintf(stderr, "Invalid error rate: %s. Must be between 0 and 1 (exclusive).\n", argv[2]);
        return -1;
    }
    if (BLOCK_SIZE <= 0)
    { // block size must be positive non zero integer
        fprintf(stderr, "Invalid block size: %s. Must be positive.\n", argv[3]);
        return -1;
    }

    // For this project, added all generated elements and checked all generated elements
    STRINGS_ADDED = NUMBER_OF_ELEMENTS;

    /* _____ GENERATE STRINGS (CPU) __________________________________________________________________________ */
    srand(1); // set seed for randomly generated strings
    size_t flattened_size_bytes = generate_flattened_strings(NUMBER_OF_ELEMENTS, 5, MAX_STRING_LENGTH, &strings_h, &positions_h);
    if (flattened_size_bytes == 0 || !strings_h || !positions_h)
    {
        fprintf(stderr, "String generation failed.\n");
        return -1; // Exit if generation failed
    }

    /* _____ INITIALIZE FILTER PARAMETERS ____________________________________________________________________ */
    // Initialize the bloom filter parameters (calculates m and k)
    // bf_h struct holds these parameters, used by both CPU and GPU paths.
    init_filter(&bf_h, STRINGS_ADDED, ERROR_RATE);

    /* _____ CPU IMPLEMENTATION & TIMING ______________________________________________________________________ */
    printf("\n--- Running CPU Implementation ---");

    // Allocate CPU bloom filter array
    byte_array_h = (uint8_t *)calloc(bf_h.num_bits, sizeof(uint8_t));
    if (!byte_array_h)
    {
        perror("Failed to allocate CPU bloom filter array");
        free(strings_h);
        free(positions_h);
        return -1;
    }

    // --- Start CPU Timer ---
    start_timer();

    // Add strings to CPU filter
    for (uint64_t i = 0; i < STRINGS_ADDED; i++)
    {
        add_to_filter_cpu(&bf_h, byte_array_h, strings_h + positions_h[i]);
    }

    // Check strings in CPU filter & count false negatives
    bf_h.misses = 0; // Reset misses counter
    for (uint64_t i = 0; i < NUMBER_OF_ELEMENTS; i++)
    {
        // check all elements since adding STRINGS_ADDED elements,
        // any miss among the first STRINGS_ADDED checks is a false negative.
        if (i < STRINGS_ADDED)
        {
            if (check_filter_cpu(&bf_h, byte_array_h, strings_h + positions_h[i]) == 0)
            {
                bf_h.misses++;
            }
        }
    }

    // --- Stop CPU Timer ---
    stop_timer();

    // Print CPU results
    printf("\n");
    printf("[CPU] Time: %.3f ms | False Negatives: %d/%llu\n", cpu_elapsed_time, bf_h.misses, (unsigned long long)STRINGS_ADDED);

    // Free CPU bloom filter array (keep strings/positions for GPU)
    free(byte_array_h);
    byte_array_h = NULL; // Avoid dangling pointer

    /* _____ GPU IMPLEMENTATION & TIMING ______________________________________________________________________ */
    printf("\n--- Running GPU Implementation ---\n");

    // Device pointers
    unsigned int *d_filter_array_uint = NULL; // Bloom filter array on GPU (as uints)
    char *d_strings = NULL;                   // Flattened strings on GPU
    int *d_positions = NULL;                  // String positions on GPU
    int *d_results = NULL;                    // Results array on GPU (for checks)

    // Host results array
    int *h_results = NULL;

    // Calculate the size of the filter array in terms of unsigned ints, rounding up.
    uint128_t num_uints = (bf_h.num_bits + 31) / 32;
    size_t filter_array_size_bytes = num_uints * sizeof(unsigned int);

    // Allocate memory on GPU
    checkCudaErrors(cudaMalloc((void **)&d_filter_array_uint, filter_array_size_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_strings, flattened_size_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_positions, NUMBER_OF_ELEMENTS * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_results, NUMBER_OF_ELEMENTS * sizeof(int)));

    // Allocate host memory for results
    h_results = (int *)malloc(NUMBER_OF_ELEMENTS * sizeof(int));
    if (!h_results)
    {
        perror("Failed to allocate host results array");
        cudaFree(d_filter_array_uint);
        cudaFree(d_strings);
        cudaFree(d_positions);
        cudaFree(d_results);
        free(strings_h);
        free(positions_h);
        return -1;
    }

    // Initialize GPU filter array to zeros
    checkCudaErrors(cudaMemset(d_filter_array_uint, 0, filter_array_size_bytes));

    // Copy data from Host to Device (Strings and Positions)
    // printf("Copying data Host -> Device...\n");
    checkCudaErrors(cudaMemcpy(d_strings, strings_h, flattened_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_positions, positions_h, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threadsPerBlock = BLOCK_SIZE;
    // Calculate grid size needed to cover all elements
    int blocksPerGrid = (NUMBER_OF_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    // printf("GPU Kernel Launch Configuration:\n Grid Size: %d blocks\n Block Size: %d threads\n", blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing GPU operations
    cudaEvent_t start_gpu, stop_gpu;
    checkCudaErrors(cudaEventCreate(&start_gpu));
    checkCudaErrors(cudaEventCreate(&stop_gpu));

    // --- Start GPU Timer ---
    // Record start event in the default stream
    checkCudaErrors(cudaEventRecord(start_gpu, 0));

    // Launch Add Kernel
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_filter_array_uint,
        d_strings,
        d_positions,
        bf_h.num_hashes,
        bf_h.num_bits,
        STRINGS_ADDED // Add the first STRINGS_ADDED elements
    );
    // Check for kernel launch errors immediately after launch
    checkCudaErrors(cudaGetLastError());

    // Launch Check Kernel
    check_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_filter_array_uint,
        d_strings,
        d_positions,
        d_results,
        bf_h.num_hashes,
        bf_h.num_bits,
        NUMBER_OF_ELEMENTS // Check all elements
    );
    checkCudaErrors(cudaGetLastError());

    // --- Stop GPU Timer ---
    // Record stop event in the default stream
    checkCudaErrors(cudaEventRecord(stop_gpu, 0));

    // Synchronize the events to wait for completion and ensure timing is accurate
    checkCudaErrors(cudaEventSynchronize(stop_gpu));

    // Calculate elapsed time between events in milliseconds
    float gpu_elapsed_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_ms, start_gpu, stop_gpu));

    // Copy results back from Device to Host
    checkCudaErrors(cudaMemcpy(h_results, d_results, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate GPU False Negatives (check results for the first STRINGS_ADDED elements)
    int gpu_false_negatives = 0;
    for (uint64_t i = 0; i < STRINGS_ADDED; i++)
    {
        if (h_results[i] == 0)
        { // If result is 0, it means check_kernel returned false (miss)
            gpu_false_negatives++;
        }
    }

    // Calculate Speedup
    float speedup = (cpu_elapsed_time > 0 && gpu_elapsed_ms > 0) ? (cpu_elapsed_time / gpu_elapsed_ms) : 0.0f;

    // Print GPU results
    printf("[GPU] Time: %.3f ms (%.2fx speedup) | False Negatives: %d/%llu\n\n",
           gpu_elapsed_ms, speedup, gpu_false_negatives, (unsigned long long)STRINGS_ADDED);

    /* _____ CLEANUP _________________________________________________________________________________________ */

    // Destroy CUDA events
    checkCudaErrors(cudaEventDestroy(start_gpu));
    checkCudaErrors(cudaEventDestroy(stop_gpu));

    // Free GPU memory
    checkCudaErrors(cudaFree(d_filter_array_uint));
    checkCudaErrors(cudaFree(d_strings));
    checkCudaErrors(cudaFree(d_positions));
    checkCudaErrors(cudaFree(d_results));

    // Free Host memory
    free(strings_h);
    free(positions_h);
    free(h_results);

    return 0;
}