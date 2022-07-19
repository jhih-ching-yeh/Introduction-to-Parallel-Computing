//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include <pthread.h>
#include <unistd.h>

#include "sha256.h"

#include <unistd.h>
////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(s, b; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// ------------------------------------------------------------------------
__constant__ WORD cuda_k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define _rotl(v, s) ((v)<<(s) | (v)>>(32-(s)))
#define _rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))

#define _swap(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

__device__ void cuda_sha256_transform(SHA256 *ctx, const BYTE *msg)
{
    WORD a, b, c, d, e, f, g, h;
    WORD i, j;

    // Create a 64-entry message schedule array w[0..63] of 32-bit words
    WORD w[64];


    // Copy chunk into first 16 words w[0..15] of the message schedule array
// 直接展開
#pragma unroll 16
    for(i=0, j=0;i<16;++i, j+=4)
    {
        w[i] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | (msg[j+3]);
    }

    // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
// 直接展開
#pragma unroll 64
    for(i=16;i<64;++i)
    {
        WORD s0 = (_rotr(w[i-15], 7)) ^ (_rotr(w[i-15], 18)) ^ (w[i-15]>>3);
        WORD s1 = (_rotr(w[i-2], 17)) ^ (_rotr(w[i-2], 19))  ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }


    // Initialize working variables to current hash value
    a = ctx->h[0];
    b = ctx->h[1];
    c = ctx->h[2];
    d = ctx->h[3];
    e = ctx->h[4];
    f = ctx->h[5];
    g = ctx->h[6];
    h = ctx->h[7];

    // Compress function main loop:
// 直接展開
#pragma unroll 64
    for(i=0;i<64;++i)
    {
        WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
        WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
        WORD ch = (e & f) ^ ((~e) & g);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp1 = h + S1 + ch + cuda_k[i] + w[i];
        WORD temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Add the compressed chunk to the current hash value
    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;


}


__device__ void cuda_sha256(SHA256 *ctx, const BYTE *msg, size_t len)
{
    // test
   // cuda_sha256_transform( NULL , NULL );

    // Initialize hash values:
    // (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;


    WORD i, j;
    size_t remain = len % 64;
    size_t total_len = len - remain;

    // Process the message in successive 512-bit chunks
    // For each chunk:
    for(i=0;i<total_len;i+=64)
    {
        cuda_sha256_transform(ctx, &msg[i]);
    }


    // Process remain data
    BYTE m[64] = {};
    for(i=total_len, j=0;i<len;++i, ++j)
    {
        m[j] = msg[i];
    }

    // Append a single '1' bit
    m[j++] = 0x80;  //1000 0000

    // Append K '0' bits, where k is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
    if(j > 56)
    {
        cuda_sha256_transform(ctx, m);
        memset(m, 0, sizeof(m));
    //    printf("true\n");
    }

    // Append L as a 64-bit bug-endian integer, making the total post-processed length a multiple of 512 bits
    unsigned long long L = len * 8;  //bits
    m[63] = L;
    m[62] = L >> 8;
    m[61] = L >> 16;
    m[60] = L >> 24;
    m[59] = L >> 32;
    m[58] = L >> 40;
    m[57] = L >> 48;
    m[56] = L >> 56;
    cuda_sha256_transform(ctx, m);

    // Produce the final hash value (little-endian to big-endian)
    // Swap 1st & 4th, 2nd & 3rd byte for each word
#pragma unroll 8
for(i=0;i<32;i+=4)
    {
        _swap(ctx->b[i], ctx->b[i+3]);
        _swap(ctx->b[i+1], ctx->b[i+2]);
    }
}

__device__ void cuda_double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;

    cuda_sha256(&tmp, (BYTE*)bytes, len);
    cuda_sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
/*
__device__ void cude_calc_merkle_root()
{
    cuda_double_sha256((SHA256*)list[j], list[i], 64);

    cuda_double_sha256
}*/

void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}


__device__ int cuda_little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    #pragma unroll 32
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}


__global__ void cudaSolve_test( unsigned int *endNonce )
{
    endNonce[0] = 1 ;
    endNonce[1] = 1 ;
    endNonce[2] = 1 ;
}

__global__ void cudaSolve(
        void * gpu_sha256_ctx ,
        void * blockBuf , unsigned int gridSize
        , const unsigned char * target_hex
        , unsigned int *endNonce
        , unsigned int hashBlockSize
        , size_t startTask
        , size_t endTask
 )
{/*
    endNonce[0] = 1 ;
    endNonce[1] = 1 ;
    endNonce[2] = 1 ;
    return ;*/

    if( endNonce[1] != 0 )
        return ;
    SHA256 sha256_ctx;
    unsigned int x =blockIdx.x * blockDim.x + threadIdx.x + startTask ;

    if(( x >= startTask )&&( x <= endTask ))
    {

      //  unsigned int y =blockIdx.x*blockDim.x + threadIdx.y;
        unsigned int runIndex = x  ;//x * gridSize + y  ;// blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int blockIndex = runIndex % gridSize ;

        HashBlock block ;
    //    HashBlock * block = ((HashBlock*)blockBuf )+ blockIndex ;
      //  block->nonce = runIndex ;
      memcpy( &block , blockBuf , sizeof( block ));
        block.nonce = runIndex ;
        cuda_double_sha256(&sha256_ctx, (unsigned char*)&block, hashBlockSize);


        if(cuda_little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {/*
            printf("Found Solution!!\n");
            printf("hash #%10u (big): ", block.nonce);
            print_hex_inverse(sha256_ctx.b, 32);
            printf("\n\n");
            */
            // 結束
            endNonce[0] = blockIndex ;
            endNonce[1] = 1 ;
            endNonce[2] = runIndex ;
            memcpy( gpu_sha256_ctx , &sha256_ctx , sizeof( sha256_ctx ));

          //  __threadfence();         // ensure store issued before trap
         //   asm("trap;");            // kill kernel with error

        }

        endNonce[3] = endNonce[3] + 1 ;
        endNonce[4] = startTask ;
        endNonce[5] = endTask ;
    }
}



__global__ void cudaSolve_2D(
        void * gpu_sha256_ctx ,
        void * blockBuf , unsigned int gridSize
        , const unsigned char * target_hex
        , unsigned int *endNonce
        , unsigned int hashBlockSize
        , size_t startTask
        , size_t endTask
        , size_t width
        , size_t height
        , int runDeviceId
)
{

    if( endNonce[1] != 0 )
        return ;
    SHA256 sha256_ctx;
  //  unsigned int x =blockIdx.x * blockDim.x + threadIdx.x + startTask ;
    unsigned int x =blockIdx.y*blockDim.y + threadIdx.x;
    unsigned int y =blockIdx.x*blockDim.x + threadIdx.y;
/*
    if(( x >= width )||( y >= height ))
        return ;
*/
    x += y * width + startTask ;
    if(( x >= startTask )&&( x <= endTask ))
    {

        //  unsigned int y =blockIdx.x*blockDim.x + threadIdx.y;
        unsigned int runIndex = x  ;//x * gridSize + y  ;// blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int blockIndex = runIndex % gridSize ;

        HashBlock block ;
        //    HashBlock * block = ((HashBlock*)blockBuf )+ blockIndex ;
        //  block->nonce = runIndex ;
        memcpy( &block , blockBuf , sizeof( block ));
        endNonce[6] = block.nonce ;
        block.nonce = runIndex ;
        cuda_double_sha256(&sha256_ctx, (unsigned char*)&block, hashBlockSize);


        if(cuda_little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
        {
            // 結束
            endNonce[0] = blockIndex ;
            endNonce[1] = 1 ;
            endNonce[2] = runIndex ;
            memcpy( gpu_sha256_ctx , &sha256_ctx , sizeof( sha256_ctx ));


        }

    //    endNonce[1] = runDeviceId + 1 ;
        endNonce[3] = endNonce[3] + 1 ;
        endNonce[4] = startTask ;
        endNonce[5] = endTask ;
    }
}

#if(1)
//
typedef struct _THREAD_DATA
{
    int id ;
    int deviceCount ;
  //  HashBlock *block ;
    HashBlock block ;
    unsigned char target_hex[32] ;
    pthread_t t ;

}THREAD_DATA;
size_t __getBlock = 0 ;
int __gatOk = 0 ;
THREAD_DATA __deviceData[8]= {} ;

// lock
pthread_mutex_t __mutex ;

// int pthread_mutex_trylock( &__mutex );
// int pthread_mutex_unlock( &__mutex );


void* threadRun(void* data)
{
    THREAD_DATA * threadData = (THREAD_DATA*)data ;
    int i ;
    size_t task ;
    cudaStream_t s0;


    pthread_mutex_lock( &__mutex );
        printf( "hello pThread: %d\n" , threadData->id );


        printf( " -------------lock: %d ---------------\n" , threadData->id );
        cudaSetDevice( threadData->id );
        cudaStreamCreate(&s0);

        SHA256 *gpu_sha256_ctx ;
        cudaMalloc(&gpu_sha256_ctx, sizeof( SHA256 ));
        // 結束的index
        unsigned int *gpuEndNonce ;
        unsigned int endNonceData[8] = {} ;
     /*
        unsigned int *endNonceData ;
        cudaMallocHost( &endNonceData , sizeof( unsigned int ) * 8 );*/
        cudaMalloc(&gpuEndNonce, sizeof( unsigned int ) * 8 );
        cudaMemcpy(gpuEndNonce, endNonceData, sizeof( unsigned int ) * 8 , cudaMemcpyHostToDevice);

     //   unsigned char target_hex[32] = {0};
        // gpu hex
        unsigned char* gpu_target_hex ;
        cudaMalloc(&gpu_target_hex, 32 );
        cudaMemcpy(gpu_target_hex, threadData->target_hex, 32 , cudaMemcpyHostToDevice);

        // 改以 1d 方式處理
        unsigned int gridSize = 8 ;
        // 記憶體
        int memorySize = sizeof( HashBlock ) *gridSize;
        HashBlock *gpu_block ;
        HashBlock *cpu_block = (HashBlock *)malloc( memorySize );
        cudaMalloc(&gpu_block, memorySize);

        // copy
        for( i = 0 ; i < gridSize ; ++i )
        {
            memcpy( cpu_block + i , &threadData->block , sizeof( HashBlock ));
            (cpu_block + i)->nonce = threadData->id + 1 ;
        }
        // copy 到 gpu
        cudaMemcpy(gpu_block, cpu_block, memorySize, cudaMemcpyHostToDevice);

        // 把圖分成這些區塊
        ///      unsigned int nonce = 0xffffffff ;
        unsigned int nonce = 0x0fffffff ;
        unsigned int width  = 0x1000;
        unsigned int height = 0x1000;
        dim3 gpuGrid(
                height/gridSize
                ,width/gridSize,1);  //
        dim3 gpuBlock(gridSize,gridSize,1);

        printf( " -------------unlock: %d ---------------\n" , threadData->id );
    pthread_mutex_unlock( &__mutex );

    usleep( 10 );
/*
 *         dim3 gpuGrid(
                height/gridSize+1
                  ,width/gridSize+1,1);  //
        dim3 gpuBlock(gridSize,gridSize,1);
*/
    task = threadData->id * 0x100 / 2 ;
    int taskEnd = task + 0x50 - 1 ;
    for( ; task <= taskEnd ; task ++  )
    {
        size_t startTask = task << 24 ;
        size_t endTask = startTask | 0x00FFFFFF ;
/*
        pthread_mutex_lock( &__mutex );
            printf( " lock " , threadData->id );
            cudaSetDevice( threadData->id );
            printf( " .... done \n"  );
        pthread_mutex_unlock( &__mutex );
*/

        cudaSolve_2D<<<gpuGrid,gpuBlock,0, s0>>>
                //   cudaSolve<<<(0x01000000/gridSize)+1,gridSize>>> // 目前版
                ( gpu_sha256_ctx
                        , gpu_block , gridSize
                        , gpu_target_hex
                        , gpuEndNonce
                        , sizeof( HashBlock )
                        , startTask
                        , endTask
                        , width
                        , height
                        , threadData->id
                );

        //    cudaSolve_test<<<1,1>>>( gpuEndNonce );

        pthread_mutex_lock( &__mutex );
         //   printf( " run lock %d " , threadData->id );
            //
            cudaSetDevice( threadData->id );
            cudaMemcpy(endNonceData, gpuEndNonce ,  sizeof( unsigned int ) * 8 , cudaMemcpyDeviceToHost);


            printf( " >> %2d run: %2x %8x %8x " , threadData->id , task , startTask , endTask );
            //   printf("\n\n----- gpu ----------\n\n");
            printf( "       : %u %u %8x %8x %8x %8x %8d %8d\n"
                    , endNonceData[0]   , endNonceData[1] , endNonceData[2] , endNonceData[3]
                    , endNonceData[4]   , endNonceData[5] , endNonceData[6] , endNonceData[7]
            );
        pthread_mutex_unlock( &__mutex );

        usleep( 10 );


        if( endNonceData[1] != 0 )
            break ;
        if( __gatOk )
            break ;
    }
    /*
    SHA256 sha256_ctx;
    memset( &sha256_ctx , 0 , sizeof( SHA256 ));
    cudaMemcpy(&sha256_ctx, gpu_sha256_ctx, sizeof( sha256_ctx ), cudaMemcpyDeviceToHost);
*/
    // copy 到 gpu
    // cudaMemcpy(cpu_block, gpu_block, memorySize, cudaMemcpyDeviceToHost);

    // 最後
    // memcpy( &block , cpu_block + endNonceData[0] , sizeof( block ));


    if( endNonceData[1] != 0 )
    {
        __gatOk = 1 ;
        __getBlock = endNonceData[2] ;
    }

    printf( " >> thread id end \n" , threadData->id );

    return NULL ;
}
//
#endif

//
//
//
// --------------------------------------------------------------------------
void solve(FILE *fin, FILE *fout , int runMode , unsigned int runCount , int deviceCount )
{
  //  cudaSolve << <1, 1 >> > ();
  //  return ;
    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    int i ;
    size_t task ;

    pthread_mutex_init( &__mutex , NULL );

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    printf("start hashing");

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");


    // **** solve block ****
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    
    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");


    // ********** find nonce **************
    
    SHA256 sha256_ctx;
    memset( &sha256_ctx , 0 , sizeof( sha256_ctx ));
#if(1)
    if( deviceCount > 1 )
    {
        __getBlock = 0 ;
        __gatOk = 0 ;
        for( i = 0 ; i < deviceCount ; ++i )
        {
            THREAD_DATA *dev = __deviceData + i ;
            dev->id = i;
            dev->deviceCount ;
            memcpy( dev->target_hex , target_hex , sizeof( target_hex ));
            memcpy( &dev->block , &block , sizeof( block ));
            pthread_create(&dev->t, NULL, threadRun, dev );


        }

        for( i = 0 ; i < deviceCount ; ++i )
        {
            THREAD_DATA *dev = __deviceData + i ;
            pthread_join(dev->t, NULL);
        }

        block.nonce = __getBlock ;
    }else
#endif
        if( runMode != 0 )
    {

      //  cudaSetDevice( 1 );

        SHA256 *gpu_sha256_ctx ;
        cudaMalloc(&gpu_sha256_ctx, sizeof( SHA256 ));
        // 結束的index
        unsigned int *gpuEndNonce ;
        unsigned int endNonceData[8] = {} ;
        cudaMalloc(&gpuEndNonce, sizeof( unsigned int ) * 8 );
        cudaMemcpy(gpuEndNonce, endNonceData, sizeof( unsigned int ) * 8 , cudaMemcpyHostToDevice);

        // gpu hex
        unsigned char* gpu_target_hex ;
        cudaMalloc(&gpu_target_hex, sizeof( target_hex ));
        cudaMemcpy(gpu_target_hex, target_hex, sizeof( target_hex ), cudaMemcpyHostToDevice);

        // 改以 1d 方式處理
        unsigned int gridSize = 8 ;
        // 記憶體
        int memorySize = sizeof( HashBlock ) *gridSize;
        HashBlock *gpu_block ;
        HashBlock *cpu_block = (HashBlock *)malloc( memorySize );
        cudaMalloc(&gpu_block, memorySize);

        // copy
        for( i = 0 ; i < gridSize ; ++i )
            memcpy( cpu_block + i , &block , sizeof( block ));
        // copy 到 gpu
        cudaMemcpy(gpu_block, cpu_block, memorySize, cudaMemcpyHostToDevice);

        // 把圖分成這些區塊
   ///      unsigned int nonce = 0xffffffff ;
        unsigned int nonce = 0x0fffffff ;
        unsigned int width  = 0x1000;
        unsigned int height = 0x1000;
        dim3 gpuGrid(
                height/gridSize
                ,width/gridSize,1);  //
        dim3 gpuBlock(gridSize,gridSize,1);
/*
 *         dim3 gpuGrid(
                height/gridSize+1
                  ,width/gridSize+1,1);  //
        dim3 gpuBlock(gridSize,gridSize,1);
*/
        for( task = 0 ; task <= 0xFF ; task ++  )
        {
            size_t startTask = task << 24 ;
            size_t endTask = startTask | 0x00FFFFFF ;
            cudaSolve_2D<<<gpuGrid,gpuBlock>>>
         //   cudaSolve<<<(0x01000000/gridSize)+1,gridSize>>> // 目前版
                    ( gpu_sha256_ctx
                            , gpu_block , gridSize
                            , gpu_target_hex
                            , gpuEndNonce
                            , sizeof( HashBlock )
                            , startTask
                            , endTask
                            , width
                            , height
                            , 0
                    );

            //    cudaSolve_test<<<1,1>>>( gpuEndNonce );

            cudaMemcpy(endNonceData, gpuEndNonce ,  sizeof( unsigned int ) * 8 , cudaMemcpyDeviceToHost);

            printf( " >> run: %x %x %x\n" , task , startTask , endTask );
            //   printf("\n\n----- gpu ----------\n\n");
            printf( "       : %u %u %x %x %x %x\n"
                    , endNonceData[0]   , endNonceData[1] , endNonceData[2] , endNonceData[3]
                    , endNonceData[4]   , endNonceData[5]
                    );

            if( endNonceData[1] != 0 )
                break ;
        }
        cudaMemcpy(&sha256_ctx, gpu_sha256_ctx, sizeof( sha256_ctx ), cudaMemcpyDeviceToHost);

        // copy 到 gpu
        cudaMemcpy(cpu_block, gpu_block, memorySize, cudaMemcpyDeviceToHost);

        // 最後
        memcpy( &block , cpu_block + endNonceData[0] , sizeof( block ));


        block.nonce = endNonceData[2] ;




    }
    else
    {
        block.nonce = 0x2008e816 ;
        block.nonce = 0x16e80820 ;
        //  for(block.nonce=0x00000000; block.nonce <= runCount ;++block.nonce)
            for(block.nonce=0x00000000; block.nonce<=0xffffffff;++block.nonce)
            {
            //sha256d
            double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
            if(block.nonce % 1000000 == 0)
            {
                printf("hash #%10u (big): ", block.nonce);
                print_hex_inverse(sha256_ctx.b, 32);
                printf("\n");
            }

            if(little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0)  // sha256_ctx < target_hex
            {
                printf("Found Solution!!\n");
                printf("hash #%10u (big): ", block.nonce);
                print_hex_inverse(sha256_ctx.b, 32);
                printf("\n\n");

                break;
            }
            break ;

        }
    }

    // print result


    printf("\n\n----- end ----------\n\n");
    //little-endian
    printf("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    printf("\n");

    //big-endian
    printf("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    printf("\n\n");

    for(int i=0;i<4;++i)
    {
        fprintf( stdout , "%02x", ((unsigned char*)&block.nonce)[i]);
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    unsigned int runCount = 0 ;
    if( argc >= 4 )
        runCount = atoi( argv[3] ) ;
    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    if( runCount == 0 )
        runCount = 0xFFFFFFFF ;


    int deviceCount ;
    cudaGetDeviceCount( &deviceCount );
    printf( " DeviceCount :%d\n" , deviceCount );


    printf( "\n\n\n\n\n\n\n\n\n\n\n\n ======= gpu ========\n\n" );
 //   fclose( fin );
  //  fclose( fout );
  //  fin = fopen(argv[1], "r");
  //  fout = fopen(argv[2], "w");
  //  fscanf(fin, "%d\n", &totalblock);
  //  fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i)
    {
        solve(fin, fout , 1 , runCount , deviceCount );
    }
    /*
    // ---------------------------------
    if( runCount == 0xFFFFFFFF )
        return 0 ;
    printf( "\n\n ======= cpu ========\n\n" );
    fclose( fin );
    fclose( fout );
    fin = fopen(argv[1], "r");
    fout = fopen(argv[2], "w");
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);
    for(int i=0;i<totalblock;++i)
    {
        solve(fin, fout , 0 , runCount );
    }*/
    return 0;
}

