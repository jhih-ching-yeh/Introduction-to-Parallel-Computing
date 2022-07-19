
// 這裡可以設定是要用 OPEN_MPI (oj) 還是 openMP
#define _OPEN_MPI 1
#if (_OPEN_MPI)
#include <mpi.h>
#endif

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <thread>
#include <cstdio>

#include <omp.h>

// timeGetTime
#include <sys/time.h>
// for linux 
unsigned int timeGetTime()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * 1000 + now.tv_usec/1000;
}


unsigned long long run2( unsigned long long r , unsigned long long k , unsigned long long ii , unsigned long long d )
{
    unsigned long long r2 = r * r ; //^2
    unsigned long long x = ii ; // start
    unsigned long long iin = x + d ; // end
    unsigned long long sq = 0 ; // sq
    unsigned long long ss = 0 ; // sq^2
    unsigned long long num = r2 - x * x ;
    // more the Max
    if( iin > r )
        iin = r ;
    unsigned long long y = 0 ;
    for ( ; x < iin ; ++x )
    {
        //cal diff
    //    num =  r2 - x*x ;
        // 沒有算過，就算第一份
        if( ss == 0 )
        {
            // cal sq
            sq = ceil(sqrtl( num )) ;
            ss = sq - 1  ; // the floor
            ss *= ss ; // ^2
            y += sq; // ++
        }else if( num > ss ) // not yet the minimum, so don't count
            y += sq ;
        else // renew
        {
#if(1)
            while( ss >= num )
            {
                sq -- ;
                ss -= ( sq * 2 - 1 );
            }
#else
#if(1)
            while(( sq * sq ) >= num )
            {
                sq -- ;
            }
            ss = sq * sq ;
            sq ++ ;

#else
            sq = ceil(sqrtl( num )) ;
            ss = sq - 1  ;
            ss *= ss ;
#endif
#endif
            y += sq;
        }
        num -= (( x << 1 ) + 1 );
    }

    return y ;
}
// In general, it can also be opened by itself
#if !(_OPEN_MPI)
unsigned long long run( unsigned long long r , unsigned long long k )
{


    omp_lock_t my_lock;
    omp_init_lock(&my_lock);
//                         4294967295
    unsigned long long d = r / 12 ;
    unsigned long long pixels = 0;
    unsigned long long ii ;
    const unsigned long long r2 = r * r ;

   // cpu_set_t cpu_set;
    //sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //int cpucnt = CPU_COUNT(&cpu_set);
//    printf( "cpucnt: %d \n " , cpucnt );
//return 0;
   // omp_set_num_threads( 1024 );
    #pragma omp parallel for //schedule(static)
    for( ii = 0 ; ii < r ; ii += d )
    {
        unsigned long long x = ii ;
        unsigned long long iin = x + d ;
        unsigned long long sq = 0 ;
        unsigned long long ss = 0 ;
        unsigned long long num = r2 - ii * ii ;
        if( iin > r )
            iin = r ;
        unsigned long long y = 0 ;
        for ( ; x < iin ; ++x )
        {
       //     num =  r2 - x*x ;

            if( ss == 0 )
            {
                sq = ceil(sqrtl( num )) ;
                ss = sq - 1  ;
                ss *= ss ;
                y += sq;

            }else if( num > ss )
                y += sq ;
            else
            {
#if(1)
              //  unsigned long long  yyy = ceil(sqrtl( num )) ;
#if(1)
                while( ss >= num )
                {
                    sq -- ;
                    ss -= ( sq * 2 - 1 );
                }
#else
                while( (sq * sq ) >= num )
                {
                    sq -- ;
                }
                sq ++ ;
                ss = sq - 1  ;
                ss *= ss ;
#endif

#else
                sq = ceil(sqrtl( num )) ;
                ss = sq - 1  ;
                ss *= ss ;
#endif
                y += sq;
            }
            // num 處理
            num -= (( x * 2 )+ 1 );
       //     y %= k;
        }

        omp_set_lock(&my_lock);
        pixels += y;
        pixels %= k;
        omp_unset_lock(&my_lock);
    }


    omp_destroy_lock(&my_lock);

    return (4 * pixels) % k ;
  //  setbuf(stdout, NULL) ;
  //  char ch[1024] ;

 //   sprintf( ch , "%llu\n", (4 * pixels) % k);
  //  printf( ch );
  //  printf( "%llu", (4 * pixels) % k);

  //  omp_destroy_lock(&my_lock);
}
#endif

#define SEND_INT 99
#include <string.h>
// lab2-judge
int main(int argc, char** argv) {


    // test ;
    if(0)
    {
        unsigned long long count = 357913941 ;
        unsigned long long rr ;
        unsigned long long kk = 0  ;
        unsigned long long r2 ;

        unsigned int start = timeGetTime()  ;
        for( rr = 0 ; rr < count ; ++rr )
        {
            kk += rr * rr ;
        }
        printf( "%llu %u\n " , kk , timeGetTime() - start );

         start = timeGetTime()  ;
         kk = 0 ;
         r2 = 0 ;
        for( rr = 0 ; rr < count ; ++rr )
        {
            kk += r2 ;
            r2 += (( rr << 1 )+ 1  );
        }
        printf( "%llu %u\n " , kk , timeGetTime() - start );
   //     return 0 ;

    }


#if(_OPEN_MPI)

    if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

    MPI_Init(&argc, &argv);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
   // printf( "%lld - " , r );
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  //  printf( "%d %d \t" , rank , world_size );
  //  return 0 ;
    int i ;
    unsigned long long ret = 0 ;
    unsigned long long recvData = 0 ;
    unsigned long long sendData[2] ;
    //
    MPI_Status status;

    //
    // divide all r equally across all cores
    unsigned int d = world_size  ;  // how many
    unsigned long long dd = ( d + r )/ d ; // Each copy should be counted, and a few more to avoid errors
    unsigned long long start = rank * dd ; // start index
    unsigned long long end = start + dd ;  // end index -> cal num of start - end 

    // start cal
    recvData = run2( r , k , start , dd );
   // printf( "%d %llu %llu %llu  " , rank , start , dd , recvData );
    // 0
    //main
    if( rank == 0 )
    {
        ret = recvData ;
        for( i = 1 ; i < world_size ; ++i )
        {
            recvData = 0 ;
            // receive operation result
            // MPI_Recv(資料, buffer, 資料大小, 格式, 從那個核心丟來的, 格式, 預設, 目前狀況)
            MPI_Recv( &recvData , sizeof( recvData ) , MPI_CHAR , i,
                     SEND_INT, MPI_COMM_WORLD, &status );

            ret += recvData ;
            ret %= k ;
        }
        printf( "%llu", (4 * ret) % k);
    }else
    {

        // output the result of the operation
        // MPI_Send(資料, 資料的大小, 資料的格式(1byte), 丟到那一個核心, 訊息代碼, 預設值)
        MPI_Send( &recvData , sizeof( recvData ) , MPI_CHAR, 0,
                SEND_INT, MPI_COMM_WORLD);                
    }
    MPI_Finalize();

    /*
    //  return 0 ;
    unsigned long long ret = run( r , k );
   // fprintf( stdou
#pragma omp critical
    {

    fprintf( stdout , "%llu",  ret );
    fflush( stdout );
    }*/

#else
    const unsigned long long rBuf[] = { 2147   , 21474836 , 214748364 , 2147483647 , 1401149118  , 4294967295,  0 } ;
    const unsigned long long kBuf[] = { 2147   , 21474836 , 214748364 , 2147483647 , 14011491183 , 1099511627775 ,  0 } ;
    const unsigned long long assBuf[] = { 2048 ,   300000 , 153006692 , 256357661  , 12260168853 , 576603832986 ,  0 } ;

    int i ;
    for( i = 0 ; rBuf[i] != 0 ; ++i )
    {
        printf( "\n\nrun %d: %lld\n" , i , rBuf[i] );
        unsigned int start = timeGetTime() ;
        printf( "%llu \n" , run( rBuf[i] , kBuf[i] ));
        printf( "%llu runTime:%d \n" , assBuf[i] , timeGetTime() - start );

    }
#endif
    return 0 ;
}
