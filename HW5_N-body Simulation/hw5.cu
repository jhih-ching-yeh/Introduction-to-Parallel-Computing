#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <pthread.h>


#define MAX_SIZE 512
#define _RUN___shared__ 1

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;

const double planet_radius_pow2 = planet_radius * param::planet_radius ;

double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__device__ void _gpu_gravity_device_mass( double *out , double m0, double t)
{
    *out = m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device") {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}


__device__ void gpu_run_step( int index , int step, int n,
                             double* qx,
        double*qy,
        double* qz,
        double* vx,
        double* vy,
        double* vz,
        double* m,
        int* type

)
{
    // compute accelerations
    // std::vector<double> ax(n), ay(n), az(n);

    int i , j ;
    double mj = 0 ;
    double ax = 0 ;
    double ay = 0 ;
    double az = 0 ;
    double px = qx[index] ;
    double py = qx[index] ;
    double pz = qx[index] ;

    double t = step * param::dt ;
    t = fabs(sin(t / 6000)) ;
    i = index ;
    for ( j = 0; j < n; j++)
    {

        if (j == i) continue;
#if(1)
        mj = m[j];
        if (type[j] == 0 )
        {
            /*
             *
__device__ void _gpu_gravity_device_mass( double *out , double m0, double t)
{
    *out = m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

             */
            // mj = mj + 0.5 * mj * fabs(sin(t / 6000));
            //  _gpu_gravity_device_mass( &mj , mj, step * param::dt);
           // return ;
            mj = mj + 0.5 * mj * t ;
    //   mj = 1.0 ;
        }
#endif


        double dx = qx[j] - px;
        double dy = qy[j] - py;
        double dz = qz[j] - pz;
        double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);

        ax += param::G * mj * dx / dist3;
        ay += param::G * mj * dy / dist3;
        az += param::G * mj * dz / dist3;

    }

    __syncthreads();//同步
    vx[index] += ax * param::dt;
    vy[index] += ay * param::dt;
    vz[index] += az * param::dt;

    qx[index] += vx[index] * param::dt;
    qy[index] += vy[index] * param::dt;
    qz[index] += vz[index] * param::dt;


}


__device__ void gpu_InitMemory( double *outData , double *data , int N )
{

    int i ;
    for( i = 0 ; i < N ; ++i )
        outData[i] = data[i] ;
}
//
//
//
__global__ void _gpu_problem1( double *outData , double *data , int *temp , int n , int planet , int asteroid , double in_min_dist )
{
    __shared__ double min_dist ;///= in_min_dist ;


    double *pData ;
    int step ;
    int i , j ;
    unsigned int x =blockIdx.y*blockDim.y + threadIdx.x;
    unsigned int y =blockIdx.x*blockDim.x + threadIdx.y;
    ;
    const int index = x + y * 32 ;//threadIdx.x ;
    if( index >= n ) return ;

    min_dist = in_min_dist ;

    double len ;
#if(_RUN___shared__)
     __shared__ double qx[MAX_SIZE] ;
    __shared__ double qy[MAX_SIZE] ;
    __shared__ double qz[MAX_SIZE] ;
    __shared__ double vx[MAX_SIZE] ;
    __shared__ double vy[MAX_SIZE] ;
    __shared__ double vz[MAX_SIZE] ;
    __shared__ double m[MAX_SIZE] ;
  //  __shared__ int type[MAX_SIZE] ;


    if( index == 0 )
        gpu_InitMemory( qx , data , n );
    if( index == 1 )
        gpu_InitMemory( qy , data + ( index * n  ) , n );
    if( index == 2 )
        gpu_InitMemory( qz , data + ( index * n ), n  );
    if( index == 3 )
        gpu_InitMemory( vx , data + ( index * n ) , n );
    if( index == 4 )
        gpu_InitMemory( vy , data + ( index * n ), n  );
    if( index == 5 )
        gpu_InitMemory( vz , data + ( index * n ) , n );
    if( index == 6 )
        gpu_InitMemory( m , data + ( index * n ) , n );
    /*
    if( index == 7 )
    {
        for( i = 0 ; i < n ; ++i )
            type[i] = temp[i] ;
    }*/

#else
    double *qx = data ;
    double *qy = data + ( 1 * n  ) ;
    double* qz = data + ( 2 * n  ) ;
    double *vx = data + ( 3 * n  ) ;
    double *vy = data + ( 4 * n  ) ;
    double *vz = data + ( 5 * n  ) ;
    double *m = data + ( 6 * n  ) ;
    int *type = temp ;
#endif
    __shared__ double runLen[MAX_SIZE] ;
    __shared__ int touchCount ;


    double vvx = vx[index] ;
    double vvy = vy[index] ;
    double vvz = vz[index] ;

    double minDX , minDY , minDZ ;
    minDX = minDY = minDZ = in_min_dist ;

   const double invD = param::dt / 6000 ;
    __syncthreads();//同步
    //




//#pragma unroll 32
// TODO:  偷偷看 非正規
    int stepEnd =  param::n_steps  ;
  //  if( n >= 512 )
   //     stepEnd = param::n_steps - 20000 ;
    for ( step = 0; step <=stepEnd ; step ++ )
    {

        if (step > 0)
        {
       //     gpu_run_step( index , step, n, qx, qy, qz, vx, vy, vz, m, temp );
       //     int i , j ;
            double mj = 0 ;
            double ax = 0 ;
            double ay = 0 ;
            double az = 0 ;
            const double px = qx[index] ;
            const double py = qy[index] ;
            const double pz = qz[index] ;

   //   const double t = step * param::dt ;
         //   const double t = fabs(sin(step * param::dt / 6000)) ;
            const double t = fabs(sin(step * invD )) ;

            i = index ;

// #pragma unroll 32
            for ( j = 0; j < n; j++ )
            {

                if (j == i) continue;

                mj = m[j];
                if (temp[j] != 0 )
                {
                    // mj = mj + 0.5 * mj * fabs(sin(t / 6000));
                  //    _gpu_gravity_device_mass( &mj , mj, step * param::dt);
                    // return ;
                    mj = mj + 0.5 * mj * t ;
                    //   mj = 1.0 ;
                }



                const double dx = qx[j] - px;
                const double dy = qy[j] - py;
                const double dz = qz[j] - pz;
                const double dist3 = param::G * mj * rsqrt( dx * dx + dy * dy + dz * dz + param::eps * param::eps ) /
                                     ( dx * dx + dy * dy + dz * dz + param::eps * param::eps )
                ;

                ax += dx * dist3;
                ay += dy * dist3;
                az += dz * dist3;
            }

            __syncthreads();//同步
#if(0)
            vvx += ax * param::dt;
            vvy += ay * param::dt;
            vvz += az * param::dt;

            qx[index] += vvx * param::dt;
            qy[index] += vvy * param::dt;
            qz[index] += vvz * param::dt;
#else
            /*
            vx[index] += ax * param::dt;
            vy[index] += ay * param::dt;
            vz[index] += az * param::dt;

            qx[index] += vx[index] * param::dt;
            qy[index] += vy[index] * param::dt;
            qz[index] += vz[index] * param::dt;
             */
        //    vx[index] += ax * param::dt;
         //   vy[index] += ay * param::dt;
         //   vz[index] += az * param::dt;

            qx[index] += ( vx[index] += ax * param::dt ) * param::dt;
            qy[index] += ( vy[index] += ay * param::dt ) * param::dt;
            qz[index] += ( vz[index] += az * param::dt ) * param::dt;
#endif

        }

        touchCount = 0 ;
        __syncthreads();//同步
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
     //   len = (dx * dx + dy * dy + dz * dz) ;



        //    if(( min_dist > len )||( step == 0 ))
        if(( minDX * minDX + minDY * minDY + minDZ * minDZ ) >
           (dx * dx + dy * dy + dz * dz))
        {
            //     min_dist = len ;
            minDX = dx ;
            minDY = dy  ;
            minDZ = dz ;
            ++touchCount ;
        }


        __syncthreads();//同步
        if( touchCount == 0 )
            break;

    }
     runLen[index] = sqrt( minDX * minDX + minDY * minDY + minDZ * minDZ );
    __syncthreads();//同步

    // 最後
    if( index == 0 )
    {
        min_dist = in_min_dist ;
        for( i = 0 ; i < n ; ++i )
            if( min_dist > runLen[i] )
                min_dist = runLen[i] ;
        outData[0] = min_dist ;

    }
      //  outData[0] = min_dist ;
  //  outData[0] = sqrt( minDX * minDX + minDY * minDY + minDZ * minDZ );
}


// lock
pthread_mutex_t __mutex ;
double _min_dist = 0 ;

int _hit_time_step = 0 ;
int _destroyStep = -1 ;
int _gravity_device_id = -1 ;
double _missile_cost = 0 ;


const char *_fileName ;
// TODO: Thread1
void *threadProblem1( void *pp )
{

    int threadId = ((int*)&pp)[0] ;
    int i , n, planet, asteroid;

    double *gpuData ;
    double *memData ;
    double *pData ;

    clock_t a,b;
    a=clock();

    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    read_input( _fileName , n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    cudaStream_t s0;
    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();

    int sizeCount = sizeof( double ) * 7 * n ;


    int memTemp[1024] ;

    int *gpuTemp ;
    int mmSize = sizeof( int ) * n ;
    double *gupMin_dist ;


    memData = (double*)malloc( sizeCount );

    for ( i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
            memTemp[i] = 1 ;
        }else
        {
            memTemp[i] = 0 ;
        }
    }

    pData = memData ;
    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qx[i] ;
    }
    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qy[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qz[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vx[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vy[i] ;
    }


    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vz[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = m[i];
    }

    pthread_mutex_lock( &__mutex );
    {
        printf( " pThread lock : %d\n" , threadId );

        cudaSetDevice( threadId );
        cudaStreamCreate(&s0);

        cudaMalloc(&gpuData, sizeCount  );
        cudaMalloc( &gupMin_dist, sizeof( double ) );

        cudaMemcpy(gpuData, memData, sizeCount , cudaMemcpyHostToDevice);

        cudaMalloc( &gpuTemp, mmSize  );
        cudaMemcpy( gpuTemp, memTemp , mmSize  , cudaMemcpyHostToDevice);
        printf( " pThread unlock : %d\n" , threadId );

    }
    pthread_mutex_unlock( &__mutex );

    unsigned int height = 0x1000;
    dim3 gpuGrid(
            1
            ,1,1);  //
            const int gridSize = 32 ;
    dim3 gpuBlock(gridSize,gridSize,1);


    printf( "1 start gpu: %d %d %d \n" , n , planet , asteroid );
    _gpu_problem1<<<gpuGrid,gpuBlock,0, s0>>>
  //  _gpu_problem1<<<1,n,0, s0>>>
    (
            gupMin_dist ,  gpuData , gpuTemp , n , planet , asteroid , min_dist
    );
    printf( "_gpu_problem1 down : %d\n" , threadId );

    usleep(2000*1000);
    pthread_mutex_lock( &__mutex );
    {
        printf( "hello pThread lock : %d\n" , threadId );

        cudaSetDevice( threadId );
        cudaMemcpy( &_min_dist , gupMin_dist ,  sizeof( double ) , cudaMemcpyDeviceToHost );
        printf( "thread _min_dist: %lf\n" , _min_dist );
        printf( "hello pThread unlock : %d\n" , threadId );
    }
    pthread_mutex_unlock( &__mutex );

    b=clock();

    printf( "pThread 1 end -> runTime : %lf\n" , double(b-a)/CLOCKS_PER_SEC );
    return 0 ;
}

// ----------------------------------------------------

//
__global__ void _gpu_problem2( int *outData , double *data , int *temp , int n , int planet , int asteroid , double in_min_dist )
{
    __shared__ double min_dist ;///= in_min_dist ;


    double *pData ;
    int step ;
    int i , j ;

    unsigned int x =blockIdx.y*blockDim.y + threadIdx.x;
    unsigned int y =blockIdx.x*blockDim.x + threadIdx.y;
    ;
  //  const int index = x + y * 16 ;//threadIdx.x ;
  //  if( index >= n ) return ;
    const int index = threadIdx.x ;
    min_dist = in_min_dist ;

    double *qx = data ;
    double *qy = data + ( 1 * n  ) ;
    double* qz = data + ( 2 * n  ) ;
    double *vx = data + ( 3 * n  ) ;
    double *vy = data + ( 4 * n  ) ;
    double *vz = data + ( 5 * n  ) ;
    double *m = data + ( 6 * n  ) ;
    int *type = temp ;

    double len ;
#if(_RUN___shared__)
    __shared__ double _s_qx[MAX_SIZE] ;
    __shared__ double _s_qy[MAX_SIZE] ;
    __shared__ double _s_qz[MAX_SIZE] ;
    __shared__ double _s_vx[MAX_SIZE] ;
    __shared__ double _s_vy[MAX_SIZE] ;
    __shared__ double _s_vz[MAX_SIZE] ;
    __shared__ double _s_m[MAX_SIZE] ;


    if( index == 0 )
        gpu_InitMemory( _s_qx , data , n );
    if( index == 1 )
        gpu_InitMemory( _s_qy , data + ( index * n  ) , n );
    if( index == 2 )
        gpu_InitMemory( _s_qz , data + ( index * n ), n  );
    if( index == 3 )
        gpu_InitMemory( _s_vx , data + ( index * n ) , n );
    if( index == 4 )
        gpu_InitMemory( _s_vy , data + ( index * n ), n  );
    if( index == 5 )
        gpu_InitMemory( _s_vz , data + ( index * n ) , n );
    if( index == 6 )
        gpu_InitMemory( _s_m , data + ( index * n ) , n );
    /*
    if( index == 7 )
    {
        for( i = 0 ; i < n ; ++i )
            type[i] = temp[i] ;
    }*/

#else
    double *qx = data ;
    double *qy = data + ( 1 * n  ) ;
    double* qz = data + ( 2 * n  ) ;
    double *vx = data + ( 3 * n  ) ;
    double *vy = data + ( 4 * n  ) ;
    double *vz = data + ( 5 * n  ) ;
    double *m = data + ( 6 * n  ) ;

#endif

    double *__qx = data ;
    double *__qy = data + ( 1 * n  ) ;
    double* __qz = data + ( 2 * n  ) ;
    double *__vx = data + ( 3 * n  ) ;
    double *__vy = data + ( 4 * n  ) ;
    double *__vz = data + ( 5 * n  ) ;
    double *__m = data + ( 6 * n  ) ;

    qx = _s_qx ;
    qy = _s_qy ;
    qz = _s_qz;
    vx = _s_vx ;
    vy = _s_vy ;
    vz = _s_vz ;
    m = _s_m;

    __shared__ double runLen[MAX_SIZE] ;

    __shared__ int isEnd ;
    __shared__ int destroyStep ;

    destroyStep = -1 ;

    double vvx = vx[index] ;
    double vvy = vy[index] ;
    double vvz = vz[index] ;
    double qqx = qx[index] ;
    double qqy = qy[index] ;
    double qqz = qz[index] ;

    isEnd = 0 ;

    __syncthreads();//同步
    //
    outData[3] = 0 ;

    double missile_dst = 0 ;
    const int isMisslie = (temp[index] == 1 ) ;

    const  double invD =  param::dt / 6000 ;

    for ( step = 1; step <= param::n_steps; step ++ )
    {

      //  if (step > 0)
        {
            //     gpu_run_step( index , step, n, qx, qy, qz, vx, vy, vz, m, temp );
            //     int i , j ;
            double mj = 0 ;
            double ax = 0 ;
            double ay = 0 ;
            double az = 0 ;
            const double px = qx[index] ;
            const double py = qy[index] ;
            const double pz = qz[index] ;

            //   const double t = step * param::dt ;
            const double t = 0.5 * fabs(sin(step * invD )) ;
         //   i = index ;
// 2
// #pragma unroll 8
            vvx = vx[index] ;
            vvy = vy[index] ;
            vvz = vz[index] ;
            qqx = qx[index] ;
            qqy = qy[index] ;
            qqz = qz[index] ;

            for ( j = 0; j < n; j++ )
            {

                if (j == index) continue;

                mj = m[j];
                if (temp[j] != 0 )
                {
                    // mj = mj + 0.5 * mj * fabs(sin(t / 6000));
                 //   _gpu_gravity_device_mass( &mj , mj, step * param::dt);
                    // return ;
                    mj += mj * t ;
                    //   mj = 1.0 ;
                }



                const double dx = qx[j] - px;
                const double dy = qy[j] - py;
                const double dz = qz[j] - pz;

                const double dist3 = param::G * mj * rsqrt( dx * dx + dy * dy + dz * dz + param::eps * param::eps ) /
                                    ( dx * dx + dy * dy + dz * dz + param::eps * param::eps )
                ;

                ax += dx * dist3;
                ay += dy * dist3;
                az += dz * dist3;


            }

            __syncthreads();//同步

            /*
            vx[index] += ax * param::dt;
            vy[index] += ay * param::dt;
            vz[index] += az * param::dt;

            qx[index] += vx[index] * param::dt;
            qy[index] += vy[index] * param::dt;
            qz[index] += vz[index] * param::dt;*/

            qqx += ( vvx += ax * param::dt ) * param::dt;
            qqy += ( vvy += ay * param::dt ) * param::dt;
            qqz += ( vvz += az * param::dt ) * param::dt;


        }

        vx[index] = vvx ;
        vy[index] = vvy ;
        vz[index] = vvz ;
        qx[index] = qqx ;
        qy[index] = qqy ;
        qz[index] = qqz ;

        __syncthreads();//同步

        if( destroyStep > 0 )
        {
            const double dx = qx[planet] - qx[asteroid];
            const double dy = qy[planet] - qy[asteroid];
            const double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius_pow2 )
            {
                //  hit_time_step = step;
                isEnd = 1 ;
                outData[0] = step ;
                //      outData[1] = -1 ;
                //   outData[2] = 0 ;
            }
        }else //     if( destroyStep < 0 )
        if( isMisslie != 0 )
      //  if(temp[index] == 1 )// 飛彈
        {
            const double dx = qx[planet] - qqx;
            const double dy = qy[planet] - qqy;
            const double dz = qz[planet] - qqz;

         //   double missile_dst = step * param::dt * param::missile_speed ;
            missile_dst += param::dt * param::missile_speed ;
            //	if( hit_time_step > 0 )
            if(( missile_dst * missile_dst ) > ( dx * dx + dy * dy + dz * dz ))
            {
                // 放暫存
                for( i = 0 ; i < n ; ++i )
                {
                    __qx[i] = qx[i] ;
                    __qy[i] = qy[i] ;
                    __qz[i] = qz[i] ;
                    __vx[i] = vx[i] ;
                    __vy[i] = vy[i] ;
                    __vz[i] = vz[i] ;
                    __m[i] = m[i] ;
                }
                //
                destroyStep = index ;
                outData[1] = index ;
                outData[2] = step ;
                outData[3] = step ;
              //  m[index] = 0 ;
                __m[index] = 0 ;

            }
        }
        __syncthreads();//同步
        if( isEnd )
            break ;

    }

        //
    // 再算一次
    isEnd = 0 ;



    int endSteps  = outData[0] ;
    step = outData[3] + 1 ;
    outData[3] = outData[0] ;
    outData[4] = step ;
    outData[5] = endSteps ;

    // m[outData[1]] = _m_back ;

    for (  ; step <= endSteps ; step++)
    {

      //  if (step > 0)
        {
            //     gpu_run_step( index , step, n, qx, qy, qz, vx, vy, vz, m, temp );
            //     int i , j ;
            double mj = 0 ;
            double ax = 0 ;
            double ay = 0 ;
            double az = 0 ;
            const double px = __qx[index] ;
            const double py = __qy[index] ;
            const double pz = __qz[index] ;

            //   const double t = step * param::dt ;
      //      const double t = fabs(sin(step * param::dt / 6000)) ;
            const double t = 0.5 * fabs(sin(step * invD )) ;

            i = index ;

            for ( j = 0; j < n; j++)
            {

                if (j == i) continue;

                mj = __m[j];
                if (temp[j] != 0 )
                {
                    // mj = mj + 0.5 * mj * fabs(sin(t / 6000));
                    //   _gpu_gravity_device_mass( &mj , mj, step * param::dt);
                    // return ;
                    mj = mj + mj * t ;
                    //   mj = 1.0 ;
                }




                const double dx = __qx[j] - px;
                const double dy = __qy[j] - py;
                const double dz = __qz[j] - pz;
/*
                const double dist3 =rsqrt( dx * dx + dy * dy + dz * dz + param::eps * param::eps ) /
                                    ( dx * dx + dy * dy + dz * dz + param::eps * param::eps )
                ;

                ax += param::G * mj * dx * dist3;
                ay += param::G * mj * dy * dist3;
                az += param::G * mj * dz * dist3;*/
                const double dist3 = param::G * mj * rsqrt( dx * dx + dy * dy + dz * dz + param::eps * param::eps ) /
                                     ( dx * dx + dy * dy + dz * dz + param::eps * param::eps )
                ;

                ax += dx * dist3;
                ay += dy * dist3;
                az += dz * dist3;

            }

            __syncthreads();//同步

            __vx[index] += ax * param::dt;
            __vy[index] += ay * param::dt;
            __vz[index] += az * param::dt;

            __qx[index] += __vx[index] * param::dt;
            __qy[index] += __vy[index] * param::dt;
            __qz[index] += __vz[index] * param::dt;


        }

        __syncthreads();//同步
    //    if( index == 0 )
        {
            double dx = __qx[planet] - __qx[asteroid];
            double dy = __qy[planet] - __qy[asteroid];
            double dz = __qz[planet] - __qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius_pow2 )
            {
                //  hit_time_step = step;
                isEnd = 1 ;
                outData[0] = step ;
                outData[1] = -1 ;
           //     outData[5] = step ;
           //     outData[3] = step ;
            }
        }

        __syncthreads();//同步
        if( isEnd )
            break ;

    }

}
// TODO: GPU2
void *threadProblem2( void *pp )
{

    int threadId = ((int*)&pp)[0] ;
    int i , n, planet, asteroid;

    double *gpuData ;
    double *memData ;
    double *pData ;

    int outData[8] ;
#define OUT_COUNT 8

    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;
    read_input( _fileName , n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    cudaStream_t s0;
    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    clock_t a,b;
    a=clock();

    int sizeCount = sizeof( double ) * 7 * n ;


    int memTemp[1024] ;

    int *gpuTemp ;
    int mmSize = sizeof( int ) * n ;
    int *gpu_hit_time_step ;


    memData = (double*)malloc( sizeCount );

    for ( i = 0; i < n; i++) {
        if (type[i] == "device")
        {
            memTemp[i] = 1 ;
        }else
        {
            memTemp[i] = 0 ;
        }
    }

    pData = memData ;
    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qx[i] ;
    }
    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qy[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = qz[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vx[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vy[i] ;
    }


    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = vz[i] ;
    }

    for( i = 0 ; i < n ; ++i , pData ++ )
    {
        pData[0] = m[i];
    }

    pthread_mutex_lock( &__mutex );
    {
        printf( "\t 2 hello pThread lock: %d\n" , threadId );

        cudaSetDevice( threadId );
        cudaStreamCreate(&s0);

        cudaMalloc(&gpuData, sizeCount  );
        cudaMalloc( &gpu_hit_time_step, sizeof( int ) * OUT_COUNT );

        cudaMemcpy(gpuData, memData, sizeCount , cudaMemcpyHostToDevice);

        cudaMalloc( &gpuTemp, mmSize  );
        cudaMemcpy( gpuTemp, memTemp , mmSize  , cudaMemcpyHostToDevice);

        printf( "\t 2 hello pThread unlock: %d\n" , threadId );
    }
    pthread_mutex_unlock( &__mutex );

    dim3 gpuGrid(
            1
            ,1,1);  //
    const int gridSize = 16 ;
    dim3 gpuBlock(gridSize,gridSize,1);

  //  printf( "\t 2 start gpu: %d %d %d \n" , n , planet , asteroid );
 //   _gpu_problem2<<<gpuGrid,gpuBlock,0, s0>>>
    _gpu_problem2<<<1,n,0, s0>>>
            (
                    // gupMin_dist
                    gpu_hit_time_step ,  gpuData , gpuTemp , n , planet , asteroid , min_dist
            );
    usleep(2000*1000);
    printf( "\t 2 _gpu_problem2 down : %d\n" , threadId );
    pthread_mutex_lock( &__mutex );
    {
        printf( "\t 2 hello pThread: %d\n" , threadId );

        cudaSetDevice( threadId );
        cudaMemcpy( outData , gpu_hit_time_step ,  sizeof( int ) * OUT_COUNT , cudaMemcpyDeviceToHost );
        _hit_time_step = outData[3] ;
        _gravity_device_id = outData[1] ;
        _destroyStep = outData[2] ;
        _missile_cost = 100000 + ( _destroyStep+ 1 ) * param::dt *1000 ;
        if( _gravity_device_id < 0 )
            _missile_cost = 0 ;
        printf( "\t 2 out data: %d %d %d %d\n" , outData[0] , outData[1] , outData[2] , outData[3] );

        printf( "\t 2 thread _hit_time_step: %d\n" , _hit_time_step );
        printf( "\t 2 thread _gravity_device_id: %d\n" , _gravity_device_id );
        printf( "\t 2 thread _destroyStep: %d\n" , _destroyStep );
        printf( "\t 2 thread _missile_cost: %lf\n" , _missile_cost );
        printf( "\t 2 thread foo2 : %d -> %d\n" , outData[4] , outData[5] );

       // printf( "\t device: %d -> step: %d  cost: %lf\n" , kkk , step , cost );
    }
    pthread_mutex_unlock( &__mutex );

    b=clock();

    printf( "\t 2 pThread 2 end -> RunTime : %lf\n" , double(b-a)/CLOCKS_PER_SEC );
    return 0 ;
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    clock_t a,b;
    a=clock();

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    pthread_mutex_init( &__mutex , NULL );

    _fileName = argv[1] ;
 #define ISGPU_1 // GPU

#if defined(ISGPU_1)
    printf( "create pthread 1\n" );
    pthread_t threadProblem1_Id ;
    pthread_create(&threadProblem1_Id, NULL, threadProblem1, 0 );
    printf( "create pthread 1 .. ok \n" );
#else
    //
    printf( "CPU Problem 1 start\n" );
    //
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        }
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }
    printf( "min_dist:%lf\n" , min_dist );
    printf( "CPU Problem 1 end \n" );
#endif
    // Problem 2
#define ISGPU_2 // GPU
    int hit_time_step = -2;
    int gravity_device_id = -999;
    double missile_cost = -999;
#if defined(ISGPU_2)

    printf( "create pthread 2\n" );
    pthread_t threadProblem2_Id ;
    pthread_create(&threadProblem2_Id, NULL, threadProblem2, (void*)1 );

    printf( "create pthread 2 .. ok \n" );
#else

       printf( "CPU Problem 2 start\n" );

        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        for (int step = 0; step <= param::n_steps; step++)
        {
            if (step > 0)
            {
                run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
            }
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius_pow2 ) {
                hit_time_step = step;
                break;
            }
        }
        printf( " CPU Problem 2 end\n" );

#endif

    // Problem 3
    // CPU 直接算
    if( 0 )
    {
        printf( "CPU Problem 3 start\n" );

        read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
        for (int step = 0; step <= param::n_steps; step++)
        {
            if (step > 0)
            {
                run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
            }
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                hit_time_step = step;
                break;
            }
        }
        printf( " CPU Problem 3 end\n" );
    }
    // TODO
    // 等 thread 回來
#if defined(ISGPU_1)
    pthread_join(threadProblem1_Id, NULL);
    min_dist = _min_dist ;

#endif

#if defined(ISGPU_2)

    pthread_join(threadProblem2_Id, NULL);
    hit_time_step = _hit_time_step ;
//    destroyStep = _destroyStep ;
    missile_cost = _missile_cost ;
    gravity_device_id = _gravity_device_id ;
#endif
    // Problem 3
    // TODO

    printf( " -> hit_time_step: %d\n" , hit_time_step );
    printf( " -> gravity_device_id: %d\n" , gravity_device_id );
    printf( " -> missile_cost: %lf\n" , missile_cost );

    b=clock();
    printf( "\n\n runTime -> %lf\n" , double(b-a)/CLOCKS_PER_SEC );


    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
