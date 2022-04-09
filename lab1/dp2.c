#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
        R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3];
    return R;
}

void randvec(long N, float *pA, float *pB){
    time_t t;
    srand((unsigned) time(&t));
    
    for (long i = 0; i < N; i++){
        pA[i] = 1.0;
        pB[i] = 1.0;
    }
}

int main(int argc, char* argv[]){
    float *pA, *pB;
    
    long vecsize = atol(argv[1]);
    int measurements = atoi(argv[2]);
    printf("Vec size: %ld of %d measurements.\n", vecsize, measurements);

    pA = (float *) malloc(vecsize * sizeof(float));
    pB = (float *) malloc(vecsize * sizeof(float));
    struct timespec start, end;
    double mean = 0;
    float result;
    randvec(vecsize, pA, pB);
    for(int i = 0; i < measurements; i++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        result = dpunroll(vecsize, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_usec = (((double)end.tv_sec + (double)end.tv_nsec/1000000000) - 
            ((double)start.tv_sec + (double)start.tv_nsec/1000000000));
        if (i >= measurements/2)
            mean += time_usec;
        double bandwidth = ((double)vecsize * 2 * 4 / 1073741824) / time_usec;
        double flopsec = (double)vecsize * 2 / time_usec / 1073741824;
        printf("R: %.06lf <T>: %.06lf sec B: %.03lf GB/sec F: %.03lf GFLOP/sec\n", 
            result, time_usec, bandwidth, flopsec);
    }
    mean = mean/(measurements/2);
    double avgbandwidth = ((double)vecsize * 2 * 4 / 1073741824) / mean;
    double avgflopmsec = (double)vecsize * 2 / mean / 1073741824;
    printf("Mean for second half repetitions: N: %li <T>: %.06f sec, B: %.03lf GB/sec, F: %.03lf GFLOP/sec\n",
        vecsize, mean, avgbandwidth, avgflopmsec);
    return 0;
}