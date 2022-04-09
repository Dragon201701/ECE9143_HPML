import argparse
import time
import numpy as np

def dp(N,A,B):
    # R = 0.0
    return np.sum(np.dot(A, B))

def main():
    print('Hello, World!')
    parser = argparse.ArgumentParser(description='<N> <repetitions>.')
    parser.add_argument('vecsize', type=int, nargs=1)
    parser.add_argument('measurements', type=int, nargs=1)
    args = parser.parse_args()
    # print('Vector size: ', args.vecsize[0], ' measurements: ', args.measurements[0])
    N = args.vecsize[0]
    measurements = args.measurements[0]
    A = np.ones(N,dtype=np.float32)
    B = np.ones(N,dtype=np.float32)
    mean = 0
    for i in range(measurements):
        start = time.monotonic()
        R = dp(N, A, B)
        end = time.monotonic()
        time_usec = end - start # in seconds
        
        if i >= measurements/2:
            mean += time_usec
        bandwidth = (N * 2 * 4 / 1073741824) / time_usec
        flopsec = N * 2 / time_usec / 1073741824
        print('R: %.6f <T>: %.6f sec B: %.3f GB/sec F: %.3f GFLOP/sec' % (R, time_usec, bandwidth, flopsec))
    mean = mean/(measurements/2)
    avgbandwidth = (N * 2 * 4 / 1073741824) / mean
    avgflopsec = N * 2 / mean / 1073741824
    print("Mean for second half repetitions: N: %.6f <T>: %.6f sec B: %.3f GB/sec F: %.3f GFLOP/sec" % (N, mean, avgbandwidth, avgflopsec))

if __name__ == '__main__':
    main()