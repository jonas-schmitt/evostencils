import sys
import os
import subprocess
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    n = int(sys.argv[1])
    absolute_compiler_path = sys.argv[2]
    base_path = sys.argv[3]
    base_path_prefix = sys.argv[4]
    problem_name = sys.argv[5]
    platform = sys.argv[6]
    os.chdir(base_path)
    for i in range(1, n+1):
        j = i - 1
        if j % nprocs == mpi_rank:
            subprocess.run(['java', '-cp', absolute_compiler_path, 'main',
                            f'{base_path}/{base_path_prefix}/{problem_name}_{i}.settings',
                            f'{base_path}/{base_path_prefix}/{problem_name}_{i}.knowledge',
                            f'{base_path}/lib/{platform}.platform'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # output = tmp.stdout.decode('utf8')
            # print(output)
            # output = tmp.stderr.decode('utf8')
            # print(output)

            tmp = subprocess.run(['make', '-j1', '-s', '-C', f'{base_path}/generated/{problem_name}_{i}'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # output = tmp.stderr.decode('utf8')
            # print(output)


if __name__ == "__main__":
    main()
