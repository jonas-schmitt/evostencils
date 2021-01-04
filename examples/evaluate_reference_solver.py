import subprocess
import shutil
import numpy as np
import os
def main():

    cwd = os.getcwd()
    platform = "linux"
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'
    settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    path_to_executable = f"{base_path}/generated/2D_FD_Helmholtz_fromL3"
    nruns = 10
    current_path = os.getcwd()
    os.chdir(base_path)
    subprocess.run(['java', '-cp',
                    compiler_path, 'Main',
                    f'{base_path}/{settings_path}',
                    f'{base_path}/{knowledge_path}',
                    f'{base_path}/lib/{platform}.platform'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chdir(current_path)
    subprocess.run(['make', '-j12', '-s', '-C', f'{path_to_executable}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_execution_time = 0
    total_iterations = 0
    for _ in range(nruns):
        result = subprocess.run([f'{path_to_executable}/exastencils'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = result.stdout.decode('utf8')
        tokens = output.split("\n")
        tokens_first_line = tokens[0].split()
        try:
            iterations = float(tokens_first_line[2])
        except:
            iterations = 10000
        total_iterations += iterations
        tokens_last_line = tokens[-2].split()
        execution_time = float(tokens_last_line[-1])
        total_execution_time += execution_time

    print(f"Average solving time: {total_execution_time / nruns}", flush=True)
    print(f"Average number of iterations: {total_iterations / nruns}", flush=True)

if __name__ == "__main__":
    main()
