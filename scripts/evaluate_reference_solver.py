import subprocess
import os


def main():
    dir_name = 'Poisson'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    platform = "linux"
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'
    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'
    path_to_executable = f"{base_path}/generated/{problem_name}"
    nruns = 20
    current_path = os.getcwd()
    os.chdir(base_path)
    _ = subprocess.run(['java', '-cp',
                        compiler_path, 'Main',
                        f'{base_path}/{settings_path}',
                        f'{base_path}/{knowledge_path}',
                        f'{base_path}/lib/{platform}.platform'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chdir(current_path)
    _ = subprocess.run(['make', '-j10', '-s', '-C', f'{path_to_executable}'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    total_execution_time = 0
    total_iterations = 0
    for _ in range(nruns):
        result = subprocess.run([f'{path_to_executable}/exastencils'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = result.stdout.decode('utf8')
        tokens = output.split("\n")
        # tokens_first_line = tokens[0].split()
        tmp = tokens[-4].split()

        try:
            iterations = float(tmp[2])
        except:
            iterations = 100
        total_iterations += iterations
        tokens_last_line = tokens[-2].split()
        execution_time = float(tokens_last_line[-1])
        total_execution_time += execution_time

    print(f"Average solving time: {total_execution_time / nruns}", flush=True)
    print(f"Average number of iterations: {total_iterations / nruns}", flush=True)

if __name__ == "__main__":
    main()
