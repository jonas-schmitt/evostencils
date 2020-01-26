import subprocess
problem_name = '2D_FD_LinearElasticity_fromL2'
for i in range(0, 10):
    print(f"Starting with run {i}", flush=True)
    optimizer = 'NSGAII'
    if i >= 5:
        optimizer = 'RANDOM'
        print("Using Multi-Objective Random Search", flush=True)
    else:
        print("Using NSGA-II", flush=True)

    result = subprocess.run(['python', 'examples/optimize.py', optimizer],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode('utf8')
    with open(f'./{problem_name}_{optimizer}_{i}.out', 'w') as file:
        file.write(output)
    subprocess.run(['cp', '-r', problem_name, f'{problem_name}_{optimizer}_{i}'])
    subprocess.run(['rm', '-r', problem_name])
    subprocess.run(['cp', '-r', f'../exastencils/Examples/LinearElasticity/{problem_name}.exa3.generated',
                    f'./{problem_name}_{optimizer}_{i}.exa3'])

