import subprocess
problem_name = '2D_FD_LinearElasticity_fromL2'
j = 0
for i in range(0, 10):
    print(f"Starting with run {i}", flush=True)
    if i % 2 == 1:
        optimizer = 'RANDOM'
        print("Using RANDOM", flush=True)
        j += 1
    else:
        optimizer = 'NSGAII'
        print("Using NSGA-II", flush=True)

    result = subprocess.run(['python', 'examples/optimize.py', optimizer],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode('utf8')
    with open(f'./{problem_name}_{optimizer}_{j}.out', 'w') as file:
        file.write(output)
    subprocess.run(['cp', '-r', problem_name, f'{problem_name}_{optimizer}_{j}'])
    subprocess.run(['rm', '-r', problem_name])
    subprocess.run(['cp', '-r', f'../exastencils/Examples/LinearElasticity/{problem_name}_0.exa3',
                    f'./{problem_name}_{optimizer}_{j}.exa3'])

