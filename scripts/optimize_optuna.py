import optuna, subprocess,re, pickle, sys, argparse
exec_path = "/Users/dinesh/Documents/code/hypre/src/test/ij"
nsweeps_max = 6
def generate_cmdline_args(trial):
    # suggest int for kappa from 1 to max_level
    kappa = trial.suggest_int('kappa', 1, max_level)
    
    # suggest rlx_down and rlx_up between 0,13,14
    rlx_down = trial.suggest_categorical('rlx_down', [0,13,14])
    rlx_up = trial.suggest_categorical('rlx_up', [0,13,14])

    # suggest number of sweeps between 0 and NSWEEPS
    ns_down = trial.suggest_int('ns_down', 0, nsweeps_max)
    ns_up = trial.suggest_int('ns_up', 0, nsweeps_max)

    cmd_args = ["-rhszero", "-x0rand","-n",str(nx),str(ny),str(nz),"-c",str(cx),str(cy),str(cz),"-kappacycle", str(kappa), "-rlx_down", str(rlx_down), "-rlx_up", str(rlx_up),"-ns_down", str(ns_down), "-ns_up", str(ns_up)]

    return cmd_args

def single_objective(trial):
    cmd_line_args = generate_cmdline_args(trial)
    output = subprocess.run([exec_path] + cmd_line_args, capture_output=True, text=True)

    # parse the output to extract wall clock time
    output_lines = output.stdout.split('\n')
    solve_phase = False
    run_time = 1e100
    i = 0 
    for line in output_lines:
        if "Solve phase times" in line:
            solve_phase=True
        elif "wall clock time" in line and solve_phase:
            match = re.search(r'\d+\.\d+', line)
            if match:
                run_time=float(match.group())*1000 # convert to milliseconds
            solve_phase=False
    return run_time

def multi_objective(trial):
    cmd_line_args = generate_cmdline_args(trial)
    output = subprocess.run([exec_path] + cmd_line_args, capture_output=True, text=True)

    # parse the output to extract wall clock time
    output_lines = output.stdout.split('\n')
    solve_phase = False
    run_time = 1e100
    n_iterations = 1e100
    convergence_factor = 1e100
    for line in output_lines:
        if "Solve phase times" in line:
            solve_phase=True
        elif "Convergence Factor" in line:
            match = re.search(r'\d+\.\d+', line)
            if match:
                convergence_factor = float(match.group())
        elif "wall clock time" in line and solve_phase:
            match = re.search(r'\d+\.\d+', line)
            if match:
                run_time=float(match.group())*1000 # convert to milliseconds
            solve_phase=False
        elif "Iterations" in line:
            match = re.search(r'\d+', line)
            if match:
                n_iterations = int(match.group())
    return run_time/n_iterations, convergence_factor

def single_objective_study(n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(single_objective, n_trials=10)
    return study

def multi_objective_study(n_trials=10):
    study = optuna.create_study(directions=['minimize', 'minimize'])
    study.optimize(multi_objective, n_trials=10)
    return study


if __name__ == "__main__":
    # command line arguments with argparse : exec_path, max_level, nx, ny, nz, cx, cy, cz, nsweeps_max, n_trials, study_type, save_path
    parser = argparse.ArgumentParser(description='Optimize Hypre')
    parser.add_argument('--exec_path', type=str, default=exec_path, help='path to the executable')
    parser.add_argument('--max_level', type=int, default=10, help='maximum level of the multigrid hierarchy')
    parser.add_argument('--nx', type=int, default=100, help='number of grid points in x direction')
    parser.add_argument('--ny', type=int, default=100, help='number of grid points in y direction')
    parser.add_argument('--nz', type=int, default=100, help='number of grid points in z direction')
    parser.add_argument('--cx', type=float, default=0.001, help='coefficient in x direction')
    parser.add_argument('--cy', type=float, default=1, help='coefficient in y direction')
    parser.add_argument('--cz', type=float, default=1, help='coefficient in z direction')
    parser.add_argument('--nsweeps_max', type=int, default=6, help='maximum number of sweeps')
    parser.add_argument('--n_trials', type=int, default=10, help='number of trials')
    parser.add_argument('--study_type', type=str, default="single", help='single or multi objective study')
    parser.add_argument('--save_path', type=str, default="single_objective_study.pkl", help='path to save the study object')
    args = parser.parse_args()
    exec_path = args.exec_path
    max_level = args.max_level
    nx = args.nx
    ny = args.ny
    nz = args.nz
    cx = args.cx
    cy = args.cy
    cz = args.cz
    nsweeps_max = args.nsweeps_max
    n_trials = args.n_trials
    save_path = args.save_path
    study_type = args.study_type

    if study_type == "single":
        study = single_objective_study(n_trials)
    elif study_type == "multi":
        study = multi_objective_study(n_trials)
    print(study.best_params)
    pickle.dump(study, open(save_path, "wb"))



   
