from matplotlib import patches
from evostencils.optimization.program import Optimizer
from evostencils.optimization.program import Optimizer
from evostencils.code_generation.hypre import ProgramGenerator
from evostencils.grammar import multigrid as initialization
from evostencils.ir import base, system
import os
import sympy
import subprocess,re
import matplotlib.pyplot as plt


def evalaute(nx, ny, nz, max_level, min_level, grammar_string):
    # TODO adapt to actual path to exastencils project
    dir_name = 'Poisson'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'

    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'

    cycle_name= "gen_mgCycle"

    # model_based_estimation = False
    program_generator = ProgramGenerator(min_level,max_level ,0)
    program_generator.nx = nx
    program_generator.ny = ny
    program_generator.nz = nz

   # Obtain extracted information from program generator
    dimension = 2#program_generator.dimension  # Dimensionality of the problem
    finest_grid = 'u'#program_generator.finest_grid  # Representation of the finest grid
    coarsening_factors = [2,2]#program_generator.coarsening_factor
    min_level = program_generator.min_level  # Minimum discretization level
    max_level = program_generator.max_level  # Maximum discretization level
    equations = [] #program_generator.equations  # System of PDEs in SymPy
    operators = [] #program_generator.operators  # Discretized differential operators
    fields = [sympy.Symbol('u')] #program_generator.fields  # Variables that occur within system of PDEs
    for i in range(min_level, max_level + 1):
        equations.append(initialization.EquationInfo('solEq', i, f"( Laplace@{i} * u@{i} ) == RHS_u@{i}"))
        operators.append(initialization.OperatorInfo('RestrictionNode', i, None, base.Restriction))
        operators.append(initialization.OperatorInfo('ProlongationNode', i, None, base.Prolongation))
        operators.append(initialization.OperatorInfo('Laplace', i, None, base.Operator))
    size = 2 ** max_level
    grid_size = tuple([size] * dimension)
    h = 1 / (2 ** max_level)
    step_size = tuple([h] * dimension)
    tmp = tuple([2] * dimension)
    coarsening_factors = [tmp for _ in range(len(fields))]
    finest_grid = [base.Grid(grid_size, step_size, max_level) for _ in range(len(fields))]
    problem_name = "test_anisotropic"#program_generator.problem_name

    if not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          program_generator=program_generator,
                          epsilon=1e-12, infinity=1e100)
    maximum_block_size = 8
    path_to_individual = "" # TODO insert path to individuals
    # with open(path_to_individual, 'r') as file:
    solving_time, convergence_factor, number_of_iterations = optimizer.generate_and_evaluate_program_from_grammar_representation(grammar_string, maximum_block_size)
    print(f'Solving Time: {solving_time}, '
        f'Convergence factor: {convergence_factor}, '
        f'Number of Iterations: {number_of_iterations}', flush=True)
def getamgcycle(min_level, max_level, grammar_string):
    cwd = os.getcwd()
    program_generator = ProgramGenerator(min_level,max_level ,0)
   # Obtain extracted information from program generator
    dimension = 2#program_generator.dimension  # Dimensionality of the problem
    finest_grid = 'u'#program_generator.finest_grid  # Representation of the finest grid
    coarsening_factors = [2,2]#program_generator.coarsening_factor
    min_level = program_generator.min_level  # Minimum discretization level
    max_level = program_generator.max_level  # Maximum discretization level
    equations = [] #program_generator.equations  # System of PDEs in SymPy
    operators = [] #program_generator.operators  # Discretized differential operators
    fields = [sympy.Symbol('u')] #program_generator.fields  # Variables that occur within system of PDEs
    for i in range(min_level, max_level + 1):
        equations.append(initialization.EquationInfo('solEq', i, f"( Laplace@{i} * u@{i} ) == RHS_u@{i}"))
        operators.append(initialization.OperatorInfo('RestrictionNode', i, None, base.Restriction))
        operators.append(initialization.OperatorInfo('ProlongationNode', i, None, base.Prolongation))
        operators.append(initialization.OperatorInfo('Laplace', i, None, base.Operator))
    size = 2 ** max_level
    grid_size = tuple([size] * dimension)
    h = 1 / (2 ** max_level)
    step_size = tuple([h] * dimension)
    tmp = tuple([2] * dimension)
    coarsening_factors = [tmp for _ in range(len(fields))]
    finest_grid = [base.Grid(grid_size, step_size, max_level) for _ in range(len(fields))]
    problem_name = "test_anisotropic"#program_generator.problem_name
    solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
    approximation = system.Approximation('x', solution_entries)
    rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
    rhs = system.RightHandSide('b', rhs_entries)
    if not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          program_generator=program_generator,
                          epsilon=1e-12, infinity=1e100)
    solver_program = ''

    approximation = optimizer.approximation
    rhs = optimizer.rhs
    levels = program_generator.max_level - program_generator.min_level
    pset, terminal_list = \
        initialization.generate_primitive_set(approximation, rhs, dimension,
                                                        coarsening_factors, max_level, equations,
                                                        operators, fields,
                                                        maximum_local_system_size=8,
                                                        depth=max_level - min_level)
    expression, _ = eval(grammar_string, pset.context, {})
    return program_generator.generate_cycle_function(expression)
def viz_cycle(grammar_string,save_path):
        num_levels = 5
        amg_cycle_spec = getamgcycle(0, num_levels-1, grammar_string)
        amg_cycle_spec = amg_cycle_spec.split("/")
        # first entry is the number of nodes
        n_nodes = int(amg_cycle_spec[0])
        # second entry is a list to specify inter-grid transfer operators
        inter_grid_transfer_operators = [int(i) for i in amg_cycle_spec[1].split(",")]
        # third entry is a list to specify smoother types
        smoother_types = [int(i) for i in amg_cycle_spec[2].split(",")]
        # fourth entry is a list to specify smoother sweeps
        smoother_sweeps = [int(i) for i in amg_cycle_spec[3].split(",")]
        # fifth entry is a list to specify smoother relaxation factors
        smoother_relaxation_factors = [round(float(i),2) for i in amg_cycle_spec[4].split(",")]
        cgc_relaxation_factors = [round(float(i),2) for i in amg_cycle_spec[5].split(",")]

        # Create a multigrid graph
        import networkx as nx
        G = nx.DiGraph()

        # construct dictionary mapping smoother types to node colors
        smoother_colors = {0: 'skyblue', 9: 'black', 13: 'orange', 14: 'red',-1: 'white'}
        smoother_names = {0: 'Jacobi', 9: 'CGS', 13: 'GS-Forward', 14: 'GS-Backward',-1: 'No Smoother'}

        # construct a list of coordinate tuples for each node
        coord_nodes = []
        node_colors =[]
        y=0
        x=0
        n_nodes = len(inter_grid_transfer_operators)+1
        for i in range(n_nodes):
            coord_nodes.append((x,y))
            node_colors.append(smoother_colors[smoother_types[i]])
            if i<n_nodes-1:
                y+=inter_grid_transfer_operators[i]
                if inter_grid_transfer_operators[i] == 0:
                    x+=2
                else:
                    x+=1
        
        G.add_nodes_from(coord_nodes) # add nodes to graph
        # construct list of edges, each edge is a tuple of two coordinate tuples, connecting consecutive nodes in the coord_nodes list
        edges = []
        for i in range(len(coord_nodes)-1):
            edges.append((coord_nodes[i],coord_nodes[i+1]))
        G.add_edges_from(edges) # add edges to graph

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 4))

        # Draw nodes
        # create a pos dictionary mapping each node to its coordinate tuple
        pos = {}
        for i in range(len(coord_nodes)):
            pos[coord_nodes[i]] = coord_nodes[i]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color=node_colors, edgecolors='black', linewidths=1.5)

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, width=1, edge_color='lightgray', arrowsize=20)

        # Annotate nodes with smoother information
        for i,relax_factor in enumerate(smoother_relaxation_factors):
            ax.annotate(relax_factor, coord_nodes[i], fontsize=12, ha='center', va='center', color='black')

        # Annotate edges with cgc relaxation factor
        for i,relax_factor in enumerate(cgc_relaxation_factors):
            if inter_grid_transfer_operators[i] == 1:
                ax.annotate(relax_factor, ((coord_nodes[i][0]+coord_nodes[i+1][0])/2,(coord_nodes[i][1]+coord_nodes[i+1][1])/2), fontsize=12, ha='center', va='center', color='black')
        # remove axis
        ax.axis('off')
        
        # add legends for node colors with edge colors
        handles = []
        labels = []
        for smoother in smoother_names:
            handles.append(patches.Patch(color=smoother_colors[smoother], label=smoother))
            labels.append(smoother_names[smoother])
        legends = plt.legend(handles, labels, title="Smoother Types", loc='lower left')
        for legend in legends.legendHandles:
            legend.set_edgecolor('black')
        
        # Set plot title
        plt.title('G3P AMG Cycle')

        # Show the plot
        plt.tight_layout()
        plt.savefig(save_path)
def print_jobscript(folder_path, tag, amg_args, problem="poisson", precond=False, scaling="weak"):
    with open(f"{folder_path}/job_{tag}.sh", 'w', newline='\n') as f:
        print("#!/bin/bash -l", file=f)
        print(f"#SBATCH --cpus-per-task=1", file=f)
        if scaling is "weak" and problem is "poisson":
            print(f"#SBATCH --nodes=6", file=f)
        else:
            print(f"#SBATCH --nodes=1", file=f)
        print(f"#SBATCH -p pbatch", file=f)
        print(f"#SBATCH -A paratime", file=f)
        print(f"#SBATCH --ntasks-per-node=36", file=f)
        print(f"#SBATCH --time=00:5:00", file=f)
        print(f"#SBATCH --job-name=evostencils_{tag}", file=f)
        print(f"#SBATCH --export=NONE", file=f)
        print(f"#SBATCH --output=out_evo_{tag}", file=f)
        print(f"unset SLURM_EXPORT_ENV", file=f)
        print(f"cd /g/g91/parthasa/hypre/src/test", file=f)



        if precond:
            cmd_line_args = "-solver 1 -x0rand -rhsisone"
        else:
            cmd_line_args = "-solver 0 -x0rand -rhszero"

        if scaling is "strong":
            num_threads_list = [1, 4, 8, 16 , 18, 20, 24, 32, 36]
            for num_threads in num_threads_list:
                print(f"export OMP_NUM_THREADS={num_threads}",file=f)
                if problem is "poisson":
                    print(f"./ij -n 100 100 100 -c 0.001 1 1 {cmd_line_args} -pout 0 -amgusrinputs 1 {amg_args[0]}",file=f)                
        elif scaling is "weak":
            if problem is "ares":
                print(f"echo \"level 5\"",file=f)
                print(f"srun -n 1 ./ij -fromfile $HOME/ares_matrices/ares_matrix_4 -solver 1 -pout 0 -amgusrinputs 1 {amg_args[0]}",file=f)
                print(f"echo \"level 6\"",file=f)
                print(f"srun -n 1 ./ij -fromfile $HOME/ares_matrices/ares_matrix_8 -solver 1 -pout 0 -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"echo \"level 7\"",file=f)
                print(f"srun -n 1 ./ij -fromfile $HOME/ares_matrices/ares_matrix_16 -solver 1 -pout 0 -amgusrinputs 1 {amg_args[2]}",file=f)
                print(f"echo \"level 8\"",file=f) 
                print(f"srun -n 4 ./ij -fromfile $HOME/ares/Ares_Matrix_num_1.UMatrix -solver 1 -pout 0 -amgusrinputs 1 {amg_args[3]}",file=f)
            elif problem is "poisson":
                print(f"echo \"level 10\"",file=f)
                print(f"srun -n 1 ./ij -n 100 100 100 -P 1 1 1 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[0]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_1",file=f)
                print(f"echo \"level 11\"",file=f)
                print(f"srun -n 8 ./ij -n 200 200 200 -P 2 2 2 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_8",file=f)
                print(f"echo \"level 11\"",file=f)
                print(f"srun -n 27 ./ij -n 300 300 300 -P 3 3 3 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_27",file=f)
                print(f"echo \"level 11\"",file=f) 
                print(f"srun -n 64 ./ij -n 400 400 400 -P 4 4 4 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_64",file=f)
                print(f"echo \"level 11\"",file=f)
                print(f"srun -n 125 ./ij -n 500 500 500 -P 5 5 5 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_125",file=f)
                print(f"echo \"level 11\"",file=f)
                print(f"srun -n 216 ./ij -n 600 600 600 -P 6 6 6 -c 0.001 1 1 {cmd_line_args} -pout 0 -printsolution -amgusrinputs 1 {amg_args[1]}",file=f)
                print(f"mv $HOME/hypre/src/test/x.out_0.0 x.out_0.0_{tag}_np_216",file=f)

def parse_logoutput(logfile):
    n_iterations = []
    with open(logfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Iterations" in line:
                match = re.search(r'\d+', line)
                if match:
                    n_iterations.append(int(match.group()))
    return n_iterations

if __name__ == "__main__":
    min_level = [6,7]
    max_level = [10,11]
    scaling_type = "weak"
    problem_name = "poisson"
    amg_precond = False
    viz_cycle_struct = False
    print_jobs = False
    plot_output = False
    problem_folder = "test2_anisotropic"
    cwd = os.getcwd()
    experiment_folder = "data_4"

    if scaling_type is not None:
        folder_path = f"{cwd}/{problem_folder}/{experiment_folder}/hof_0"
        save_results_folder = f"{cwd}/{problem_folder}/{experiment_folder}/{scaling_type}_scaling"
        if not os.path.exists(save_results_folder):
            os.makedirs(save_results_folder)
        # find all text files in folder
        # files_list = os.listdir(folder_path)
        files_list = ["individual_0.txt", "individual_1.txt", "individual_2.txt", "individual_20.txt", "individual_30.txt", "individual_40.txt"]
        #files_list = ["individual_0.txt"]
        #files_list = ["individual_0.txt", "individual_4.txt", "individual_8.txt", "individual_16.txt"]
        #files_list = ["individual_0.txt", "individual_4.txt", "individual_8.txt", "individual_16.txt","individual_24.txt", "individual_32.txt", "individual_40.txt", "individual_48.txt"]
        #files = [f for f in files_list if os.path.isfile(os.path.join(folder_path, f))]
        # read string from files
        for file in files_list:
            amg_args = []
            with open(f"{folder_path}/{file}", 'r') as f:
                grammar_string = f.read()
            tag = experiment_folder + "_" + file.split(".")[0]
            for i in range(len(min_level)):
                amg_args.append(getamgcycle(min_level[i], max_level[i], grammar_string))
            print_jobscript(save_results_folder, tag, amg_args,problem=problem_name,precond=amg_precond,scaling=scaling_type)
            subprocess.check_call(["sbatch", f"job_{tag}.sh"], cwd=save_results_folder)

    if viz_cycle_struct:
        folder_path = f"{cwd}/{problem_folder}/{experiment_folder}/hof_0"
        files_list = os.listdir(folder_path)
        files = [f for f in files_list if os.path.isfile(os.path.join(folder_path, f))]
        save_results_folder = f"{cwd}/{problem_folder}/{experiment_folder}/solvers"
        if not os.path.exists(save_results_folder):
            os.makedirs(save_results_folder)
        # read string from files
        for file in files:
            amg_args = []
            with open(f"{folder_path}/{file}", 'r') as f:
                grammar_string = f.read()
            tag = experiment_folder + "_" + file.split(".")[0]
            save_path = save_results_folder + "/" + tag + ".png"
            viz_cycle(grammar_string,save_path)

    if plot_output:
        n_procs = [1,8,27,64,125,216]
        # find all out_evo files in save_results_folder
        files = [f for f in os.listdir(save_results_folder) if os.path.isfile(os.path.join(save_results_folder, f)) and "out_evo" in f]
        # read string from files
        for file in files:
            n_iterations = parse_logoutput(f"{save_results_folder}/{file}")
            file_no = file.split("_")[-1]
            # use last len(n_procs) entries of n_iterations
            n_iterations = n_iterations[-len(n_procs):]
            plt.plot(n_procs, n_iterations, label=file_no)
        # mark x label as number of processes
        plt.xticks(n_procs)
        # wrap with seaborn
        import seaborn as sns
        sns.set()
        # set labels
        plt.xlabel("Number of Processes")
        plt.ylabel("Number of Iterations")
        plt.legend()
        #save plot as pdf   
        plt.savefig(f"{save_results_folder}/weak_scaling_4nodes.pdf")
