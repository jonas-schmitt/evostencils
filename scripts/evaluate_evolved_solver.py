import os,sys
# Compute the path to the directory containing `evostencils`.
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add this path to sys.path.
sys.path.append(parent_dir)
sys.path.append(os.getcwd())

from matplotlib import patches
from evostencils.optimization.program import Optimizer
from evostencils.code_generation.hyteg import ProgramGenerator
from evostencils.grammar import multigrid as initialization
from evostencils.ir import base, system
import sympy
from mpi4py import MPI
import subprocess,re
import matplotlib.pyplot as plt
from deap import tools, gp


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


def create_optimizer_pset(min_level, max_level):
    # Set up MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    dimension = 2
    finest_grid = 'u'
    coarsening_factors = [2, 2]
    equations = []
    operators = []
    fields = [sympy.Symbol('u')]
    for i in range(min_level, max_level + 1):
        equations.append(initialization.EquationInfo('solEq', i, f"( xxxxxxx@xxx * x@xxx ) == xxxxx@xxx"))
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
    convergence_evaluator = None
    performance_evaluator = None
    checkpoint_directory_path = None
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                            mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=1,
                            program_generator=program_generator,
                            convergence_evaluator=convergence_evaluator,
                            performance_evaluator=performance_evaluator,
                            checkpoint_directory_path=checkpoint_directory_path)
    
    pset, _ = \
    initialization.generate_primitive_set(optimizer.approximation, optimizer.rhs, dimension,
                                                    coarsening_factors, max_level, equations,
                                                    operators, fields,
                                                    maximum_local_system_size=8,
                                                    depth=max_level - min_level)
    return optimizer, pset

def print_jobscript(save_path, build_path, tag, cmd_line_args):
    with open(f"{save_path}/job_{tag}.sh", 'w', newline='\n') as f:
        print("#!/bin/bash -l", file=f)
        print(f"#SBATCH --nodes=1", file=f)
        print(f"#SBATCH --time=00:30:00", file=f)
        print(f"#SBATCH --cpu-freq=2200000-2200000", file=f)
        print(f"#SBATCH --job-name=evostencils_{tag}", file=f)
        print(f"#SBATCH --export=NONE", file=f)
        print(f"#SBATCH --output=out_evo_{tag}", file=f)
        print(f"unset SLURM_EXPORT_ENV", file=f)
        print(f"cd {build_path}", file=f)


        for cmd_line_arg in cmd_line_args:
            cmd_line_arg = " ".join(cmd_line_arg)
            print(f"./{cmd_line_arg}",file=f)


if __name__ == "__main__":
    number_of_flexible_levels = 5
    max_level_list = [4,5,6,7,8,9]
    experiment_folder = "/home/hpc/iwia/iwia058h/evostencils/2dpoisson_allsmoothers"
    number_of_experiments = 5
    binary = "MultigridStudies"
    build_path = "/home/hpc/iwia/iwia058h/hyteg-build_local/apps/MultigridStudies"
    save_path = f"{experiment_folder}/scaling"
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    hof_indices = [10,24,33,41,56]
    cgs_level = 0
    
    '''
    for max_level in max_level_list:
        program_generator = ProgramGenerator(max_level - number_of_flexible_levels + 1, max_level, cgs_level)
        optimizer, pset = create_optimizer_pset(max_level - number_of_flexible_levels + 1, max_level)
        global_hof = tools.ParetoFront()
        for j in range(number_of_experiments):
            pop = optimizer.load_data_structure(f'{experiment_folder}/data_{j}/pop_0.p')
            global_hof.update(pop)

        cmd_line_args_list = [] 
        for index in hof_indices:
            cmd_line_args = [binary, f"{binary}.prm"]
            individual = global_hof[index]  
            expression, _  = gp.compile(individual, pset)
            cmd_line_args += program_generator.generate_cycle_function(expression)
            cmd_line_args_list.append(cmd_line_args)

        print_jobscript(save_path, build_path, f"maxLvl{max_level}", cmd_line_args_list)
        subprocess.run(["sbatch", f"job_maxLvl{max_level}.sh"], cwd=save_path)
    ''' 
    
    for max_level in max_level_list:
        print_jobscript(save_path, f"maxLvl{max_level}", f"refsolvers_maxLvl{max_level}", [[binary, f"{binary}.prm", "-preSmoothingSteps", 1 ,"-postSmoothingSteps" ,1 ,"-maxLevel", max_level],
                                                                                           [binary, f"{binary}.prm", "-preSmoothingSteps", 2 ,"-postSmoothingSteps" ,2 ,"-maxLevel", max_level],
                                                                                           [binary, f"{binary}.prm", "-preSmoothingSteps", 3 ,"-postSmoothingSteps" ,3 ,"-maxLevel", max_level]])
        subprocess.run(["sbatch", f"job_refsolvers_maxLvl{max_level}.sh"], cwd=save_path)