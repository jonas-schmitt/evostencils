import sys, pickle, os
from evostencils.optimization.program import Optimizer


def main():
    if len(sys.argv[1:]) > 0 and os.path.exists(sys.argv[1]):
        path_to_log_file = sys.argv[1]
        log = pickle.load(open(path_to_log_file, "rb"))
        Optimizer.plot_average_fitness(log)


if __name__ == "__main__":
    main()

