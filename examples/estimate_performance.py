import sys, os
from evostencils.code_generation.exastencils import ProgramGenerator


def main():
    if len(sys.argv[1:]) == 2 and os.path.exists(sys.argv[2]):
        problem_name = sys.argv[1]
        exastencils_path = sys.argv[2]
        dimensions = 2
        min_level = 2
        max_level = 10
        program_generator = ProgramGenerator(problem_name, exastencils_path,
                                             None, None, None, None, None, None, dimensions, None, min_level, max_level, None)
        runtime_per_iteration = program_generator.estimate_runtime_per_iteration()
        print(f'Estimated runtime per iteration: {runtime_per_iteration} ms')


if __name__ == "__main__":
    main()

