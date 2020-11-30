import subprocess
import shutil
import numpy as np

def main():
    path_to_executable = "../exastencils/Examples/generated/2D_FD_Helmholtz_fromL3"
    output_path = path_to_executable
    path_to_file = f'{output_path}/Global/Global_declarations.cpp'
    values = np.linspace(0.5, 1.5, 21)
    for value in values:
        shutil.copyfile(path_to_file, f'{path_to_file}.backup')
        parameter = "omegaRelax"
        print(f"omegaRelax = {value}")
        with open(path_to_file, 'r') as file:
            lines = file.readlines()
            content = ''
            for line in lines:
                include_line = True
                if parameter in line:
                    tokens = line.split('=')
                    tokens[-1] = ' ' + str(value) + ';\n'
                    tmp = '='.join(tokens)
                    content += tmp
                    include_line = False
                if include_line:
                    content += line
        with open(path_to_file, 'w') as file:
            file.write(content)
        subprocess.run(['make', '-j12', '-s', '-C', f'{path_to_executable}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = subprocess.run([f'./{path_to_executable}/exastencils'],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        output = result.stdout.decode('utf8')
        print(output)
        shutil.copyfile(f'{path_to_file}.backup', path_to_file)

if __name__ == "__main__":
    main()
