from distutils.core import setup

setup(name='evostencils',
      version='1.0',
      description='Automated Multigrid Solver Design with Evolutionary Program Synthesis',
      author='Jonas Schmitt',
      author_email='jonas.schmitt@fau.de',
      url='https://github.com/jonas-schmitt/evostencils',
      packages=['evostencils'],
      install_requires=[
          'sympy', 'deap', 'mpi4py', 'pytest'
      ],
     )

