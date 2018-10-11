from setuptools import setup, find_packages
import moses


setup(name='moses',
      version=moses.__version__,
      python_requires='>=3.5.0',
      packages=find_packages() + ['moses/metrics/SA_Score',
                                  'moses/metrics/NP_Score'],
      install_requires=[
          'tqdm==4.26.0',
          'keras==2.2.4',
          'matplotlib==3.0.0',
          'numpy==1.15.2',
          'pandas==0.23.4',
          'scipy==1.1.0',
          'tensorflow==1.11.0',
          'pytorch==0.4.1',
          # rdkit
      ],
      description='MOSES: A benchmarking platform for molecular generation models',
      author='Neuromation & Insilico Teams',
      author_email='engineering@neuromation.io',  # TODO: add email of insilico team
      # TODO: add license
      )
