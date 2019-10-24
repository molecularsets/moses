from setuptools import setup, find_packages
import moses


setup(name='molsets',
      version=moses.__version__,
      python_requires='>=3.5.0',
      packages=find_packages() + ['moses/metrics/SA_Score',
                                  'moses/metrics/NP_Score'],
      install_requires=[
          'tqdm>=4.26.0',
          'matplotlib>=3.0.0',
          'numpy>=1.15',
          'pandas>=0.25',
          'scipy>=1.1.0',
          'torch==1.1.0',
          'fcd_torch>=1.0.5',
          'seaborn>=0.9.0'
      ],
      description=('Molecular Sets (MOSES): '
                   'A Benchmarking Platform for Molecular Generation Models'),
      author='Insilico Medicine',
      author_email='moses@insilico.com',
      license='MIT',
      package_data={
          '': ['*.csv', '*.h5', '*.gz'],
      }
      )
