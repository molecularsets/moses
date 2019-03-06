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
          'pandas>=0.23',
          'scipy>=1.1.0',
          'torch>=1.0.1',
          'fcd_torch>=1.0.5',
          'seaborn'
      ],
      description=('Molecular Sets (MOSES): '
                   'A benchmarking platform for molecular generation models'),
      author='Neuromation & Insilico Medicine Teams',
      author_email='developers@neuromation.io, zhebrak@insilico.com',
      license='MIT',
      package_data={
          '': ['*.csv', '*.h5', '*.gz'],
      }
      )
