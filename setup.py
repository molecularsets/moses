from setuptools import setup, find_packages

setup(name='mnist4molecules',
      version='0.1.0',
      python_requires='>=3.5.0',
      packages=find_packages() + ['mnist4molecules/metrics/SA_Score',
                                  'mnist4molecules/metrics/NP_Score'],
      package_data={
        '': ['*.csv', '*.h5', '*.txt', '*.gz'],
      },
      # Make sure to pin versions of install_requires
      install_requires=[],
      # TODO
      # description='??',
      author='Neuromation Team',
      author_email='engineering@neuromation.io',
      # TODO (rauf): make repo public and update URL
      # TODO license='??'
      )
