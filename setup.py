from setuptools import setup, find_packages

setup(name='ProNE',
      version='1.0.3',
      description="Graph Embedding Algorithms: ProNE",
      author="THUDN",
      maintainer="nlp4whp",
      url="https://github.com/nlp4whp/ProNE",
      install_requires=[
        'numpy', 'scipy', 'sklearn'
      ],
      packages=find_packages())