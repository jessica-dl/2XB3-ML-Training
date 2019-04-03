from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='example to run keras on gcloud ml-engine',
      install_requires=[
          'tensorflow',
          'keras',
          'h5py',
          'numpy',
          'opencv-python',
          'matplotlib',
          'h5py'
      ],
      zip_safe=False)
