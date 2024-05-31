from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    reqs = f.readlines()

setup(
   name='DeepTable',
   version='0.1',
   description='The module implements the DeepTable architecture',
   license='MIT',
   long_description='The module implements the DeepTable architecture. \
    It includes the DeepTable itself, \
        the Conditional GAN ​​based on it for generating and classifying tables \
            to evaluate the quality of work of various classical algorithms on different datasets, \
                the OpenMLDataset metadataset class, \
                    as well as the trainer necessary for training the GAN model and its evaluation.',
   author='Buda Vampilov (ITMO), Alexey Zabashta (ITMO)',
   author_email='fccfcbyjhtk@gmail.com',
   url='https://github.com/kupaqu/DeepTable.git',
   packages=find_packages(),
   install_requires=reqs,
)