from distutils.core import setup
import sys

install_dir = sys.path[1]

setup(name = 'Anduril',
      version = '1.0',
      description = 'Anduril Neural Network library',
      author = 'ShahRukh Athar(SNU and MIPT)',
      author_email = 'srx.dev0619@gmail.com',
      py_modules = ['Anduril'],
      data_files = [(install_dir,['NNet.so'])],
      )
