from setuptools import setup, find_packages

def read_file(fpath):
  with open(fpath) as fp:
    data = fp.read()
  return data

setup(
  name = 'neograd',
  version = '0.0.4',
  author = 'Pranav Sastry',
  author_email = 'pranava.sri@gmail.com',
  maintainer = 'Pranav Sastry',
  url = 'https://github.com/pranftw/neograd',
  python_requires = '>=3.7',
  install_requires = read_file('requirements.txt').split('\n'),
  description = 'A deep learning framework created from scratch with Python and NumPy',
  license = 'GPL-3.0',
  keywords = 'python  ai  deep-learning  numpy automatic-differentiation  autograd neural-networks  pytorch-api',
  packages = find_packages(),
  long_description = read_file('README.md'),
  long_description_content_type='text/markdown'
)