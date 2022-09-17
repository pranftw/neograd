if [ -d "dist" ]; then
  rm -rf dist build
  rm -rf neograd.egg-info
fi

source ~/.bash_profile
source ~/.bashrc

repo=$1

if [ $repo == "testpypi" ]; then
  setup_file="testpypi_setup.py"
elif [ $repo == "pypi" ]; then
  setup_file="setup.py"
fi

cd tests
uenv venv
cd ..
pip install --upgrade setuptools
pip install wheel
pip install twine
python $setup_file sdist bdist_wheel
twine upload -r $repo dist/*
deactivate

if [ -d "dist" ]; then
  rm -rf dist build
  rm -rf neograd.egg-info
fi
