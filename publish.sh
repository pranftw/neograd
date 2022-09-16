if [ -d "dist" ]; then
  rm -rf dist build
  rm -rf neograd.egg-info
fi

source ~/.bash_profile
source ~/.bashrc

repo=$1

cd tests
uenv venv
cd ..
pip install --upgrade setuptools
pip install wheel
pip install twine
python setup.py sdist bdist_wheel
twine upload -r $repo dist/*
deactivate

if [ -d "dist" ]; then
  rm -rf dist build
  rm -rf neograd.egg-info
fi
