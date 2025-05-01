# First try out everything for the local build then upload to pypi.
local_build=1
check_pep=1
pip uninstall -y app-come-before
rm dist/*
python3 setup.py sdist bdist_wheel
if [ "$local_build" -eq 1 ]; then
  pip install dist/app_come_before-0.1.5-py3-none-any.whl
  pytest -v --capture=no
  status=$?
  if [ $status -ne 0 ]; then
    echo "One of the tests is failing, prematurely exiting the build"
    exit
  fi
  if [ "$check_pep" -eq 1 ]; then
    pylint --recursive=y .
    status=$?
    if [ $status -ne 0 ]; then
      echo "Code is not properly formatted"
      exit
    fi
  fi
  pylint --recursive=y .
else
  twine upload dist/* --verbose
  pip install app-come-before==0.1.5
fi

app-come-before
