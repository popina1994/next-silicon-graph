# First try out everything for the local build then upload to pypi.
local_build=1

pip uninstall -y app-come-before
rm dist/*
python3 setup.py sdist bdist_wheel

if [ "$local_build" -eq 1 ]; then
  pip install dist/app_come_before-0.1.4-py3-none-any.whl
  pytest -v --capture=no
  status=$?
  if [ $status -ne 0 ]; then
    echo "One of the tests is failing, prematurely exiting the build"
  exit
  fi
else
  twine upload dist/* --verbose
  pip install app-come-before==0.1.4
fi

app-come-before
