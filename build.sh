local_build=0

pip uninstall -y app-come-before
rm dist/*
python3 setup.py sdist bdist_wheel

if [ "$local_build" -eq 1 ]; then
  pip install dist/app_come_before-0.1.3-py3-none-any.whl
else
  twine upload dist/* --verbose
  pip install app-come-before==0.1.4
fi

app-come-before
