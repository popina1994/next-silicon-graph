local_build=0  # 1 represents true, 0 represents false

pip uninstall -y app-come-before
python3 setup.py sdist bdist_wheel

if [ "$local_build" -eq 1 ]; then
  pip install dist/app_come_before-0.1.3-py3-none-any.whl
else
  twine upload dist/* --verbose
  pip install app-come-befor=0.1.3
fi

app-come-before
