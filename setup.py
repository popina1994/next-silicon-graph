from setuptools import setup, find_packages

setup(
    name='app-come-before',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[
        'flask',
        'pygraphviz',
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'app-come-before=server.app:main',
        ]
    }
)
