import setuptools

setuptools.setup(
    name='trav_lib',
    version = '0.0.1',
    author = 'Travis Whitfield',
    description = "Travis's python package with useful custom functions and classes for data analysis.",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires = '>=3.7',
    install_requires = [
        "matplotlib",
        "pandas",
        "seaborn",
        "numpy",
        "sklearn",
        "notebook"
    ]
)