from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

VERSION = '0.1.1'
DESCRIPTION = 'Quantification Library'

# Setting up
setup(
    name="mlquantify",
    version=VERSION,
    url="https://github.com/luizfernandolj/QuantifyML/tree/master",
    maintainer="Luiz Fernando Luth Junior",    
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['scikit-learn', 'numpy', 'scipy', 'joblib', 'tqdm', 'pandas', 'xlrd', 'matplotlib'],
    keywords=['python', 'machine learning', 'quantification', 'quantify'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)