from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Lê a versão do arquivo VERSION.txt (gerado no CI). Fora do CI o arquivo pode
# não existir, então usamos um fallback PEP 440 válido para builds locais.
version_file = here / 'VERSION.txt'
if version_file.exists():
    VERSION = version_file.read_text(encoding='utf-8').strip()
else:
    VERSION = '0.4.0.dev0'

DESCRIPTION = 'Quantification Library'

# --- optional Cython acceleration -------------------------------------------
# Compiled kernels are an optimisation: if Cython/numpy/a compiler are missing
# the package still installs and runs via the pure-Python fallbacks.
ext_modules = []
try:
    from setuptools import Extension
    from Cython.Build import cythonize
    import numpy as _np

    ext_modules = cythonize(
        [
            Extension(
                "mlquantify.matching._histogram_sweep",
                ["mlquantify/matching/_histogram_sweep.pyx"],
                include_dirs=[_np.get_include()],
            ),
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except Exception as _exc:  # pragma: no cover
    import warnings
    warnings.warn(f"mlquantify: building without Cython acceleration ({_exc}).")

setup(
    ext_modules=ext_modules,
    name="mlquantify",
    version=VERSION,
    url="https://github.com/luizfernandolj/mlquantify",
    project_urls={
        "Documentation": "https://luizfernandolj.github.io/mlquantify/",
        "Source": "https://github.com/luizfernandolj/mlquantify",
        "Issue Tracker": "https://github.com/luizfernandolj/mlquantify/issues",
    },
    author="Luiz Fernando Luth Junior, André Gustavo Maletzke",
    maintainer="Luiz Fernando Luth Junior",
    maintainer_email="luizfernandoluth@gmail.com",
    license="BSD-3-Clause",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        'scikit-learn>=1.1',
        'numpy>=1.23',
        'scipy>=1.8',
        'joblib>=1.1',
        'tqdm>=4.60',
        'pandas>=1.4',
        'xlrd>=2.0',
        'matplotlib>=3.5',
        'abstention>=0.1.3.1',
    ],
    keywords=['python', 'machine learning', 'quantification', 'quantify'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
