from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Lê a versão do arquivo VERSION.txt
version_file = here / 'VERSION.txt'
VERSION = version_file.read_text(encoding='utf-8').strip()

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
                "mlquantify.matching._matching_fast",
                ["mlquantify/matching/_matching_fast.pyx"],
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
    url="https://github.com/luizfernandolj/QuantifyML/tree/master",
    maintainer="Luiz Fernando Luth Junior",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=['scikit-learn', 'numpy', 'scipy', 'joblib', 'tqdm', 'pandas', 'xlrd', 'matplotlib', 'abstention'],
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
