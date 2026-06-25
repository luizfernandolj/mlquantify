"""Build script for mlquantify.

Static project metadata lives in ``pyproject.toml`` (PEP 621). This file only
handles the two things that cannot be expressed there:

* the version, which is read from ``VERSION.txt`` (generated at release time
  from the git tag), with a fallback for local source builds; and
* the optional compiled Cython kernel, which accelerates distribution matching
  but is not required -- the package falls back to a pure-Python implementation
  when the extension is unavailable.
"""

import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

# Read the version from VERSION.txt (written by CI from the git tag). Outside
# CI the file may be absent, so fall back to a PEP 440-valid dev version.
version_file = here / "VERSION.txt"
if version_file.exists():
    VERSION = version_file.read_text(encoding="utf-8").strip()
else:
    VERSION = "0.5.0.dev0"

# --- optional Cython acceleration -------------------------------------------
# Compiled kernels are an optimisation: if Cython/numpy/a compiler are missing
# the package still installs and runs via the pure-Python fallbacks.
ext_modules = []
try:
    from setuptools import Extension
    from Cython.Build import cythonize
    import numpy as _np

<<<<<<< HEAD
# --- optional Cython acceleration -------------------------------------------
# Compiled kernels are an optimisation: if Cython/numpy/a compiler are missing
# the package still installs and runs via the pure-Python fallbacks.
ext_modules = []
try:
    from setuptools import Extension
    from Cython.Build import cythonize
    import numpy as _np

=======
>>>>>>> master
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

<<<<<<< HEAD
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
=======
setup(version=VERSION, ext_modules=ext_modules)
>>>>>>> master
