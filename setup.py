import os
from pathlib import Path
from setuptools import setup,Extension,find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig

# this setup and subsequent CMakeLists.txt are inspired by
# https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py/48015772#48015772
# https://pytorch.org/tutorials/advanced/torch_script_custom_ops

VERSION = "0.1"#setup.version_from_git(plain=True)

PROJECT_ROOT = Path(__file__).resolve().parent

BUILD_DIR = PROJECT_ROOT/"build"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        """
            name: string
                Name of the extension
            sourcedir: string
                Directory of the root CMakeLists.txt
        """

        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()

class build_ext(build_ext_orig):
    def run(self):
        # forpass cmake
        for ext in self.extensions:
            self.build_cmake(ext)
        # perform subsequent processing
        super().run()

    def build_cmake(self, ext):
        pwd = Path().absolute()

        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)
        extdir = Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        config = 'Debug' if self.debug else 'Release'
        cmake_args = []

        build_args = [
            '--', '-j4'
        ]

        os.chdir(str(build_lib))
        self.spawn(['cmake', str(pwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)

        os.chdir(str(pwd))

complexifyTorch = CMakeExtension(
    name="ComplexifyTorch_cpp",
    sourcedir="src/ComplexifyTorch"
)

setup_kwargs = dict(
    name = "MLP",
    version = VERSION,
    author = "Marcel Rodekamp",
    author_email = "marcel.rodekamp@gmail.com",
    description="Machine Learning Parameterization of Holomorphic Flow Equation",
    long_description="This package applies machine learning techniques to "
                    +"parametetrize a holomorphic flow equation, in lattice field theory, "
                    +"by applying complex valued neuronal networks. "
                    +"These networks are developed using the PyTorch interface.",
    license="MIT",
    # define a package structure
    package_dir = {
        "MLP": "src/",
        "MLP.complexifyTorch": "src/ComplexifyTorch/",
        "MLP.activation": "src/Activations/",
        "MLP.layer": "src/Layers/",
        "MLP.loss": "src/LossFunctions/",
        "MLP.data": "src/DataHandling/",
    },
    packages = [
        "MLP",
        "MLP.complexifyTorch",
        "MLP.activation",
        "MLP.layer",
        "MLP.loss",
        "MLP.data",
    ],
    ext_modules=[
        complexifyTorch,
    ],
    cmdclass={
        'build_ext': build_ext,
    },
    package_data={'':['libcomplexifyTorch_cpp.so']},
    zip_safe=False,
    install_requires=["torch","isle"]
    # isle already requires: "numpy", "PyYAML", "h5py", "pybind11", "scipy", "pentinsula", "psutil"
)

if __name__ == '__main__':
    setup(**setup_kwargs)
