from distutils.core import setup

setup(
    name = "MLP",
    version = "0.1",
    author = "Marcel Rodekamp",
    author_email = "marcel.rodekamp@gmail.com",
    package_dir = {
        "MLP": "src/",
        "MLP.activation": "src/Activations/",
        "MLP.layer": "src/Layers/",
        "MLP.loss": "src/LossFunctions/",
        "MLP.complexifyTorch": "src/ComplexifyTorch/",
    },
    packages = [
        "MLP",
        "MLP.activation",
        "MLP.layer",
        "MLP.loss",
        "MLP.complexifyTorch",
    ],
)
