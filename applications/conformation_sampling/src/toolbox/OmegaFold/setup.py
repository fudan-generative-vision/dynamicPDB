from setuptools import setup, find_packages
import sys

with open("README.md", "r") as f:
    readme = f.read()

def get_url() -> str:
    if sys.version_info[:2] == (3, 8):
        _ver = "cp38"
    elif sys.version_info[:2] == (3, 9):
        _ver = "cp39"
    elif sys.version_info[:2] == (3, 10):
        _ver = "cp310"
    else:
        raise Exception(f"Python {sys.version} is not supported.")

    # FIXME: how to download on macox??
    if sys.platform == "win32":
        _os = "win_amd64"
    else:
        _os = "linux_x86_64"

    return f"https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-{_ver}-{_ver}-{_os}.whl"

setup(
    name="OmegaFold",
    description="OmegaFold Release Code",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={"console_scripts": ["omegafold=omegafold.__main__:main",],},
    install_requires=[
        "biopython",
        f"torch@{get_url()}"
    ],
    python_requires=">=3.8",
)
