import os
from setuptools import find_packages
from setuptools import setup

this_directory = os.path.dirname(__file__)
req_path = os.path.join(this_directory, "requirements_dion.txt")
req_dev_path = os.path.join(this_directory, "requirements_dev.txt")
req_train_path = os.path.join(this_directory, "requirements_train.txt")

def read_requirements(path):
    if not os.path.exists(path):
        print(f"Warning: requirements file {path} does not exist.")
        return []
    with open(path) as fp:
        return [
            line.strip()
            for line in fp
            if line.strip() and not line.startswith("#")
        ]

# requirements_dion contains the dependencies for the standalone optimizer
install_requires = read_requirements(req_path)

# requirements_dev contains the dependencies for development, e.g., testing, linting, etc.
install_dev_requires = install_requires + read_requirements(req_dev_path)

# requirements_train contains the dependencies for training, e.g., datasets, etc.
install_train_requires = install_requires + read_requirements(req_train_path)

readme_path = os.path.join(this_directory, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fp:
        readme_contents = fp.read().strip()

## uncomment the following lines to read the version from a file
## will be useful if you use tools like `bump2version` to manage versions
# with open(os.path.join(this_directory, "dion/VERSION")) as f:
#     version = f.read().strip()
version = "0.1.0"  # versions < 1.0 are considered pre-release versions, which allow for breaking changes if necessary

setup(
    # Name of the package:
    name="dion",
    # Version of the package:
    version=version,
    # Find the package automatically (include everything):
    packages=find_packages(include=["dion", "dion.*"]),
    ## uncomment the following line to include version file
    # package_data={
    #     "dion": ["VERSION"],  # Add the VERSION file
    # },
    # Author information:
    author="Ahn, Kwangjun and Xu, Byron and Abreu, Natalie and Langford, John", # as listed in the paper
    author_email="{kwangjunahn, byronxu}@microsoft.com", # left this form to prevent bots from harvesting emails
    # Description of the package:
    description="Dion: Distributed Orthonormal Updates.",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    # Plugins entry point
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "dev": install_dev_requires, # Can be installed with `pip install dion[dev]`
        "train": install_train_requires,
    },
)