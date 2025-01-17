from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="s2chunking",
    version="0.1.0", 
    author="Prashant Verma",
    author_email="prashant27050@gmail.com",
    description="A library for structural-semantic chunking of documents.",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    url="https://github.com/vprashant/s2-chunking-lib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True, 
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "s2chunking=s2chunking.cli:main",
        ],
    },
)