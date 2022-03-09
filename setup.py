from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='PyRoot',
    version='0.2.0',
    description='The purpose of this Python library is to provide implementations of advanced bracketed root-finding methods for single-variable functions.',
    packages=["pyroot"],
    python_requires='>=3.6',
    url="https://github.com/SimpleArt/pyroot",
    author="Jack Nguyen, Daniel Wilczak",
    author_email="danielwilczak101@gmail.com",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "tabulate >=0.8.7",
        "numpy >=1.21.3"
    ],
)
