from setuptools import setup

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name='PyRoot',
    version='0.3.1',
    description='The purpose of this Python library is to provide implementations of advanced bracketed root-finding methods for single-variable functions.',
    packages=["pyroot"],
    python_requires='>=3.8',
    url="https://github.com/SimpleArt/pyroot",
    author="Jack Nguyen",
    author_email="jackyeenguyen@gmail.com",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
