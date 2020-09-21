import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensor-canvas",
    version="0.1",
    author="Peter Whidden",
    author_email="all.cows.like.to.moo@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pwhiddy/tensor-canvas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)