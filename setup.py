import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ab_tester", # Replace with your own username
    version="0.0.2",
    author="Vitaly Cheremisinov, Vladislav Kozlov",
    author_email="vlad.kozlov.ds@gmail.com",
    description="A package with statistical and analytic functions and classes for A/B testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)