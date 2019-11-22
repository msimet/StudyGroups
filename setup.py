import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StudyGroups", # Replace with your own username
    version="0.1",
    author="Melanie Simet",
    author_email="melanie.simet@gmail.com",
    description="Divide a group of people into subgroups multiple times with minimal repeating of groups or pairs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msimet/StudyGroups",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pyyaml'],
)
