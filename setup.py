import setuptools


setuptools.setup(
    name="HCmanager",
    version="0.0.1",
    description="Library that manages different Hierarchical Clustering algorithms. Allows to run all of them on the same dataset and do comparisons",
    url="https://github.com/SebastianMacaluso/HCmanager",
    author="",
    author_email="seb.macaluso@nyu.edu",
    license="MIT",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
