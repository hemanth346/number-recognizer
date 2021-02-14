import setuptools

with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='number-recognizer',
    version='0.0.1',
    author='Hemanth Reddy K',
    author_email='hemanth346@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    include_package_data=True,
)
