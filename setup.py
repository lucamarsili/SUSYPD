from setuptools import setup,find_packages

setup(
    name='SUSYPD',
    version = '0.1',
    description = 'proton decay code',
    author = 'L.Marsili',
    author_email = 'luca.marsili@durham.ac.uk',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy',
        ],
    python_requires = '>3.6.0',
)
