import setuptools
import urllib.request, sys

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy',
                    'setuptools',
					'scipy']


setuptools.setup(
    name="lang2vec",
    version="1.1.6",
    author="Patrick Littell, David Mortensen, Antonis Anastasopoulos",
    author_email="aanastas@cs.cmu.com",
    description="A simple library for querying the URIEL typological database.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antonisa/lang2vec",
    install_requires=install_requires,
    scripts=['lang2vec/lang2vec.py'],
    packages=['lang2vec'],
    package_dir={'lang2vec': 'lang2vec'},
    package_data={'lang2vec': ['data/*.npz', 'data/*.json', 'data/distances2.zip', 'data/learned.npy', 'data/distances_languages.txt']},
    zip_safe=True,
    classifiers=['Operating System :: OS Independent',
               'Programming Language :: Python :: 3',
               'Topic :: Software Development :: Libraries :: Python Modules',
               'Topic :: Text Processing :: Linguistic']
)
