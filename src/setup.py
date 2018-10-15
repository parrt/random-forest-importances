from setuptools import setup

long_description = """A library that provides feature importances, based upon
the permutation importance strategy, for general scikit-learn
models and implementations specifically for random forest out-of-bag scores.
Built by Terence Parr and Kerem Turgutlu.
See <a href="http://explained.ai/rf-importance/index.html">Beware Default
Random Forest Importances</a> for a deeper discussion of the issues surrounding
feature importances in random forests.
"""

setup(
    name='rfpimp',
    version='1.2.2',
    url='https://github.com/parrt/random-forest-importances',
    license='MIT',
    py_modules=['rfpimp'],
    author='Terence Parr, Kerem Turgutlu',
    author_email='parrt@antlr.org, kcturgutlu@dons.usfca.edu',
    install_requires=['numpy','pandas','sklearn','matplotlib'],
    description='Permutation and drop-column importance for scikit-learn random forests and other models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='scikit-learn random forest feature permutation importances',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
