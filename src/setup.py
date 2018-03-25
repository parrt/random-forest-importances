from setuptools import setup

setup(
    name='rfimpo',
    version='1.0',
    url='https://github.com/parrt/random-forest-importances',
    license='MIT',
    py_modules=['rfimpo'],
    author='Terence Parr, Kerem Turgutlu',
    author_email='parrt@antlr.org, kcturgutlu@dons.usfca.edu',
    install_requires=['numpy','pandas','sklearn','matplotlib'],
    description='Permutation and drop-column importance for scikit-learn random forests',
    keywords='scikit-learn random forest feature importances',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
