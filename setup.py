from setuptools import setup, find_packages

setup(
    name="churn_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'matplotlib',
        'seaborn'
    ]
) 