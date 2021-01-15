import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='coai',  
     version='0.1',
     author="Gabriel Erion",
     author_email="erion@uw.edu",
     description="Tools for making any predictive model cost-aware.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/suinleelab/coai",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering"
     ],
    install_requires=['numpy', 'shap', 'tqdm', 'sklearn', 'ortools', 'lightgbm', 'dill'],  
 )
