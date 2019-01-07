from setuptools import setup

setup(name='phynix_gym',
      url="https://github.com/Phynix/phynix_gym",
      author="Jonas Eschle 'Mayou36'",
      author_email="jonas.eschle@cern.ch",
      version='0.0.3',
      install_requires=['gym', 'numpy', 'tensorflow'],
      classifiers=[
          "Development Status :: 2 - Pre-Alpha",
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          "License :: OSI Approved :: Apache Software License",
          'Programming Language :: Python :: 3.6',
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
          ],
      python_requires=">=3",
      include_package_data=True,
      license='Apache License 2.0',

      )
