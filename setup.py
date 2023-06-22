from setuptools import setup, find_packages, Extension
module = Extension('mykmeanssp', sources=['kmeans.c'])
setup(name='mykmeanssp',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])
# setup(
#     name='mykmeanssp',
#     version='0.1.0',
#     author='Nadav & Saar',
#     author_email='saargerassi@mail.tau.ac.il',
#     description='kmeans pp project hw 2',
#     install_requires=['invoke'],
#     packages=find_packages(),
#     license='GPL-2',
#     ext_modules=[
#         Extension(
#             'mykmeanssp',
#             ['kmeans.c']
#         )
#     ]
# )