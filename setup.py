import os 
import sys 

#Make sure python 2.7
if not (sys.version_info[0] == 2 and sys.version_info[1] == 7):
    sys.exit("Sorry, only Python 2.7 is currently supported currently.")

try:
	from setuptools import setup 
except: 
	from disutils.core import setup 


#write verion to file so installation is not required to 
#import library prior to installation 
MAJOR = 0
MINOR = 1 
MICRO = 0 
VERSION = '{0}.{1}.{2}'.format(MAJOR,MINOR,MICRO)
NAME = 'quantum_optimal_control'
URL = 'https://github.com/SchusterLab/quantum-optimal-control'
AUTHOR = '''Nelson Leung, Mohamed Abdelhafez,
			Jens Koch and David Schuster'''
AUTHOR_EMAIL = 'nelsonleuon@uchicago.edu'
MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL

KEYWORDS = ['quantum','GRAPE','optimal','control','tensorflow','gpu','qubit']

DESCRIPTION = 'Tensorflow implementation of GRAPE, a quantum optimal control algorithm.'

			  
REQUIRES = [
			'numpy (>=1.8)',
			'scipy (>=0.15)',
			'tensorflow (>=1.0)',
			'qutip (>=4.0)',
			'matplotlib (>=2.0)',
			'h5py (>=2.5)',
			'IPython (>=4.0)'
			]
INSTALL_REQUIRES = [
			'numpy>=1.8',
			'scipy>=0.15',
			'tensorflow>=1.0',
			'qutip>=4.0',
			'matplotlib>=2.0',
			'h5py>=2.5',
			'IPython>=4.0'
			]

PACKAGES = [
		'quantum_optimal_control',
		'quantum_optimal_control/main_grape',
		'quantum_optimal_control/core',
		'quantum_optimal_control/helper_functions'
		]

#project needs a license 
LICENSE = ''

PLATFORMS=['linux']

CLASSIFIERS = [
	'Development Status :: Beta',
	'Intended Audience :: Science/Research',
	'Natural Language :: English',
    'Operating System :: Linux',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Scientific/Engineering :: Physics',
]
version_file = 'version.py'

def write_version(version_file=version_file):
	#open and overwrite old version file
	with open(version_file,'w+') as v:
	
		v.write("""
			version = {0}

			""".format(VERSION))


write_version()



try: 
	readme = open('README.md','r')
	LONG_DESCRIPTION= readme.read()
except:
	LONG_DESCRIPTION = ''





#perform setup
setup(
	name=NAME,
	version=VERSION,
	url=URL,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	maintainer=MAINTAINER,
	maintainer_email = MAINTAINER_EMAIL,
	packages=PACKAGES,
	keywords=KEYWORDS,
	description=DESCRIPTION,
	platforms=PLATFORMS,
	install_requies=INSTALL_REQUIRES,
	classifiers=CLASSIFIERS
	)