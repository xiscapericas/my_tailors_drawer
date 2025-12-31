from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='radcure-processor',
    version='0.1.0',
    author='Xisca Pericàs',
    description='A Python package for processing RADCURE DICOM cases with TotalSegmentator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'boto3>=1.26.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'SimpleITK>=2.2.0',
        'pydicom>=2.3.0',
        'totalsegmentator>=1.5.0',
        'rt-utils>=1.2.0',
        'nibabel>=3.2.0',
        'matplotlib>=3.5.0',
        'scikit-image>=0.19.0',
        'scipy>=1.7.0',
        'opencv-python>=4.5.0',
        'python-dotenv>=0.19.0',
        'p-tqdm>=1.4.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

