from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='radcure-processor',
    version='0.1.0',
    author='Your Name',
    description='A Python package for processing RADCURE DICOM cases with TotalSegmentator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'boto3',
        'numpy',
        'pandas',
        'SimpleITK',
        'pydicom',
        'totalsegmentator',
        'rt-utils',
        'nibabel',
        'matplotlib',
        'scikit-image',
        'scipy',
        'opencv-python',
        'python-dotenv',  # For .env file support
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

