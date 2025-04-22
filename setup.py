"""
Setup script for the Student Performance Predictor package.
"""

from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    Get package requirements from file.
    
    Args:
        file_path: Path to the requirements file
        
    Returns:
        List of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
        
    return requirements


setup(
    name="student_performance_predictor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning application that predicts student math scores",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/student_performance_predictor",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education :: Testing",
    ],
    entry_points={
        "console_scripts": [
            "student-predictor=main:main",
        ],
    },
)