from setuptools import setup, find_packages

setup(
    name="HiResCAM",  # Change this to your package name
    version="1.0.0",  # Versioning for updates
    author="D. Swainson Sujana and  D Peter Augustine",
    author_email="d.sujana@res.christuniversity.in",
    description="High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using Structural MRI Images",
    long_description=open("README.md").read(),  # Reads the README file for package details
    long_description_content_type="text/markdown",
    url="https://swainsonsujana.github.io/XAI_HiResCAM",  # Update with your repo URL
    packages=find_packages(),  # Finds all Python packages in the directory
    install_requires=[
        "tensorflow>=2.16.1",  # Ensure TensorFlow version compatibility
        "keras>=3.3.3",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "nibabel>=5.2.1",  # If working with neuroimaging data
        "opencv-python",  # If using image processing
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.14",  # Ensures Python version compatibility
)
