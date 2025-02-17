from setuptools import setup, find_packages

setup(
    name="HiResCAM", 
    version="1.0.0", 
    author="D. Swainson Sujana and  D Peter Augustine",
    author_email="d.sujana@res.christuniversity.in",
    description="High-Resolution XAI Explanations for Deep Learning-Based Autism Diagnosis Using Structural MRI Images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://swainsonsujana.github.io/XAI_HiResCAM", 
    packages=find_packages(),  
    install_requires=[
        "tensorflow>=2.16.1",  
        "keras>=3.3.3",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "nibabel>=5.2.1",  
        "opencv-python", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.14",  
)
