from setuptools import find_packages, setup

setup(
    name="Smart-cooling-edge_AI",
    version="0.0.1",
    author="Pasindu Gunathilake",
    author_email="pasindugunathilake144@gmail.com",
    description="An edge computer vision system for smart AC energy saving",
    packages=find_packages(),
    install_requires=[
        "flask==3.0.2",
        "flask-cors==4.0.0",
        "opencv-python==4.9.0.80",
        "numpy==1.26.4",
        "onnxruntime==1.17.1",
        "requests==2.31.0"
    ]
)

# pip install -e .