from setuptools import setup, find_packages

setup(
    name="adv_camouflage",
    version="1.0.0",
    description="Physically Grounded Binary Adversarial Camouflage for Aerial Object Detectors",
    author="leeIITM",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "numpy>=1.23.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
