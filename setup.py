from setuptools import setup, find_packages

setup(
    name="deepfake-detection",
    version="1.0.0",
    author="Farrukh",
    description="Deepfake Image Detection using EfficientNet-B4 Transfer Learning",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "grad-cam>=1.4.8",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
    ],
)
