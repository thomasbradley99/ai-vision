import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="player-counting-sam2",
    version='0.1.0',
    python_requires=">=3.8",
    description="Player counting and tracking using SAM2 and embedding vectors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/clannai/player-counting-sam2",
    author="ClannAI",
    author_email="",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "supervision>=0.16.0",
        "ultralytics>=8.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'jupyter>=1.0.0',
        ],
        'sam2': [
            'segment-anything-2 @ git+https://github.com/facebookresearch/segment-anything-2.git',
        ]
    },
    entry_points={
        'console_scripts': [
            'player-count=src.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Computer Vision',
        'Operating System :: OS Independent',
    ]
)

