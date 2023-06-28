from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
        name="gapnet",
        version="0.1.0",
        description="gapnet",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/pjnr1/gapnet",
        author="Jens C Thuren Lindahl",
        author_email="jensctl@gmail.com",
        classifiers=[],
        keywords="",
        package_dir={"", "."},
        packages=find_packages(where="."),
        python_requires=">=3.10, <4",
        install_requires=[],
        extra_require={
            "dev": [],
            "test": [],
        },
        package_data={},
        entry_points={},
        project_urls={
            "documentation": "https://pjnr1.github.io/gapnet/api/index.html",
        },
)
