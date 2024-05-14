from setuptools import setup, find_packages

print(find_packages(exclude=["tests*"]))
setup(
    name="discovery",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
)
