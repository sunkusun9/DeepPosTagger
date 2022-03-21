from setuptools import setup, find_packages
import os

pkgs = find_packages('src')

def long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme
print(pkgs)

install_requires = []
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        install_requires = f.read().splitlines()

setup(
    name = "deeppostagger",
    version = "0.1",
    long_description = long_description(),
    author="sun9sun9",
    author_email="sun9sun9@gmail.com",
    packages = pkgs,
    install_requires=install_requires,
    package_dir = {
        '': 'src',
    },
    package_data = {
        '': ['res/sejong_nikl/model.p', 'res/sejong_nikl/config.json', 'res/sejong_nikl/tokenizer.json']
    }
    , exclude_package_data={"": ["README.md"]}
)