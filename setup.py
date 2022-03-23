from setuptools import setup, find_packages
import os

pkgs = find_packages('src')

def long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme
print(pkgs)

install_requires = [
    "tensorflow>=2.3.0",
    "tokenizers>=0.11.4"
    ]
print(install_requires)
setup(
    name = "deeppostagger",
    version = "0.16",
    url = "https://github.com/sunkusun9/DeepPosTagger",
    description="Sequence-To-Sequence 모델을 활용한 한국어 형태소 분석기",
    long_description = long_description(),
    long_description_content_type='text/markdown',
    author="sun9sun9",
    author_email="sun9sun9@gmail.com",
    packages = pkgs,
    install_requires=install_requires,
    setup_requires=install_requires,
    package_dir = {
        '': 'src',
    },
    package_data = {
        '': ['res/sejong_nikl/model.p', 'res/sejong_nikl/config.json', 'res/sejong_nikl/tokenizer.json']
    }, 
    exclude_package_data={"": ["README.md"]}
)
