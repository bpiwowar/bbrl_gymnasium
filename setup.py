from setuptools import find_packages, setup

setup(
    name="bbrl_gymnasium",
    packages=[package for package in find_packages() if package.startswith("my_gym") or package.startswith("bbrl_gym")],
    url="https://github.com/osigaud/my_gym",
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    license="MIT",
    description="A set of additional gym environments",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    install_requires=open("requirements.txt", "r").read().splitlines(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
)
