from setuptools import find_packages, setup

setup(
    name="bbrl_gymnasium",
    packages=[
        package
        for package in find_packages()
        if package.startswith("my_gym") or package.startswith("bbrl_gym")
    ],
    url="https://github.com/osigaud/bbrl_gym",
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    install_requires=open("requirements.txt", "r").read().splitlines(),
)
