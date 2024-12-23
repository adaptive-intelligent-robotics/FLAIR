from setuptools import find_packages, setup

package_name = "flair"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maxime Allard",
    maintainer_email="maxime.allard@imperial.ac.uk",
    description="FLAIR",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "run_adaptation = flair.adaptation:main",
            "run_adaptation_rl = flair.adaptation_rl:main",
            "run_adaptation_lqr = flair.adaptation_lqr:main",
            "run_bridge = flair.bridge:main",
            "run_vicon = flair.vicon:main",
        ],
    },
)
