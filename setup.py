from setuptools import setup
from glob import glob

package_name = 'nocode_robot_programming'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='petr',
    maintainer_email='petr.vanc@cvut.cz',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "switcher = nocode_robot_programming.state_decision.state_decider_node:main",
        ],
    },
)
