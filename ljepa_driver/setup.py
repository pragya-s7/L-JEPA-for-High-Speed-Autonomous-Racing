from setuptools import setup

package_name = 'ljepa_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pragya',
    description='L-JEPA policy driver node for F1TENTH',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ljepa_node = ljepa_driver.ljepa_node:main',
        ],
    },
)
