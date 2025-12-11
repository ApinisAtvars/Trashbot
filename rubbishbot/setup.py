from setuptools import setup

package_name = 'rubbishbot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agilex',
    maintainer_email='agilex@todo.todo',
    description='Rubbishbot navigation package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_manager = rubbishbot.mission_manager:main',
            'pose_setter = rubbishbot.pose_setter:main',
            'detector = rubbishbot.object_detector:main',
        ],
    },
)