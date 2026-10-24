from setuptools import setup, find_packages

package_name = 'sim2real_rl'

setup(
    name='sim2real-robot-rl',
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml'],
    zip_safe=True,
    maintainer='Youseef Abdel Tawwab',
    maintainer_email='youseefahmed794@gmail.com',
    description='Sim-to-Real Reinforcement Learning for Dynamic Quadruped Locomotion',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_inference_node = sim2real_rl.deployment.ros2_inference_node:main',
        ],
    },
)
