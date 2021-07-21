__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-yolov5-segmenter',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executor that segments images into bounding boxes of detected objects',
    url='https://github.com/jina-ai/executor-yolov5',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.segmenter.yolov5_segmenter'],
    package_dir={'jinahub.segmenter': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
