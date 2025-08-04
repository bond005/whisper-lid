from setuptools import setup, find_packages

import whisper_lid


long_description = '''
Whisper-LID
===============

This is a spoken language identification system that is based on the Whisper
model. The system uses the Whisper-based algorithm to identify spoken languages
or non-speech event. The Section 2.3 of the paper about Whisper
(https://arxiv.org/abs/2212.04356) states that language tags or non-speech
tags need to be predicted after the `<|startoftranscript|>` special token.
Based on this information, the system estimates a probability distribution
for the next token after the `<|startoftranscript|>` and selects the token
with the highest probability as the final spoken language prediction. Since
the predicted token can be either a language tag or a non-speech tag, the
system combines the features of a spoken language identifier and a voice
activity detector.
'''

setup(
    name='whisper-lid',
    version=whisper_lid.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Spoken Language IDentification (LID) using multilingual Whisper model',
    long_description=long_description,
    url='https://github.com/bond005/whisper-lid',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords=['whisper', 'LID', 'spoken-language', 'language-identification', 'spoken-language-identification'],
    install_requires=['librosa>=0.10.0', 'numpy', 'scipy', 'sentencepiece', 'soundfile>=0.11.0',
                      'torch>=2.0.1', 'torchaudio>=2.0.1', 'transformers>=4.38.1', 'datasets<4.0'],
    test_suite='tests'
)
