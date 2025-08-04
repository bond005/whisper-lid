[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/whisper-lid/blob/master/LICENSE)
![Python 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-green.svg)

# Whisper-LID
The **Whisper-LID** is a spoken language identification system, which is based on the [Whisper model](https://huggingface.co/docs/transformers/model_doc/whisper). The system uses the Whisper-based algorithm to identify spoken languages or non-speech event. The Section 2.3 of the [paper about Whisper](https://arxiv.org/abs/2212.04356) states that language tags or non-speech tags need to be predicted after the `<|startoftranscript|>` special token. Based on this information, the system estimates a probability distribution for the next token after the `<|startoftranscript|>` and selects the token with the highest probability as the final spoken language prediction. Since the predicted token can be either a language tag or a non-speech tag, the system combines the features of a spoken language identifier and a voice activity detector.

Installing
----------


For installation, you need to Python 3.10 or later. You can install the **Whisper-LID** from the [PyPi](https://pypi.org/project/whisper-lid) using the following command:

```
python -m pip install whisper-lid
```

If you want to install the **Whisper-LID** in a Python virtual environment, you don't need to use `sudo`, but before installing, you will need to activate this virtual environment. For example, you can do this by using `conda activate your_favourite_environment` in the Linux terminal, or in the Anaconda Prompt for Microsoft Windows).

Also, 

To build this project from sources, you should run the following commands in the Terminal:

```
git clone https://github.com/bond005/whisper-lid.git
cd whisper-lid
python -m pip install .
```

In this case, you can run the unit tests to check workability and environment setting correctness:

```
python setup.py test
```

or

```
python -m unittest
```

Usage
-----

After installing the **Whisper-LID**, you can use it as a Python package in your projects. For example, you can use  the `bond005/whisper-podlodka-turbo` and apply it to some audio file with Russian speech as follows:

```python
from librosa import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_lid.whisper_lid import detect_language_in_speech, TARGET_SAMPLING_RATE

sound_fname = '/path/to/your/wav/audio/in/Russian.wav'  # for example, this is a path to your audiofile with Russian speech
waveform, _ = load(sound_fname, sr=TARGET_SAMPLING_RATE, mono=True)

model_id = 'bond005/whisper-podlodka-turbo'  # it can be any Whisper model
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

identified = detect_language_in_speech(
    waveform,
    processor.feature_extractor, processor.tokenizer, model
)
for language, probability in identified:
    print('P({0}) = {1:.4f}'.format(language, probability))
```

As a result of the execution, you will see something like this:

```text
P(russian) = 0.9823
P(english) = 0.0092
P(portuguese) = 0.0013
P(castilian) = 0.0011
P(spanish) = 0.0011
P(french) = 0.0009
P(german) = 0.0005

... ... ...
P(faroese) = 0.0000
P(lao) = 0.0000
P(cantonese) = 0.0000
P(NO SPEECH) = 0.0000
```

As you see, all languages, including the non-speech event tag, are sorted in descending order of probability.

If you need to process an audio signal that is longer than 30 seconds, you can use the `detect_language_in_long_speech` function instead of `detect_language_in_speech`. This function splits the signal into chunks of 30 seconds with a 10-second overlapping and accumulates the total identification results for all chunks. For example, it might look like this:

```python
from librosa import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_lid.whisper_lid import detect_language_in_long_speech, TARGET_SAMPLING_RATE

sound_fname = '/path/to/your/very/long/audio.wav'
waveform, _ = load(sound_fname, sr=TARGET_SAMPLING_RATE, mono=True)

model_id = 'bond005/whisper-podlodka-turbo'  # it can be any Whisper model
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

identified, num_chunks = detect_language_in_long_speech(
    waveform,
    processor.feature_extractor, processor.tokenizer, model
)
print('United result for {0} chunks:'.format(num_chunks))
for language, probability in identified:
    print('P({0}) = {1:.4f}'.format(language, probability))
```

Demo
----

In the `demo` directory, you can find the **lid_demo.py** script. This script shows how to use the `detect_language_in_speech` and `detect_language_in_long_text` functions. To run the script, simply call it from the command line like this:

```shell
python lid_demo.py \
    -m /path/to/your/whisper \
    -s /path/to/wav/pcm/sound.wav \
    -r "reference language" \
    --long
```

The argument **-m** is a name of the used Whisper model (for example, `openai/whisper-large-v3-turbo` or `bond005/whisper-podlodka-turbo`). This can be the name of the model in the HuggingFace Hub or the path to a folder containing the model on a user's local device.

The argument **-s** is an input WAV PCM sound file.

The argument **-r** is a reference language (for comparison with the system prediction).

The optional argument **--long** indicates the need to call the `detect_language_in_long_text` function instead of `detect_language_in_speech` to process a long (more than 30 seconds) audio file.

If the input sound file is not specified, then the random sound from [Multilingual Librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) is used. In this case, the reference language can only be one of those languages that are supported in the Multilingual Librispeech dataset.

License
-------

The **Whisper-LID** (`whisper-lid`) is Apache 2.0 - licensed.