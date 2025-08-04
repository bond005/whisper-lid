from argparse import ArgumentParser
import gc
import os
import random
import sys

from datasets import load_dataset, DownloadConfig
from datasets.features import Audio
from librosa import load
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

try:
    from whisper_lid.whisper_lid import detect_language_in_speech, detect_language_in_long_speech, TARGET_SAMPLING_RATE
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from whisper_lid.whisper_lid import detect_language_in_speech, detect_language_in_long_speech, TARGET_SAMPLING_RATE


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--sound', dest='sound_name', type=str, required=False,
                        default=None, help='The sound file for language identification demo.')
    parser.add_argument('-r', '--reference', dest='reference_language', type=str, required=False,
                        default=None, help='The reference language for demo.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=False,
                        default='bond005/whisper-podlodka-turbo', help='The Whisper model name.')
    parser.add_argument('--long', dest='long_sound', action='store_true',
                        help='Is the sound file a very long?')
    args = parser.parse_args()

    if args.sound_name is None:
        librispeech_languages = ['dutch', 'french', 'german', 'italian', 'polish', 'portuguese', 'spanish']
        if args.reference_language is None:
            test_language = random.choice(librispeech_languages)
        else:
            if args.reference_language.lower() not in librispeech_languages:
                err_msg = f'The language "{args.reference_language}" is not supported in the Multilingual Librispeech!'
                raise ValueError(err_msg)
            test_language = args.reference_language.lower()
        download_config = DownloadConfig(
            max_retries=500,
            num_proc=max(1, os.cpu_count())
        )
        ds = load_dataset('facebook/multilingual_librispeech', test_language, split='test',
                          trust_remote_code=True, download_config=download_config)
        if ds.features['audio'].sampling_rate != TARGET_SAMPLING_RATE:
            ds = ds.cast_column('audio', Audio(sampling_rate=TARGET_SAMPLING_RATE))
        test_sound = random.choice(ds['audio'])['array']
        del ds
        gc.collect()
    else:
        if args.reference_language is None:
            test_language = 'not specified'
        else:
            test_language = args.reference_language.lower()
        sound_fname = os.path.normpath(args.sound_name)
        if not os.path.isfile(sound_fname):
            raise IOError(f'The file "{sound_fname}" does not exist!')
        test_sound, sample_rate = load(sound_fname, sr=TARGET_SAMPLING_RATE, mono=True, dtype=np.float32)
        if sample_rate != TARGET_SAMPLING_RATE:
            err_msg = (f'{sound_fname}: the sound has a wrong sample rate! '
                       f'Expected {TARGET_SAMPLING_RATE}, got {sample_rate}.')
            raise IOError(err_msg)

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    if torch.cuda.is_available():
        model = model.to('cuda:0')
        print('CUDA is available!')
    model.eval()
    if args.long_sound:
        detected, num_chunks = detect_language_in_long_speech(
            test_sound,
            processor.feature_extractor, processor.tokenizer, model,
            True
        )
    else:
        detected = detect_language_in_speech(
            test_sound,
            processor.feature_extractor, processor.tokenizer, model
        )
        num_chunks = 1
    print(f'\nReference language: {test_language}\n')
    if num_chunks > 1:
        print(f'There are {num_chunks} chunks in the analyzed sound.')
    print('Detected:')
    max_text_width = max([len(it[0]) for it in detected])
    for lang, proba in detected:
        print('  {0:>{1}} {2:.4f}'.format(lang, max_text_width, proba))


if __name__ == '__main__':
    main()
