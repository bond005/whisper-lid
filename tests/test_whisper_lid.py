import sys
import os
import unittest

import numpy as np
from scipy.io.wavfile import read

try:
    from whisper_lid.whisper_lid import calculate_sound_energy, TARGET_SAMPLING_RATE, CHUNK_LENGTH
    from whisper_lid.whisper_lid import SPECTROGRAM_HOP_LENGTH, SPECTROGRAM_WIN_LENGTH
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from whisper_lid.whisper_lid import calculate_sound_energy, TARGET_SAMPLING_RATE, CHUNK_LENGTH
    from whisper_lid.whisper_lid import SPECTROGRAM_HOP_LENGTH, SPECTROGRAM_WIN_LENGTH


class TestWhisperLID(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_sound_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'test_sound_ru.wav')
        rate, data = read(test_sound_fname)
        if rate != TARGET_SAMPLING_RATE:
            err_msg = f'{test_sound_fname}: the sampling rate is wrong! Expected {TARGET_SAMPLING_RATE}, got {rate}.'
            raise IOError(err_msg)
        if len(data.shape) != 1:
            err_msg = f'{test_sound_fname}: the channels number is wrong! Expected 1, got {len(data.shape)}.'
            raise IOError(err_msg)
        if str(data.dtype) not in {'int16', 'uint8'}:
            err_msg = f'{test_sound_fname}: only 8 bits or 16 bits per sample are supported! Got {data.dtype}'
            raise IOError(err_msg)
        if str(data.dtype) == 'int16':
            cls.test_sound_with_speech_ = data.astype(np.float32) / 32768.0
        else:
            cls.test_sound_with_speech_ = (data.astype(np.float32) - 128.0) / 128.0
        cls.test_zero_sound_ = np.zeros(((TARGET_SAMPLING_RATE * CHUNK_LENGTH) // 5,), dtype=np.float32)

    def test_calculate_sound_energy_01(self):
        energy = calculate_sound_energy(self.test_sound_with_speech_)
        self.assertIsInstance(energy, np.ndarray)
        self.assertEqual(len(energy.shape), 1)
        self.assertLess(energy.shape[0], self.test_sound_with_speech_.shape[0])
        ref_energy_len = 1 + self.test_sound_with_speech_.shape[0] // SPECTROGRAM_HOP_LENGTH
        err_msg = (f'Sound length = {self.test_sound_with_speech_.shape[0]}: expected energy length = {ref_energy_len},'
                   f' got one = {energy.shape[0]}')
        self.assertEqual(energy.shape[0], ref_energy_len, msg=err_msg)
        self.assertGreaterEqual(np.min(energy), 0.0)
        self.assertGreater(np.max(energy), 0.0)
        self.assertGreater(np.max(energy), np.min(energy))

    def test_calculate_sound_energy_02(self):
        energy = calculate_sound_energy(self.test_zero_sound_)
        self.assertIsInstance(energy, np.ndarray)
        self.assertEqual(len(energy.shape), 1)
        self.assertLess(energy.shape[0], self.test_zero_sound_.shape[0])
        ref_energy_len = 1 + self.test_zero_sound_.shape[0] // SPECTROGRAM_HOP_LENGTH
        err_msg = (f'Sound length = {self.test_zero_sound_.shape[0]}: expected energy length = {ref_energy_len},'
                   f' got one = {energy.shape[0]}')
        self.assertEqual(energy.shape[0], ref_energy_len, msg=err_msg)
        self.assertEqual(energy.shape[0], ref_energy_len)
        self.assertAlmostEqual(np.min(energy), 0.0)
        self.assertAlmostEqual(np.max(energy), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
