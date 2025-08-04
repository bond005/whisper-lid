from typing import List, Tuple

import numpy as np
from scipy.special import softmax
import torch
from torchaudio.transforms import Spectrogram
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from tqdm import trange


TARGET_SAMPLING_RATE: int = 16_000
CHUNK_LENGTH: int = 30
HOP_LENGTH: int = 20
SPECTROGRAM_HOP_LENGTH: int = 160
SPECTROGRAM_WIN_LENGTH: int = 400
AMPLITUDE_EPSILON: float = 1e-4


def calculate_sound_energy(sound: np.ndarray) -> np.ndarray:
    if len(sound.shape) != 1:
        err_msg = f'The sound dimension is wrong! Expected 1-D, got {len(sound.shape)}-D.'
        raise RuntimeError(err_msg)
    if (np.max(sound) - np.min(sound)) <= AMPLITUDE_EPSILON:
        energy = np.zeros((1 + sound.shape[0] // SPECTROGRAM_HOP_LENGTH,), dtype=np.float32)
    else:
        fe = Spectrogram(n_fft=512, win_length=SPECTROGRAM_WIN_LENGTH, hop_length=SPECTROGRAM_HOP_LENGTH, center=True)
        with torch.no_grad():
            spectrogram = fe(torch.tensor(sound, dtype=torch.float32, device='cpu')).numpy()
        del fe
        energy = np.zeros((spectrogram.shape[1],), dtype=np.float32)
        for time_idx in range(spectrogram.shape[1]):
            energy[time_idx] = np.sum(spectrogram[:, time_idx])
        del spectrogram
    return energy


def find_best_chunk(sound: np.ndarray) -> np.ndarray:
    if len(sound.shape) != 1:
        err_msg = f'The sound dimension is wrong! Expected 1-D, got {len(sound.shape)}-D.'
        raise RuntimeError(err_msg)
    max_sound_len = TARGET_SAMPLING_RATE * CHUNK_LENGTH
    if sound.shape[0] < max_sound_len:
        sound_ = sound
    else:
        if (np.max(sound) - np.min(sound)) <= AMPLITUDE_EPSILON:
            sound_ = sound[0:max_sound_len]
        else:
            energy = calculate_sound_energy(sound)
            max_energy_idx = int(np.argmax(energy))
            chunk_start = max_energy_idx * SPECTROGRAM_HOP_LENGTH - max_sound_len // 2
            if chunk_start < 0:
                chunk_start = 0
            chunk_end = chunk_start + max_sound_len
            if chunk_end > sound.shape[0]:
                chunk_start -= (chunk_end - sound.shape[0])
                chunk_end = sound.shape[0]
            sound_ = sound[chunk_start:chunk_end]
    return sound_


def detect_language_in_speech(all_sound: np.ndarray,
                              fe: WhisperFeatureExtractor, tok: WhisperTokenizer,
                              asr: WhisperForConditionalGeneration) -> List[Tuple[str, float]]:
    if (np.max(all_sound) - np.min(all_sound)) <= AMPLITUDE_EPSILON:
        return [('NO SPEECH', 1.0)] + sorted([(k, 0.0) for k in TO_LANGUAGE_CODE], key=lambda it: it[0])
    sound_chunk = find_best_chunk(all_sound)
    inputs = fe(
        sound_chunk.reshape((1, sound_chunk.shape[0])),
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors='pt',
        truncation=False,
        padding='longest',
        return_attention_mask=True,
    )
    del sound_chunk
    input_features = inputs.input_features
    attention_mask = inputs.attention_mask
    if input_features.shape[-1] < 3000:
        padded_features = torch.zeros(
            1, input_features.shape[1], 3000,
            device=input_features.device, dtype=input_features.dtype
        )
        padded_attention_mask = torch.zeros(
            1, 3000,
            device=attention_mask.device, dtype=attention_mask.dtype
        )
        padded_features[:, :, :input_features.shape[2]] = input_features
        padded_attention_mask[:, :attention_mask.shape[1]] = attention_mask
    else:
        padded_features = input_features
        padded_attention_mask = attention_mask
    del input_features, attention_mask
    decoder_token_ids = tok.encode(
        '<|startoftranscript|>',
        add_special_tokens=False
    )
    decoder_input_ids = torch.tensor(
        [decoder_token_ids],
        device=asr.device,
        dtype=torch.long
    )
    with torch.no_grad():
        logits = asr(
            input_features=padded_features.to(asr.device, dtype=asr.dtype),
            attention_mask=padded_attention_mask.to(asr.device),
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        ).logits.cpu().numpy()
    languages = []
    language_and_nospeech_ids = []
    for k in TO_LANGUAGE_CODE:
        languages.append(k)
        language_and_nospeech_ids += tok.encode('<|' + TO_LANGUAGE_CODE[k] + '|>', add_special_tokens=False)
    languages.append('NO SPEECH')
    language_and_nospeech_ids += tok.encode('<|nospeech|>', add_special_tokens=False)
    probabilities = softmax(logits[0, 0, language_and_nospeech_ids].astype(np.float64))
    del logits
    results = sorted(
        [(languages[idx], float(probabilities[idx])) for idx in range(len(language_and_nospeech_ids))],
        key=lambda it: (-it[1], it[0])
    )
    del probabilities, language_and_nospeech_ids, languages
    return results


def detect_language_in_long_speech(all_sound: np.ndarray,
                                   fe: WhisperFeatureExtractor, tok: WhisperTokenizer,
                                   asr: WhisperForConditionalGeneration,
                                   verbose: bool = False) -> Tuple[List[Tuple[str, float]], int]:
    if len(all_sound.shape) != 1:
        err_msg = f'The sound dimension is wrong! Expected 1-D, got {len(all_sound.shape)}-D.'
        raise RuntimeError(err_msg)
    if all_sound.shape[0] <= (TARGET_SAMPLING_RATE * CHUNK_LENGTH):
        n_chunks = 1
    else:
        n_chunks = 1 + (all_sound.shape[0] - CHUNK_LENGTH * TARGET_SAMPLING_RATE) // (HOP_LENGTH * TARGET_SAMPLING_RATE)
    if (np.max(all_sound) - np.min(all_sound)) <= AMPLITUDE_EPSILON:
        results = [('NO SPEECH', 1.0)] + sorted([(k, 0.0) for k in TO_LANGUAGE_CODE], key=lambda it: it[0])
    else:
        if n_chunks < 2:
            results = detect_language_in_speech(all_sound, fe, tok, asr)
        else:
            results = []
            for chunk_idx in (trange(n_chunks) if verbose else range(n_chunks)):
                chunk_start = chunk_idx * HOP_LENGTH * TARGET_SAMPLING_RATE
                chunk_end = min(chunk_start + TARGET_SAMPLING_RATE * CHUNK_LENGTH, all_sound.shape[0])
                results_ = detect_language_in_speech(all_sound[chunk_start:chunk_end], fe, tok, asr)
                if len(results) == 0:
                    del results
                    results = results_
                else:
                    for lang, proba in results_:
                        found_idx = -1
                        for idx in range(len(results)):
                            if results[idx][0] == lang:
                                found_idx = idx
                                break
                        if found_idx < 0:
                            raise RuntimeError(f'The language "{lang}" is not found!')
                        results[found_idx] = (lang, results[found_idx][1] + proba)
                del results_
            sum_proba = 0.0
            for _, proba in results:
                sum_proba += proba
            for idx in range(len(results)):
                results[idx] = (results[idx][0], results[idx][1] / sum_proba)
            results.sort(key=lambda it: (-it[1], it[0]))
    return results, n_chunks
