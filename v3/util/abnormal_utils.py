import numpy as np


def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    return (
        1.0
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    )


def filt(input_scores, dim=9, range=302, mu=21):
    filter_2d = gaussian_filter(np.arange(1, range), mu)
    frame_scores = np.asarray(input_scores, dtype=np.float64)
    padding_size = len(filter_2d) // 2
    padded = np.concatenate(
        (np.zeros(padding_size), frame_scores, np.zeros(padding_size))
    )
    return np.correlate(padded, filter_2d, "valid")
