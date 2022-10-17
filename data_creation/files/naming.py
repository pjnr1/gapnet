"""
Helper functions for naming files.

Most functions are fairly simple, but made to keep a unified and easy-to-handle way of setting filenames.
Which should reduce number of errors by typos etc.

"""
import os
from typing import List, Union


def embed_gap_in_filename(filename: str, gap_position: float, gap_duration: float) -> str:
    """
    Embed gap information in filename (position and duration

    @param filename:
    @param gap_duration:
    @param gap_position:
    @return:
    """
    return embed_in_filename(filename,
                             ['{0:.2f}ms'.format(gap_duration * 1e3),
                              '{0:.2f}ms'.format(gap_position * 1e3)])


def embed_nogap_in_filename(filename: str) -> str:
    """
    embed 'nogap' in filename

    @param filename:
    @return:
    """
    return embed_in_filename(filename, 'nogap')


def embed_in_filename(filename: str, embeddings: Union[List[str], str]) -> str:
    """
    General function for embedding strings in the filename

    @param filename:
        original filename
    @param embeddings:
        string or list of strings to embed in the filename
    @return:
    """
    fn, ext = os.path.splitext(filename)
    if isinstance(embeddings, str):
        embeddings = [embeddings]
    return '_'.join([fn, *embeddings]) + ext


def get_next_non_existent(filepath: str) -> str:
    """
    Generate a new filename if path exists, e.g.:

    When 'test.txt' exists, the function will check 'test-{n=1}.txt' and increment `n`
    until the filename doesn't exist.

    @param filepath:
    @return:
    """
    fn_new = filepath
    root, ext = os.path.splitext(filepath)
    i = 0
    while os.path.exists(fn_new):
        i += 1
        fn_new = '{}-{}{}'.format(root, str(i), ext)
    return fn_new


def get_condition_folders(mode, bin_width):
    """

    @param mode:
        specify whether to use summed ANF or individual channels (defined
    @param bin_width:
        specify width of the bins used to down-sample
    @return:
    """
    return ['mode_' + mode, 'bw_' + str(bin_width)]