import os


def get_next_non_existent(filepath: str) -> str:
    """
    Generate a new filename if path exists, e.g.:

    When 'test.txt' exists, the function will check 'test-{n=1}.txt' and increment `n`
    until the filename doesn't exist.
    """
    fn_new = filepath
    root, ext = os.path.splitext(filepath)
    i = 0
    while os.path.exists(fn_new):
        i += 1
        fn_new = '{}-{}{}'.format(root, str(i), ext)
    return fn_new
