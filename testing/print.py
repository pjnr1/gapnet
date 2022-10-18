import datetime


def teardown(exc_type, exc_value, tb):
    if exc_type is not None:
        tb.print_exception(exc_type, exc_value, tb)
        return False  # uncomment to pass exception through

    return True


class PrintContext:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        print(self.message, end='...')

    def __exit__(self, exc_type, exc_value, tb):
        print('done')
        return teardown(exc_type, exc_value, tb)


class TimeAndPrintContext(PrintContext):
    def __init__(self, message):
        super().__init__(message)
        self.time_of_enter = datetime.datetime.now()

    def __enter__(self):
        super(TimeAndPrintContext, self).__enter__()
        self.time_of_enter = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, tb):
        elapsed = datetime.datetime.now() - self.time_of_enter
        print(f'done. Elapsed time: {elapsed}')
        return teardown(exc_type, exc_value, tb)
