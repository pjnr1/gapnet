class UpDownDetector:
    """
    UpDownDetector

    A class for handling an x-up y-down experiment setup
    Detects and counts reversals and whether the stimulus parameter should be increased or decreased
    """
    last_parameter_change: int
    consecutive_correct: int
    consecutive_wrong: int
    number_of_reversals: int

    def __init__(self, entry: int = 2, correct=1, wrong=0):
        """

        @param entry:
            refers to the "Entry" column in Table 1 of Levitt 1971 (default: 2)
        @param correct:
            response that represents a correct answer
        @param wrong:
            response that represents an incorrect answer
        """
        self.correct = correct
        self.wrong = wrong
        self.entry = entry
        self.response_sequence_dict = {
            1: [1, 1],
            2: [1, 2],
            3: [2, 1],
            4: [1, 3],
            5: [1, 4],
            6: [4, 1],
        }
        self.experiment_log = list()
        self.reset()

    def __len__(self):
        return len(self.experiment_log)

    def reset(self):
        """
        Reset the internal counters, as if a new experiment is started

        Note that this also clears the internal experiment log
        """
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.last_parameter_change = 0
        self.number_of_reversals = 0
        self.experiment_log.clear()

    def _update_experiment_log(self, answer, parameter_change, reversal_detected, parameter_value):
        self.experiment_log.append({'answer': answer,
                                    'parameter_value': parameter_value,
                                    'parameter_change': parameter_change,
                                    'reversal_detected': reversal_detected,
                                    'number_of_reversals': self.number_of_reversals,
                                    'consecutive_correct': self.consecutive_correct,
                                    'consecutive_wrong': self.consecutive_wrong})

    def next(self, answer, parameter_value=None):
        """
        Takes the answer, can be anything, but should be either equal to `correct` or `wrong`,
        provided in `__init__` (default is int(1) for correct and int(0) for wrong).

        @param answer:
            Any type, but must be compatible with `answer == self.correct` and `answer == self.correct`
        @param parameter_value:
            Any type that can be summed and calculated the mean of.
        @return:
            0 : whether the stimulus parameter should be updated up or down (1 or -1), or kept as is (0)
            1 : whether a reversal occurred (True or False)

        """
        if answer == self.correct:
            self.consecutive_correct += 1
        elif answer == self.wrong:
            self.consecutive_wrong += 1
        else:
            raise "Answer was neither correct or wrong. Check your code!"
        parameter_change = 0
        if self.consecutive_correct == self.response_sequence_dict[self.entry][1]:
            parameter_change = -1
        if self.consecutive_wrong == self.response_sequence_dict[self.entry][0]:
            parameter_change = 1

        # A reversal is when the last change to the parameter was down and the next change is up
        reversal_detected = (self.last_parameter_change == -1) and (parameter_change == 1)
        if reversal_detected:
            self.number_of_reversals += 1
        if parameter_change != 0:
            self.consecutive_wrong = 0
            self.consecutive_correct = 0
            self.last_parameter_change = parameter_change

        self._update_experiment_log(answer, parameter_change, reversal_detected, parameter_value)

        return parameter_change, reversal_detected

    def get_from_log(self, parameter):
        """
        Returns the log for the given parameter.

        If log is empty, None is returned

        @param parameter:
            parameter to return from log
        @raise ValueError:
            if parameter is not in log keys
        """
        if len(self.experiment_log) == 0:
            return None
        if hasattr(self.experiment_log[0], parameter):
            raise ValueError('parameter not found')
        return [x[parameter] for x in self.experiment_log]

    def get_mean_parameter_value(self, n_reversals=1):
        """
        @param n_reversals:
            number of reversals to include
        @return:
            the mean parameter value from the last n_reversals
        """
        parameter_values_at_reversal = [x['parameter_value'] for x in reversed(self.experiment_log)][:n_reversals]
        return float(sum(parameter_values_at_reversal)) / float(n_reversals)
