import os
import signal
import sys
import traceback

from booster.utils import logging_sep


class Success():
    """handles the `success.txt` file generated for each run."""
    file = 'success.txt'
    success = f"Success."
    aborted_by_user = f"Aborted by User."
    failure_base = "Failed."
    sigterm = "SIGTERM."

    def failure(exception):
        return f"{Success.failure_base} Exception : \n{exception}\n\n{traceback.format_exc()}"

    def __init__(self, logdir=None):
        signal.signal(signal.SIGTERM, lambda *args: self.__exit__(*Success.sigterm_handler(*args)))
        self.logdir = logdir
        if logdir is not None:
            if not os.path.exists(logdir):
                os.makedirs(logdir)

    def __enter__(self):
        pass

    @staticmethod
    def sigterm_handler(_signo, _stack_frame):
        return (Success.sigterm, Success.sigterm, _stack_frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # handle success message
        message = {
            None: Success.success,
            KeyboardInterrupt: Success.aborted_by_user,
            Success.sigterm: Success.aborted_by_user,
        }.get(exc_type, None)

        if message is not None:
            # if message is handle
            print(f"{logging_sep('=')}\n@ {sys.argv[0]} : {message}\n{logging_sep('=')}")
            if self.logdir is not None:
                with open(os.path.join(self.logdir, Success.file), 'w') as f:
                    f.write(message)
        else:
            # if exception is unknown
            print(
                f"{logging_sep('=')}\n@ {sys.argv[0]}: Failed with exception {exc_type} = `{exc_val}` \n{logging_sep('=')}")
            traceback.print_exception(exc_type, exc_val, exc_tb)
            with open(os.path.join(self.logdir, Success.file), 'w') as f:
                f.write(Success.failure(exc_val))

        if exc_type == Success.sigterm:
            exit(0)