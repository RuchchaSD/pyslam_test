#!/usr/bin/env python
"""
GlobalPrinter.py

A singleton class that provides a global interface to print messages either to
stdout or to a log file using FileLogger. If the chosen log file already exists,
it appends a numeric suffix (e.g., main_1.log, main_2.log) to avoid collisions.

This version leaves the existing FileLogger class untouched.
We simply handle collisions, close the old logger properly, and catch any
exceptions in creating the new FileLogger.
"""

import sys
import os
import logging
import threading
from typing import Optional

# Import the same FileLogger you shared. No modifications needed there.
import torch.multiprocessing as mp

import utils_sys
from utils_sys import FileLogger
import logging

class SingletonMeta(type):
    """
    A simple thread-safe Singleton metaclass using double-checked locking.
    """
    _instance = None
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

def _resolve_conflicting_logfile(desired_path: str) -> str:
    """
    If desired_path already exists, append _1, _2, etc. until it doesn't.
    Example: main.log -> main_1.log -> main_2.log.
    """
    base, ext = os.path.splitext(desired_path)
    candidate = desired_path
    counter = 0
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base}_{counter}{ext}"
    return candidate


# We'll store Python's original built-in print
_original_print = print

class GlobalPrinter(metaclass=SingletonMeta):
    """
    A singleton printer that writes to stdout or to a FileLogger.

    Usage:
      - GlobalPrinter.get_instance() returns the single instance (creates if needed).
      - Or GlobalPrinter.create(print_to_terminal, logfile_path) sets the instance config.
      - Then call .print(...) or any python 'print(...)' call.
    """

    def __init__(self, print_to_terminal: bool = True, log_file_path: str = "printer_log.txt"):
        """Called only once."""
        self._print_to_terminal = print_to_terminal
        self._file_logger: Optional[FileLogger] = None
        self._set_config(print_to_terminal, log_file_path)

    @classmethod
    def create(cls, print_to_terminal: bool, log_file_path: str):
        """
        If the instance doesn't exist, create it.
        If it exists, reconfigure it with new settings.
        """
        instance = cls()
        instance._set_config(print_to_terminal, log_file_path)
        return instance

    def _set_config(self, print_to_terminal: bool, log_file_path: str):
        """Reconfigure the logger at runtime."""
        self._print_to_terminal = print_to_terminal

        # Close old logger if any
        if self._file_logger is not None:
            self._file_logger.close()
            del self._file_logger

        # Attempt to create a new logger
        safe_path = _resolve_conflicting_logfile(log_file_path)
        self._log_file_path = safe_path
        try:
            self._file_logger = FileLogger(log_file=safe_path, level=logging.INFO)
        except Exception as e:
            # Fallback: If creation fails, revert to terminal prints only
            self._file_logger = None
            self._print_to_terminal = True
            _original_print(f"[GlobalPrinter] Failed to create FileLogger at '{safe_path}': {e}", file=sys.stderr)
            _original_print("[GlobalPrinter] Falling back to terminal printing.", file=sys.stderr)

    @classmethod
    def get_instance(cls):
        """Return the existing GlobalPrinter singleton or create it with default config."""
        return cls()

    def set_print_to_terminal(self, enabled: bool):
        """
        Enables or disables printing to terminal. If disabled, logs go to the file.
        """
        self._print_to_terminal = enabled
        _original_print(f"[GlobalPrinter] PRINT_TO_TERMINAL set to {enabled}", file=sys.stderr)

    def set_log_file(self, log_file_path: str):
        """Change the log file path at runtime."""
        self._set_config(self._print_to_terminal, log_file_path)
        _original_print(f"[GlobalPrinter] Log file changed to: {self._log_file_path}", file=sys.stderr)

    def print(self, *args, level=logging.INFO, sep=" ", end="\n"):
        """
        Our custom print function.
        - If _print_to_terminal is True, print to stdout (using _original_print).
        - Otherwise, log using the FileLogger instance.
        """
        message = sep.join(str(a) for a in args)

        if self._print_to_terminal:
            # Use the original print to avoid recursion
            _original_print(message, end=end)
        else:
            if not self._file_logger:
                # If something went wrong and there's no logger, fallback
                _original_print("File Logger is Not there")
                _original_print(message, end=end)
                return
            if level == logging.DEBUG:
                self._file_logger.log(logging.DEBUG, message)
            elif level == logging.WARNING:
                self._file_logger.log(logging.WARNING, message)
            elif level == logging.ERROR:
                self._file_logger.log(logging.ERROR, message)
            elif level == logging.CRITICAL:
                self._file_logger.log(logging.CRITICAL, message)
            else:
                self._file_logger.log(logging.INFO, message)


# Next, we override builtins.print with our custom wrapper
# that delegates to the GlobalPrinter singleton.

def _global_print_override(*args, sep=" ", end="\n", file=None, flush=False, **kwargs):
    """
    This function will replace the built-in print.
    Extra arguments that aren't recognized (e.g. level=...) are handled via kwargs.
    """
    level = kwargs.pop('level', logging.INFO)

    # We'll ignore 'file' and 'flush' since we override that behavior.
    # We'll also ignore any leftover kwargs that might exist.

    gp = GlobalPrinter.get_instance()
    gp.print(*args, sep=sep, end=end, level=level)


print = _global_print_override


# A separate convenience function if you prefer gprint(...)
def gprint(*args, level=logging.INFO, sep=" ", end="\n"):
    """
    A direct call to GlobalPrinter's .print() method.
    This does the same as the overridden print(..., level=...), but may be clearer.
    """
    GlobalPrinter.get_instance().print(*args, level=level, sep=sep, end=end)


if __name__ == "__main__":
    # Example usage

    # This call now goes through our overridden print
    print("Hello from the default constructor!","This is a test")

    # Reconfigure with new file
    GlobalPrinter.create(print_to_terminal=False, log_file_path="main.log")
    print("Now this goes to main.log or main_1.log if main.log existed.")
    print("Hello from the default constructor!","This is a test")
    print("Info level test")
    print("Debug level test", level=logging.DEBUG)
    print("Warning level test", level=logging.WARNING)
    print("Error level test", level=logging.ERROR)
    print("Critical level test", level=logging.CRITICAL)
    

    # Switch back to terminal
    GlobalPrinter.get_instance().set_print_to_terminal(True)
    print("Back to terminal!")
    # Demonstration of various log levels
    print("Info level test")
    print("Debug level test", level=logging.DEBUG)
    print("Warning level test", level=logging.WARNING)
    print("Error level test", level=logging.ERROR)
    print("Critical level test", level=logging.CRITICAL)

    # Reconfigure to log only to file
    GlobalPrinter.get_instance().set_log_file("test.log")
    GlobalPrinter.get_instance().set_print_to_terminal(False)
    print("Logging to file only now")
    print("Logging to file only now","new line")
    print("Logging to file only now","new line","allss",sep="*****")
    print("Logging to file only now","new line",end="$$$$$$$$$\n")
    

    # Switch back to terminal printing
    GlobalPrinter.get_instance().set_print_to_terminal(True)
    print("Terminal printing restored")
    print("Logging to file only now","new line")
    print("Logging to file only now","new line","allss",sep="*****")
    print("Logging to file only now","new line",end="$$$$$$$$$\n")
