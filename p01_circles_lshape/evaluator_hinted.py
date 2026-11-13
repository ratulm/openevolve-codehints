import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "common"))
import harness
import task


def evaluate(program_path):
    return harness.evaluate_hinted(program_path, task)
