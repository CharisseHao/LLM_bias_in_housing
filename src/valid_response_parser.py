import re
import numpy as np
import pandas as pd

def parse_response(x) -> int:
    """
    Function to determine whether or not a response followed
    the given prompt instructions.

    Parameters:
    x (str): The string containing the score(s).

    Returns:
    int: True if the response followed the prompt, else False
    """
    if pd.isna(x) or (len(x) == 0):
        return np.nan

    x = x.strip()

    # Regex pattern to match scores
    pattern = r'Score: (\d+)/100'
    matches = re.findall(pattern, x)

    return min(len(matches), 1)