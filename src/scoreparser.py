import re
import numpy as np
import pandas as pd

def parse_score(x) -> int:
    """
    Function to extract scores from a string and return their average.
    Designed for parsing scores of the format "Score: X/100", 
    considering only scores between 0 and 100. Ignores amounts 
    outside this range.

    Parameters:
    x (str): The string containing the score(s).

    Returns:
    int: The average of the valid scores, or np.nan if none are valid.
    """
    if pd.isna(x) or (len(x) == 0):
        return np.nan

    x = x.lower().strip()

    # Regex pattern to match scores
    pattern = r'(\d+)\s*/\s*100'
    matches = re.findall(pattern, x)

    # Extract numbers
    valid_amounts = []
    for match in matches:
        # Strip non-numeric parts and convert to float
        if len(match) > 0:
            number = float(match)
            if 0 <= number <= 100:
                valid_amounts.append(number)

    if valid_amounts:
        return sum(valid_amounts) / len(valid_amounts)
    return np.nan