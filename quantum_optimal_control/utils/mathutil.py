"""
mathutil.py - A module for math constants and operations.
"""

import numpy as np

def get_creation_operator(d):
    """
    Construct the creation operator, truncated at level d.
    Args:
    d :: int - the level to truncate the operator at (d >= 1).
    Returns:
    creation_operator :: np.matrix - the creation operator at level d.
    """
    creation_operator = np.zeros((d, d))
    
    for i in range(1, d):
        creation_operator[i][i - 1] = np.sqrt(i)

    return creation_operator
            

def get_annihilation_operator(d):
    """
    Construct the annihilation operator, truncated at level d.
    Args:
    d :: int - the level to truncate the operator at (d >= 1).
    Returns:
    annihilation_operator :: np.matrix - the annihilation operator at level d.                
    """
    annihilation_operator = np.zeros((d, d))
    
    for i in range(d - 1):
        annihilation_operator[i][i + 1] = np.sqrt(i + 1)

    return annihilation_operator                                


_CA_TEST_COUNT = 1000

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: nothing
    """

    # Use the fact that (create)(annihilate) is the number operator.
    for i in range(1, _CA_TEST_COUNT):
        number_operator = np.zeros((_CA_TEST_COUNT, _CA_TEST_COUNT))
        for j in range(_CA_TEST_COUNT):
            number_operator[j][j] = j
        supposed_number_operator = np.matmul(get_creation_operator(i), get_annihilation_operator(i))
        assert number_operator.all() == supposed_number_operator.all()
        

if __name__ == "__main__":
    _tests()
