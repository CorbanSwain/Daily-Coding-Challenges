#!/bin/python3
# 1805_may.py
# Corban Swain, 2018
# May Daily Coding Challenges


def _180516():
    """Given a list of numbers, return whether any two sums to k.
    For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
    17. Bonus: Can you do this in one pass?"""

    def sum_in(ls, k):
        try:
            # check if half of k is a value in the list
            # this will raise a ValueError if k / 2 is not in the list
            i_half = ls.index(k / 2)
            
            # if k / 2 is in the list delete one of the instances from the list
            ls = list(ls) # copy to prevent mutating the original list
            del ls[i_half]
        except ValueError:
            pass
        
        # the sum if present if k minus a value in the list is a 
        # number also in the list
        return any(k - x in ls for x in ls)

    assert sum_in([10, 50, 3, 7], 17) is True
    assert sum_in([5, 2, 3], 10) is False
    assert sum_in([5, 2, 3, 5], 10) is True


if __name__ == "__main__":
    _180516()
