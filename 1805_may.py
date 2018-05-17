#!/bin/python3
# 1805_may.py
# Corban Swain, 2018
# May Daily Coding Challenges

# 180516
# Given a list of numbers, return whether any two sums to k. 
# For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17. 
# Bonus: Can you do this in one pass?


def sum_in(ls, k):
    half = k / 2
    if half in ls:
        del ls[ls.index(half)]
    return any(k - x in ls for x in ls)


assert sum_in([10, 50, 3, 7], 17)
