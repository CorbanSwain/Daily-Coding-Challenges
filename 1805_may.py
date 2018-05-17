#!/bin/python3
# 1805_may.py
# Corban Swain, 2018
# May Daily Coding Challenges


def _180516():
    """Given a list of numbers, return whether any two sums to k.
    For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is
    17.

    Bonus: Can you do this in one pass?"""

    def sum_in(ls, k):
        try:
            del ls[ls.index(k / 2)]
        except ValueError:
            pass
        return any(k - x in ls for x in ls)

    # TESTS
    assert sum_in([10, 50, 3, 7], 17) is True
    assert sum_in([5, 2, 3], 10) is False
    assert sum_in([5, 2, 3, 5], 10) is True


def _180517():
    """This problem was asked by Uber.

    Given an array of integers, return a new array such that each element at
    index i of the new array is the product of all the numbers in the
    original array except the one at i.

    For example, if our input was [1, 2, 3, 4, 5], the expected output would be
    [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output
    would be [2, 3, 6].

    Follow-up: what if you can't use division?"""

    def prod_list_1(ls):
        # compute the product of all items
        mult = 1
        for x in ls:
            mult *= x

        # result is each item divided by the total product
        return [mult / x for x in ls]

    def prod_list_2(ls):
        # create a list of ones
        out = [1 for _ in ls]
        for i, x in enumerate(ls):
            for j, o in enumerate(out):
                if j is not i:
                    # multiply each element in out by each value in the list
                    # as long as the indices are not the same
                    out[j] = o * x
        return out

    # TESTS
    epsilon = 1e-10
    tests = [
        ([1, 2, 3, 4, 5], [120, 60, 40, 30, 24]),
        ([3, 2, 1], [2, 3, 6]),
        ([], []),
        ([20], [1]),
        ('hello', [1])
    ]
    for a, b in tests:
        assert all(x - y < epsilon for x, y in zip(prod_list_1(a), b))
        assert all(x - y < epsilon for x, y in zip(prod_list_2(a), b))


if __name__ == "__main__":
    _180517()