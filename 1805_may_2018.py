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
        # solution without division

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
    tests = [([1, 2, 3, 4, 5], [120, 60, 40, 30, 24]),
             ([3, 2, 1], [2, 3, 6]),
             ([], []),
             ([20], [1])]
    for a, b in tests:
        assert all(x - y < epsilon for x, y in zip(prod_list_1(a), b))
        assert all(x - y < epsilon for x, y in zip(prod_list_2(a), b))


def _180518():
    """This problem was asked by Google.

    Given the root to a binary tree, implement serialize(root), which serializes
    the tree into a string, and deserialize(s), which deserializes the string
    back into the tree.

    For example, given the following Node class

    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    The following test should pass:

    node = Node('root', Node('left', Node('left.left')), Node('right'))
    assert deserialize(serialize(node)).left.left.val == 'left.left'
    """

    from enum import Enum, auto

    class Node:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def quotify(s):
        return '\'' + s + '\''

    def serialize(n):
        if n is None:
            return 'None'
        ser_str = '%s(%s,%s)' % (quotify(n.val),
                                 serialize(n.left),
                                 serialize(n.right))
        return ser_str

    class Token(Enum):
        SEPARATOR = auto()
        LITERAL = auto()

    def lex(ser_str):
        tokens = []
        is_parsing_str = False
        x = ''
        for c in ser_str:
            if c in ['(', ')', ',']:
                tokens.append((Token.SEPARATOR, c))
            elif c == '\'':
                if is_parsing_str:
                    tokens.append((Token.LITERAL, x))
                    x = ''
                    is_parsing_str = False
                else:
                    is_parsing_str = True
            else:
                x += c
                if not is_parsing_str:
                    if x == 'None':
                        tokens.append((Token.LITERAL, None))
                        x = ''
        return tokens

    def deserialize(ser_str=None, tokens=None):
        if tokens is None:
            tokens = lex(ser_str)

        t0, v0 = tokens.pop(0)
        assert t0 is Token.LITERAL
        n = Node(v0)
        t_left = []
        t_right = []
        is_parsing_left = True
        level = 0
        for t, v in tokens:
            if t is Token.SEPARATOR:
                if v == '(':
                    level += 1
                elif v == ')':
                    level -= 1
                elif v == ',':
                    if level == 1:
                        is_parsing_left = False
                else:
                    raise ValueError('Unexpected separator token \'%s\'.' % v)

                if level == 1:
                    continue
                if level == 0:
                    break

            if is_parsing_left:
                t_left.append((t, v))
            else:
                t_right.append((t, v))

        if not (len(t_left) == 1 and t_left[0][1] is None):
            n.left = deserialize(tokens=t_left)
        if not (len(t_right) == 1 and t_right[0][1] is None):
            n.right = deserialize(tokens=t_right)
        return n

    # TEST
    node = Node('root', Node('left', Node('left.left')), Node('right'))
    assert deserialize(serialize(node)).left.left.val == 'left.left'


def _180519():
    """This problem was asked by Stripe.

    Given an array of integers, find the first missing positive integer in
    linear time and constant space. In other words, find the lowest positive
    integer that does not exist in the array. The array can contain duplicates
    and negative numbers as well.

    For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0]
    should give 3.

    You can modify the input array in-place.
    """

    def lowint(l):
        top = max(l)
        if min(l) < 1:
            l = [x for x in l if x > 0]
        for x in range(1, top):
            if x not in l:
                return x
            else:
                while True:
                    try:
                        del l[l.index(x)]
                    except ValueError:
                        break
        return top + 1

    # TESTS
    tests = [([3, 4, -1, 1], 2), ([1, 2, 0], 3)]
    for a, b in tests:
        assert lowint(a) == b

    a = list(range(int(5e4)))
    b = a.pop(len(a) // 2)
    assert lowint(a) == b


if __name__ == "__main__":
    _180519()