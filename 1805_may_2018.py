#!python3
# 1805_may_2018.py
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
            # FIXME - string parsing is problematic ... '(' within a string
            # will break the lexer
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
        l.sort(reverse=True)
        out = 1
        top = l.pop(0)
        if top < out:
            return out
        while l:
            new = l.pop()
            if new <= 1:
                continue
            out = out + 1
            if out == new:
                pass
            else:
                return out
        return top + 1

    # TESTS
    tests = [([3, 4, -1, 1], 2), ([1, 2, 0], 3)]
    for a, b in tests:
        assert lowint(a) == b
    a = list(range(int(1e7)))
    b = a.pop(len(a) // 2)
    assert lowint(a) == b


def _180520():
    """This problem was asked by Jane Street.

    cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
    and last element of that pair. For example, car(cons(3, 4)) returns 3, and
    cdr(cons(3, 4)) returns 4.

    Given this implementation of cons:

    def cons(a, b):
        def pair(f):
            return f(a, b)
    return pair

    Implement car and cdr."""

    def cons(a, b):
        def pair(f):
            return f(a, b)
        return pair

    def pair_func(*args):
        return args

    def car(pair):
        return pair(pair_func)[0]

    def cdr(pair):
        return pair(pair_func)[-1]

    # TESTS
    assert car(cons(3, 4)) is 3
    assert cdr(cons(3, 4)) is 4


def _180521():
    """This problem was asked by Google.

    An XOR linked list is a more memory efficient doubly linked list. Instead of
    each node holding next and prev fields, it holds a field named both, which
    is an XOR of the next node and the previous node. Implement an XOR linked
    list; it has an add(element) which adds the element to the end, and a
    get(index) which returns the node at index.

    If using a language that has no pointers (such as Python), you can assume
    you have access to get_pointer and dereference_pointer functions that
    converts between nodes and memory addresses."""

    obj_store = {}

    def get_pointer(x):
        p = id(x)
        obj_store.setdefault(p, x)
        return p

    def dereference_pointer(p):
        try:
            return obj_store[p]
        except KeyError:
            raise ValueError('p must be a valid pointer obtained from'
                             ' get_pointer().')

    class Node:
        def __init__(self, value, p_both=None):
            self.value = value
            self.p_both = p_both

    class LinkedList:
        def __init__(self):
            self.tail = None
            self._len = 0

        def __len__(self):
            return self._len

        def add(self, element):
            self._len += 1
            if self.tail is None:
                self.tail = Node(element)
            else:
                new = Node(element, get_pointer(self.tail))
                p_new = get_pointer(new)
                try:
                    self.tail.p_both ^= p_new
                except TypeError:
                    # self.tail.p_both is None; that is, tail has no previous
                    self.tail.p_both = p_new
                self.tail = new
            return None  # modification in place

        def append(self, item):
            return self.add(item)

        def get(self, index):
            if not 0 <= index < len(self):
                raise IndexError('Index out of bounds.')
            n_steps = len(self) - index - 1
            node = self.tail
            p = (get_pointer(node), node.p_both)
            while n_steps > 0:
                node = dereference_pointer(p[1])
                p = (p[1], node.p_both ^ p[0])
                n_steps -= 1
            return node

        def __getitem__(self, key):
            return self.get(key).value


    # TEST
    test_ls = ['hi', 278, ('good', 'bad')]
    ll = LinkedList()
    [ll.append(item) for item in test_ls]
    assert all(ll[i] is test_ls[i] for i in range(len(test_ls)))


def _180522():
    """This problem was asked by Facebook.

    Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count
    the number of ways it can be decoded.

    For example, the message '111' would give 3, since it could be decoded as
    'aaa', 'ka', and 'ak'.

    You can assume that the messages are decodable. For example, '001' is not
    allowed."""

    def count_decodes(string, greedy=None):
        if greedy is None:
            return count_decodes(string, True) + count_decodes(string, False)
        else:
            string = list(string)
            x = int(string.pop(0))

            if greedy:
                if not string:
                    return 0
                x = 10 * x + int(string.pop(0))
                if x > 26:
                    return 0

            if string:
                return count_decodes(string)
            else:
                return 1

            # TEST
    assert count_decodes('111') == 3


def _180523():
    """This problem was asked by Google.

    A unival tree (which stands for "universal value") is a tree where all nodes
    under it have the same value.

    Given the root to a binary tree, count the number of unival subtrees.

    For example, the following tree has 5 unival subtrees:

       0
      / \
     1   0
        / \
       1   0
      / \
     1   1
    """

    class Node:
        def __init__(self, value, left=None, right=None):
            self.value = value
            self.left = left  # type: Node
            self.right = right  # type: Node

    # FIXME - this is all wrong, need to assess the whole subtree
    def count_unival(head: Node) -> int:
        if head is None:
            return 0

        a = head.left
        b = head.right
        count = 0
        if a is None or b is None:
            if a is b:
                count = 1
        else:
            try:
                if a.value == b.value:
                    count = 1
            except AttributeError:
                pass

        return count + count_unival(a) + count_unival(b)

    # TEST
    tree = Node(0, Node(1), Node(0, Node(1, Node(1), Node(1)), Node(0)))
    assert count_unival(tree) == 5


if __name__ == "__main__":
    _180523()