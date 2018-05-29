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

    # FIXME - This is not in linear time because of the sort step
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
    """
    This problem was asked by Jane Street.

    cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
    and last element of that pair. For example, car(cons(3, 4)) returns 3, and
    cdr(cons(3, 4)) returns 4.

    Given this implementation of cons:

    def cons(a, b):
        def pair(f):
            return f(a, b)
    return pair

    Implement car and cdr.
    """

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

    def is_leaf(node: Node) -> bool:
        return node.left is None and node.right is None

    def count_unival_helper(head: Node):
        if head is None:
            return None
        if is_leaf(head):
            return True, 1, head

        out = (count_unival_helper(n) for n in (head.left, head.right))
        out = (tup for tup in out if tup is not None)

        is_uni = True
        total_counts = 0
        for sub_uni, sub_count, sub_node in out:
            if is_uni:
                if sub_uni:
                    is_uni = sub_node.value == head.value
                else:
                    is_uni = False
            total_counts += sub_count

        if is_uni:
            total_counts += 1
        return is_uni, total_counts, head

    def count_unival(head: Node) -> int:
        return count_unival_helper(head)[1]

    # TESTS
    tree = Node(0, Node(1), Node(0, Node(1, Node(1), Node(1)), Node(0)))
    assert count_unival(tree) == 5


def _180524():
    """This problem was asked by Airbnb.

    Given a list of integers, write a function that returns the largest sum of
    non-adjacent numbers. Numbers can be 0 or negative.

    For example, [2, 4, 6, 8] should return 12, since we pick 4 and 8.
    [5, 1, 1, 5] should return 10, since we pick 5 and 5.

    Follow-up: Can you do this in O(N) time and constant space?"""

    # by brute force we can simply calculate all of the possibilities
    def big_sum_1(lst):
        s = None
        for i, x in enumerate(lst):
            for j, y in enumerate(lst):
                valid = all((j is not i, (j - 1) is not i, (j + 1) is not i))
                if valid:
                    if s is None:
                        s = x + y
                    else:
                        s = max(s, x + y)
        return s
    # this will run in O(N^2) time and constant space

    # We can go through the list once and keep track of the top two values
    # and their indices, not allowing two numbers side-by-side to both be in
    # the list, and taking precedence to keep the biggest number not the the
    # second biggest ... this doesnt work ...

    # Alternatively, keep track of the top three numbers, if the top 2 are
    # next to each other, return that, otherwise return the top plus the
    # third highest
    class Item:
        def __init__(self, index, val):
            self.index = index
            self.val = val

    def big_sum_2(lst):
        if len(lst) < 3:
            raise ValueError('List must have at least 3 values')
        top = [None, None, None]
        for i, x in enumerate(lst):
            for j, item in enumerate(top):
                if item is None:
                    top[j] = Item(i, x)
                    break
                if x > item.val:
                    del top[-1]
                    top.insert(j, Item(i, x))
                    break

        diff = abs(top[0].index - top[1].index)
        return top[0].val + (top[2].val if diff == 1 else top[1].val)

    # TESTS
    tests = (([2, 4, 6, 8], 12), ([5, 1, 1, 5], 10))
    for query, ans in tests:
        assert big_sum_1(query) == ans
        assert big_sum_2(query) == ans

    from timeit import timeit
    print('Time 1: %8.3f s' % timeit(lambda: big_sum_1(range(int(1e3))),
                                     number=10))
    print('Time 2: %8.3f s' % timeit(lambda: big_sum_2(range(int(1e3))),
                                     number=10))
    import numpy as np
    import matplotlib.pyplot as plt
    ns = [3, 10, 50, 100, 500, 1e3]
    y1 = np.zeros(len(ns))
    y2 = np.zeros(len(ns))
    for i, n in enumerate(ns):
        y1[i] = timeit(lambda: big_sum_1(range(int(n))), number=100)
        y2[i] = timeit(lambda: big_sum_2(range(int(n))), number=100)

    x = np.array(ns)
    plt.style.use('seaborn-notebook')
    ax1 = plt.subplot(211)
    plt.plot(x, y1, x, y2)
    # ax1.set_xscale('log')
    ax2 = plt.subplot(212)
    # ax2.set_xscale('log')
    plt.plot(x, y2)
    plt.tight_layout()
    plt.show()


def _180525():
    """This problem was asked by Apple.

    Implement a job scheduler which takes in a function f and an integer n, and
    calls f after n milliseconds."""

    from threading import Thread
    from time import sleep

    class Job(Thread):
        def __init__(self, func, delay=0, **kwargs):
            self.delay = delay
            super().__init__(target=func, **kwargs)

        def run(self):
            if self.delay > 0:
                sleep(self.delay)
            super().run()

    def schedule(func, delay: int=0):
        job = Job(func, delay/1000)
        job.start()

    # TEST
    def make_echo_func(x):
        def echo_func():
            print(x)
        return echo_func

    for i in range(10):
        s = '... Echoing %2d!' % (i + 1)
        schedule(make_echo_func(s), (i + 1) * 1000)
    print('Done Main!')


def _180526():
    """This problem was asked by Twitter.

    Implement an autocomplete system. That is, given a query string s and a set
    of all possible query strings, return all strings in the set that have s as
    a prefix.

    For example, given the query string de and the set of strings
    [dog, deer, deal], return [deer, deal].

    Hint: Try preprocessing the dictionary into a more efficient data structure
    to speed up queries."""

    from functools import reduce

    class Entry:
        def __init__(self, s):
            self.s = s

        def __str__(self):
            return self.s

    class Node:
        def __init__(self, value, depth=-1):
            self.value = value
            self._depth = depth
            self.sub_nodes = {}  # type: dict
            self.entries = []  # type: list

        def __getitem__(self, key):
            return self.sub_nodes[key]

        def __setitem__(self, key, value):
            self.sub_nodes[key] = value

        def parse_entries(self, new_entries=None):
            if new_entries is not None:
                self.entries = new_entries
            idx = self._depth + 1
            for entry in self.entries:
                try:
                    c = entry.s[idx]
                except IndexError:
                    continue
                try:
                    sub_node = self[c]
                except KeyError:
                    sub_node = Node(c, idx)
                    self[c] = sub_node
                sub_node.entries.append(entry)

            # base case comes when self.sub_nodes is an empty dict
            for n in self.sub_nodes.values():
                n.parse_entries()
            return None  # modification in place

    class Autocompleter:
        def __init__(self, dictionary=None):
            self.dictionary = dictionary
            self._tree_dict = None  # type: Node

        # for brute force, we can just look at every item in the list
        def autocomplete_bf(self, query, useLambda=True):
            if useLambda:
                return filter(lambda s: query in s and s.index(query) is 0,
                              self.dictionary)
            else:
                return [s for s in self.dictionary if query in s and
                        s.index(query) is 0]

        # alternatively we can construct a tree where each node layer contains
        # a list of the items that have a character in each successive position
        @property
        def tree_dict(self) -> Node:
            if self._tree_dict is None:
                entries = map(lambda s: Entry(s), self.dictionary)
                self._tree_dict = Node('')
                self._tree_dict.parse_entries(entries)
            return self._tree_dict

        def autocomplete(self, query):
            try:
                node = reduce(lambda n, char: n[char], query, self.tree_dict)
            except KeyError:
                return []
            return map(lambda e: e.s, node.entries)

    # TESTS
    d = ['dog', 'deer', 'deal']
    q = 'de'
    ac = Autocompleter(d)
    assert set(ac.autocomplete_bf(q)) == {'deer', 'deal'}
    assert set(ac.autocomplete(q)) == {'deer', 'deal'}

    from timeit import timeit
    from random import randint

    def wrap(statement, *args, **kwargs):
        def fun():
            statement(*args, **kwargs)
        return fun

    d = map(lambda _: str(randint(0, int(1e7))), range(int(1e6)))
    ac = Autocompleter(d)
    q = '9542'
    n_queries = 1000000
    print('Brute force (ls cmp): %8.4f s' % timeit(
        wrap(ac.autocomplete_bf, q, False),
        number=n_queries))
    print('Brute force (filter): %8.4f s' % timeit(wrap(ac.autocomplete_bf, q),
                                                   number=n_queries))
    print('      Compiling tree: %8.4s s' % timeit(wrap(ac.autocomplete, ''),
                                                   number=1))
    print('         Tree Method: %8.4f s' % timeit(wrap(ac.autocomplete, q),
                                                   number=n_queries))
    # > Brute force (ls cmp):   1.4382 s
    # > Brute force (filter):   0.0366 s
    # >       Compiling tree:     2.45 s
    # >          Tree Method:   0.1232 s

    # Notes
    # N = number of items in dictionary
    # K = number of queries
    # L = number of chars in the query

    # The brute force method executes in O(N * K) time ... with list
    # comprehension, but filter seems much faster

    # The compilation of the tree executes in O(N^2) time ( ... at worst -> all
    # elements the same ... maybe call set on the dictionary at creation)

    # The tree autocomplete method executed in O(K * L) time,
    # this is independent of N. So although the initial compilation will take
    # a while, after many queries, orders of magnitude time will be saved,
    # especially for a large dictionary.


def _180527():
    """
    This problem was asked by Amazon.

    There exists a staircase with N steps, and you can climb up either 1 or 2
    steps at a time. Given N, write a function that returns the number of unique
    ways you can climb the staircase. The order of the steps matters.

    For example, if N is 4, then there are 5 unique ways:

        1, 1, 1, 1
        2, 1, 1
        1, 2, 1
        1, 1, 2
        2, 2

    What if, instead of being able to climb 1 or 2 steps at a time, you could
    climb any number from a set of positive integers X? For example, if
    X = {1, 3, 5}, you could climb 1, 3, or 5 steps at a time.
    """

    from functools import reduce

    # if n = num_steps return 1, if n < num_steps return 0 else return
    # ways to climb(n - num_steps)
    def ways_to_climb(n, num_steps=None):
        if num_steps is None:
            return ways_to_climb(n, 1) + ways_to_climb(n, 2)
        n = n - num_steps
        return 0 if n < 0 else 1 if n == 0 else ways_to_climb(n)


    # for the advanced case, simply go through the set of possible steps in
    # the initial setup
    def ways_to_climb_2(n, step_set, num_steps=None):
        if num_steps is None:
            return reduce(
                lambda tot, x: tot + ways_to_climb_2(n, step_set, x),
                step_set, 0)
        n = n - num_steps
        return 0 if n < 0 else 1 if n == 0 else ways_to_climb_2(n, step_set)

    # TESTS
    assert ways_to_climb(4) == 5
    assert ways_to_climb_2(5, {1, 3, 5}) == 5
    '''
    Ways to climb 5 steps taking 1, 3, or 5 steps at a time:
    1: 1, 1, 1, 1, 1
    2: 1, 3, 1
    3: 1, 1, 3
    4: 3, 1, 1
    5: 5
    '''


def _180528():
    """
    This problem was asked by Amazon.

    Given an integer k and a string s, find the length of the longest
    substring that contains at most k distinct characters.

    For example, given s = "abcba" and k = 2, the longest substring with k
    distinct characters is "bcb".
    """

    # at each position we can keep track of the number of distinct characters
    # recursion might be our friend here
    # subtract one from k upon encountering the same character then call again
    # with the substring
    # need to start at each position in the string ... maybe also pass an
    # index in the recursive loop
    # base case: k = 0 -> return length
    # ixd == len(s) -> return length
    def find_long(k, s):
        return max(map(lambda i: _find_long_recurse(k, s, i), range(len(s))))

    def _find_long_recurse(k, s, idx=0, char_set=None, length=0):
        if idx == len(s):
            return length

        char_set = char_set if char_set is not None else set()
        char_set.add(s[idx])

        if len(char_set) > k:
            return length
        else:
            return _find_long_recurse(k, s, idx + 1, char_set, length + 1)

    # This runs in O(N^2) time -> looking through every remaining element for
    # each position

    # probably makes more sense to do this in a simple loop
    def find_long_2(k, s):
        if len(s) == 0 or k < 1:
            return 0
        best = -1
        idx = -1
        while best < len(s) - idx:
            char_set = set()
            run_length = 0
            idx += 1
            for char in s[idx:]:
                char_set.add(char)
                if len(char_set) > k:
                    best = run_length if run_length > best else best
                    break
                else:
                    run_length += 1
        return best

    def wrap(target, *args, **kwargs):
        def f():
            target(*args, **kwargs)
        return f

    assert find_long(2, 'abcba') == 3
    assert find_long_2(2, 'abcba') == 3

    from timeit import timeit
    k, s = 2, 'abcba'
    print('Method 1: %10.4f s' % timeit(wrap(find_long, k, s)))
    print('Method 2: %10.4f s' % timeit(wrap(find_long_2, k, s)))
    # > Method 1:     5.4519 s
    # > Method 2:     2.3880 s


def _180529():
    """
    This problem was asked by Google.

    The area of a circle is defined as πr^2. Estimate π to 3 decimal places
    using a Monte Carlo method.

    Hint: The basic equation of a circle is x^2 + y^2 = r^2.
    """
    from random import random

    # for a fouth of a unit circle (r = 1) with in a unit square (s = 1)
    # the area inside should be A_in = pi/4 while the total area A_tot = 1
    # therefore A_in / A_tot will approach pi / 4
    # we'll use this ratio times 4 to be our estimator for pi
    # a random dart in the unit square will be our estimator ...
    # how to assess precision?  error should go down as the number of darts
    # go up ... unsure of the specific formula
    # ... of course, since we are squaring the random numbers we should
    # square the precision and take the inverse to get the necessary number
    # of itterations

    def mc_pi(precision=1e-3, n=None) -> float:
        n = round(1 / (precision ** 2)) if not n else n
        n_in = 0
        for _ in range(n):
            r = random() ** 2 + random() ** 2
            if r <= 1:
                n_in += 1
        return 4 * n_in / n

    # TESTS
    from math import pi
    import matplotlib.pyplot as plt
    import numpy as np

    pi_est = mc_pi()
    assert abs(pi_est - pi) <= 5e-3
    print('PI estimate: %f' % pi_est)
    print('Error:       %f' % abs(pi_est - pi))
    # > PI estimate: 3.141464
    # > Error:       0.000129

    vals = np.unique(np.round(np.logspace(0, 6, num=15))).astype(np.int)
    n_reps = 10
    x = np.repeat(vals, n_reps)
    y = [abs(mc_pi(n=v) - pi) for v in x]
    fig = plt.figure(1)
    ax = fig.subplots()
    ax.scatter(x, y, s=2**2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([min(y), max(y)])
    ax.set_xlim([min(x), max(x)])
    plt.show()


if __name__ == "__main__":
    _180529()