import numpy as np
import sys

class Datasmashing:

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    def stream_sumation(self, s1, s2):
        '''
        Function sums streams s1 and s2
        '''
        result = []
        for i1, i2 in zip(s1, s2):
            if i1 == i2:
                result.append(i1)

        return result

    def fwn(self, length):
        '''
        function create flat white noise (uniform distribution
        of alphabet symbols) of length length
        '''
        return np.random.randint(1, self.alphabet_size+1, size=length)

    def independent_stream_copy(self, s):
        '''
        function make intedpendent stream copy of s
        '''
        return self.stream_sumation(s, self.fwn(len(s)))
        # TODO: what if it is empty?

    def stream_inversion(self, s):
        '''
        function invert stream s
        '''
        stream_copies = []
        shortest_length = sys.maxint
        for i in range(self.alphabet_size-1):
            s_copy = self.independent_stream_copy(s)
            if shortest_length > len(s_copy):
                shortest_length = len(s_copy)
            stream_copies.append(s_copy)

        stream_copies_matrix = np.empty((self.alphabet_size - 1, shortest_length))
        for i in range(len(stream_copies)):
            stream_copies_matrix[i, :] = stream_copies[i][:shortest_length]

        stream_invert = []
        for i in range(shortest_length):
            inverted_el = list(filter(lambda x: x not in stream_copies_matrix[:, i], range(1, self.alphabet_size+1)))
            if len(inverted_el) == 1:
                stream_invert.append(inverted_el[0])

        return stream_invert

    def deviation(self, s, l):
        coef = (self.alphabet_size - 1.0) / self.alphabet_size

        sum = 0
        for i in range(len(s)):
            for j in range(l):
                if i + j > len(s):
                    break

                x = s[i:i+j]

                top = 0

                bot = self.alphabet_size ** (2 * len(x))

                sum += top / bot


        return coef * sum

    def phi(self, s, x):
        '''

        :param s: string
        :param x: substring of s
        :return:
        '''
        s = ''.join(s)
        x = ''.join(x)
        result = []
        denom = s.count(x)

        for i in range(self.alphabet_size):
            result.append(s.count(x + str(i)) / denom)

        return result

for i in range(1):
    a = [1, 1, 4, 4, 2, 1, 1, 4, 3, 3, 2, 1, 4, 1, 1, 1, 4, 4, 1, 1]
    b = [1, 4, 4, 2, 1, 1, 4, 3, 3, 2, 1, 4, 1, 1, 1, 4, 4, 1, 1, 4]
    d = []
    for i in range(100):
        d = d + a
    ds = Datasmashing(4)
    print ds.stream_inversion(d)
