import numpy as np
import sys
import itertools as it


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

    def deviation(self, s, l_max):
        coef = float(self.alphabet_size - 1.0) / self.alphabet_size

        uniform_v = np.ones(self.alphabet_size) / self.alphabet_size

        # for empty string
        phi_v = self.phi(s, '')
        numerator = np.linalg.norm(phi_v - uniform_v, np.inf)                                
        denominator = 1
        sum = numerator / denominator

        for l in range(1, l_max + 1):
            for variation in it.product(range(1, self.alphabet_size + 1), repeat=l):
                # phi vector
                phi_v = self.phi(s, variation)

                numerator = np.linalg.norm(phi_v - uniform_v, np.inf)
                                
                denominator = self.alphabet_size ** (2 * len(variation))

                sum += numerator / denominator

        return coef * sum

    def phi(self, s, x):
        '''

        :param s: string
        :param x: substring of s
        :return:
        '''

        s = ''.join(str(el) for el in s)
        x = ''.join(str(el) for el in x)
        result = []
        
        for i in range(1, self.alphabet_size + 1):
            nominator = s.count(x + str(i))
            result.append(float(nominator))

        denom = sum(result)

        if denom == 0:
            return [0] * self.alphabet_size
        return [x / float(denom) for x in result]

    def generate(self, probs, length):
        '''
        function generate random vector wiht distribution probs
        param:
        probs - probabilities
        length - generated vector length
        '''
        # from scipy import stats
        # symbols = range(1, self.alphabet_size + 1)
        # custm = stats.rv_discrete(name='custm', values=(symbols, probs))

        # return custm.rvs(size=length)
        vector = []
        for i in range(1, self.alphabet_size + 1):
            vector = vector + ([i] * int(probs[i-1] * length))
        from random import shuffle
        shuffle(vector)
        shuffle(vector)
        return vector

    def annihilation_circut(self, s1, s2, treshold):
        s1_inverted = self.stream_inversion(s1)
        s2_inverted = self.stream_inversion(s2)

        l = int(np.log(1 / treshold) / np.log(self.alphabet_size))
        
        epsilon11 = self.deviation(self.stream_sumation(s1, s1_inverted), l)
        epsilon21 = self.deviation(self.stream_sumation(s1_inverted, s2), l)
        epsilon12 = self.deviation(self.stream_sumation(s1, s2_inverted), l)
        epsilon22 = self.deviation(self.stream_sumation(s2, s2_inverted), l)

        matrix = np.array([[epsilon11, epsilon12], [epsilon21, epsilon22]])

        return matrix, (epsilon22 < treshold and epsilon11 < treshold)
