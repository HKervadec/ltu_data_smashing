import numpy as np
import sys
import itertools as it


class Datasmashing:

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size

    def stream_sumation(self, s1, s2):
        '''
        Function sums streams s1 and s2
        param: s1 - stream one
        param: s2 - stream two
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
        param: length - length of stream we crate
        '''
        return np.random.randint(1, self.alphabet_size+1, size=length)

    def independent_stream_copy(self, s):
        '''
        function make intedpendent stream copy of s
        param: s - stream for which we create independetn stream copy
        '''
        return self.stream_sumation(s, self.fwn(len(s)))
        # TODO: what if it is empty?

    def stream_inversion(self, s):
        '''
        function invert stream s
        param: s - stream we invert
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
        '''
        function calculate deviation measure defined in datasmashing algorithm_test
        param: s - stream for which deviation is calculated
        param: l_max - maximal lenght used in deviation
        '''

        coef = float(self.alphabet_size - 1.0) / self.alphabet_size
        
        uniform_v = np.ones(self.alphabet_size) / self.alphabet_size

        # for empty string
        phi_v = self.phi(s, '')
        numerator = np.linalg.norm(phi_v - uniform_v, np.inf)                                
        denominator = 1
        sum1 = numerator / denominator

        for l in range(1, l_max + 1):            
            for variation in it.product(range(1, self.alphabet_size + 1), repeat=l):
                # phi vector
                phi_v = self.phi(s, variation)

                if sum(phi_v) != 0:
                    # print phi_v
                    numerator = np.linalg.norm(phi_v - uniform_v, np.inf)
                                    
                    denominator = self.alphabet_size ** (2 * len(variation))

                    sum1 += numerator / denominator

        return coef * sum1

    def phi(self, s, x):
        '''
        function calcuate devation phi defined in data smashing algorithm
        param s: string
        param x: substring of s
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

    def deviation_magic(self, s, l_max):
        '''
        function calculate deviation measure using Denis approach
        param: s - stream for which deviation is calculated
        param: l_max - maximal lenght used in deviation
        '''
        stream_len = len(s)

        ksi = 0

        i = 1
        while i <= l_max:
            count = np.zeros(self.alphabet_size**i)
            U = (1/self.alphabet_size ** i) * np.ones(self.alphabet_size ** i)

            num = stream_len - (i - 1)
            for j in range(num):
                for j2 in range(self.alphabet_size ** i):
                    atom = self.dec2bin(j2, i)
                    if atom == ''.join(str(el) for el in s[j: j+i]):
                        print count[j2-1]
                        count[j2-1] = count[j2-1] + 1
                        print count[j2-1]
            print count
            count = count / num
            w = np.linalg.norm(np.absolute(count-U), np.inf)
            w = w / (self.alphabet_size ** (2*(i-1)))
            ksi = ksi + w
            i = i + 1

        return 0.5 * ksi


    def dec2bin(self, num, length):
        '''
        function is used to change stream made of 0 and 10
        to stream made of 1 and 2
        param: num - stream to be changed
        param: length - length 
        '''

        binary_string = bin(num)[2:]
        num_of_zeros = length - len(binary_string)
        binary_string = binary_string.replace('1', '2').replace('0', '1')
        if num_of_zeros > 0:
            return '1' * num_of_zeros + binary_string
        return binary_string


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
        '''
        annihilation_circut calculate deviations between s1 and s2 such
        that we get similarity between those two streams
        param: s1 - stream 1
        param: s2 - stream 2
        param: threshold - threshold is border which tell if streams are similar or not
        '''
        s1_inverted = self.stream_inversion(s1)
        s2_inverted = self.stream_inversion(s2)

        l = int(np.log(1 / treshold) / np.log(self.alphabet_size))
        
        # print len(self.stream_sumation(s1, s1_inverted))

        epsilon11 = self.deviation(self.stream_sumation(s1, s1_inverted), l)
        epsilon21 = self.deviation(self.stream_sumation(s1_inverted, s2), l)
        epsilon12 = self.deviation(self.stream_sumation(s1, s2_inverted), l)
        epsilon22 = self.deviation(self.stream_sumation(s2, s2_inverted), l)

        matrix = np.array([[epsilon11, epsilon12], [epsilon21, epsilon22]])

        return matrix, (epsilon22 < treshold and epsilon11 < treshold)

    def copy_smashing(self, s1, threshold):
        '''
        function makes deviations with strem itself
        param: s1 - stream 
        param: threshold - threshold is border which tell if streams are similar or not
        '''

        s_copy = self.independent_stream_copy(s1)
        s1_inverted = self.stream_inversion(s1)

        l = int(np.log(1 / threshold) / np.log(self.alphabet_size))       

        epsilon11 = self.deviation(self.stream_sumation(s_copy, s1_inverted), l)

        return epsilon11
