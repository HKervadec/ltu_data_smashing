from random import random, randrange


class Automata:
	def __init__(self, proba):
		self.size = len(proba)

		self.P = [l[:] for l in proba]

		for i in xrange(self.size):
			for j in xrange(1, self.size):
				self.P[i][j] += self.P[i][j-1]	

		# first state of automata
		self.current_state = randrange(self.size)

	def next_state(self):

		random_value = random()
		for j in xrange(self.size):
			if random_value < self.P[self.current_state][j]:
				self.current_state = j
				return

	def gen_stream(self, length):
		r = [0] * length

		for i in xrange(length):
			r[i] = self.current_state + 1

			self.next_state()

		return r

# if __name__ == "__main__":
# 	P = [[0] * 2 for _ in range(2)]

# 	P[0][0] = 0.8
# 	P[0][1] = 0.2
# 	# P[0][2] = 0.2

# 	P[1][0] = 0.6
# 	P[1][1] = 0.4
# 	# P[1][2] = 0.35

# 	# P[2][0] = 0.15
# 	# P[2][1] = 0.05
# 	# P[2][2] = 0.8

# 	a = Automata(P)

# 	print ''.join(map(str, a.gen_stream(10)))