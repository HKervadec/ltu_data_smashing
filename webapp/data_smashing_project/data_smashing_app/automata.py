from random import random, randrange


class Automata:
	def __init__(self, proba):
		self.size = len(proba)

		self.P = [s[:] for s in proba]	

		self.current_state = randrange(self.size)

	def next_state(self):
		v = random()

		for j in range(self.size):
			s = sum(self.P[self.current_state][:j+1])

			if v < s:
				self.current_state = j
				return

	def gen_stream(self, size):
		r = [0] * size

		for i in xrange(size):
			r[i] = self.current_state

			self.next_state()

		return r

# if __name__ == "__main__":
# 	P = [[0] * 3 for _ in range(3)]

# 	P[0][0] = 0.5
# 	P[0][1] = 0.3
# 	P[0][2] = 0.2

# 	P[1][0] = 0.25
# 	P[1][1] = 0.4
# 	P[1][2] = 0.35

# 	P[2][0] = 0.15
# 	P[2][1] = 0.05
# 	P[2][2] = 0.8

# 	a = Automata(P)

# 	print ''.join(map(str, a.gen_stream(10)))