class NeuralNetwork:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		self.lr = learning_rate
		self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
		self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
		self.activation_func = lambda x: scipy.special.expit(x)


	def train(self, inputs_list, target_list):
		inputs = np.array(inputs_list,ndmin=2).T
		targets = np.array(target_list,ndmin=2).T

		#Calculate the final output
		hidden_inputs = np.dot(self.wih,inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)

		#calculate the error
		error = targets - final_outputs
		hidden_error = np.dot(self.who.T, error)

		#updating link weights
		self.who += self.lr * np.dot(error*final_outputs*(1.0-final_outputs),hidden_outputs.T)
		self.wih += self.lr * np.dot(hidden_error*hidden_outputs*(1.0-hidden_outputs),inputs.T)
		

	def classify(self, inputs_list):
		inputs = np.array(inputs_list,ndmin=2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_func(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)
		return final_outputs