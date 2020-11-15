

class Statistics(object):
	"""docstring for Statistics"""
	def __init__(self):
		super(Statistics, self).__init__()
		
		self.train_loss_data = list()
		self.train_accuracy_data = list()
		self.test_loss_data = list()
		self.test_accuracy_data = list()

	def __str__(self):
		print(self.test_accuracy_data[-1])

	def add_train_loss(self, value):
		self.train_loss_data.append(value)

	def add_train_accuracy(self, value):
		self.train_accuracy_data.append(value)

	def add_test_loss(self, value):
		self.test_loss_data.append(value)

	def add_test_accuracy(self, value):
		self.test_accuracy_data.append(value)

	def get_train_loss(self):
		return self.train_loss_data[-1]

	def get_train_accuracy(self):
		return self.train_accuracy_data[-1]

	def get_test_loss(self):
		return self.test_loss_data[-1]

	def get_test_accuracy(self):
		return self.test_accuracy_data[-1]

	def plot_statistics(self):
		print("PLOT FUNCTION")
		pass