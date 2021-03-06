import matplotlib.pyplot as plt
import numpy as np



class Statistics(object):
	"""docstring for Statistics"""
	def __init__(self, test_eval_rate, train_eval_rate=1):
		super(Statistics, self).__init__()
		
		self.test_eval_rate = test_eval_rate
		self.train_eval_rate = train_eval_rate

		self.train_loss_data = {'y': list(), 'x':list()}
		self.train_accuracy_data = {'y': list(), 'x':list()}
		self.test_loss_data = {'y': list(), 'x':list()}
		self.test_accuracy_data = {'y': list(), 'x':list()}

	def __str__(self):
		print(self.test_accuracy_data[-1])

	def add_train_loss(self, value):
		self.train_loss_data['y'].append(value)
		self.train_loss_data['x'] = np.arange(0, len(self.train_loss_data['y'] * self.train_eval_rate), self.train_eval_rate)

	def add_train_accuracy(self, value):
		self.train_accuracy_data['y'].append(value)
		self.train_accuracy_data['x'] = np.arange(0, len(self.train_accuracy_data['y'] * self.train_eval_rate), self.train_eval_rate)

	def add_test_loss(self, value):
		self.test_loss_data['y'].append(value)
		self.test_loss_data['x'] = np.arange(0, len(self.test_loss_data['y'] * self.test_eval_rate), self.test_eval_rate)

	def add_test_accuracy(self, value):
		self.test_accuracy_data['y'].append(value)
		self.test_accuracy_data['x'] = np.arange(0, len(self.test_accuracy_data['y']) * self.test_eval_rate, self.test_eval_rate)

	def get_train_loss(self):
		return self.train_loss_data[-1]

	def get_train_accuracy(self):
		return self.train_accuracy_data[-1]

	def get_test_loss(self):
		return self.test_loss_data[-1]

	def get_test_accuracy(self):
		return self.test_accuracy_data[-1]

	def smooth(self, x_in, N):
		# x = copy(x_in)
		cumsum = np.cumsum(np.insert(x_in, 0, 0)) 
		return (cumsum[N:] - cumsum[:-N]) / float(N)

	def plot_statistics(self, **kwargs):
		
		smooth_N = 25

		fig, ax = plt.subplots(1, 2, figsize=(12,6))
		ax[0].plot(self.train_loss_data['x'][:-(smooth_N-1)], self.smooth(self.train_loss_data['y'], smooth_N))
		ax[0].plot(self.test_loss_data['x'], self.test_loss_data['y'])
		ax[0].set_title("Loss")
		ax[0].legend(('Train', 'Test'), loc='upper right')

		ax[1].plot(self.train_accuracy_data['x'][:-(smooth_N-1)], self.smooth(self.train_accuracy_data['y'], smooth_N))
		ax[1].plot(self.test_accuracy_data['x'], self.test_accuracy_data['y'])
		if kwargs['hline']:
			ax[1].hlines((kwargs['hline']), 0, self.train_accuracy_data['x'][-1], color='green')
		ax[1].set_title("Accuracy")
		ax[1].legend(('Train', 'Test', str(kwargs['hline']) if kwargs['hline'] else None), loc='lower right')

		if kwargs['title']:
			plt.suptitle(kwargs['title'])

		plt.show()

