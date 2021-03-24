import neurolab as neuro
import numpy as np
import matplotlib.pyplot as plt

# >40 - 2, 31...40 - 1, <-30 - 0
# wysoki - 1, sredni - 2, niski - 0
# tak-1, nie - 0
train_in = [[0, 1, 0, 0],
          [0, 1, 0, 1],
          [1, 1, 0, 0],
          [2, 1, 0, 0],
          [2, 2, 0, 0],
          [2, 0, 1, 1],
          [1, 0, 1, 1],
          [0, 2, 0, 0],
          [0, 0, 0, 0],
          [2, 2, 1, 0]]

train_out = [[0], [0], [1], [1], [1], [0], [1], [0], [1], [1]]

test_in = [[0, 2, 1, 1],
           [1, 2, 0, 1],
           [1, 1, 1, 0],
           [2, 2, 0, 0]]

test_out = [[1], [1], [1], [0]]

perceptron = neuro.net.newp([[0, 2], [0, 2], [0, 1], [0, 1]], 1)

error = perceptron.train(train_in, train_out, epochs=100, show=1, lr=0.01)

result_out = perceptron.sim(test_in)

func = neuro.error.SSE()
test_error = func(test_out, result_out)

print('Classification:', result_out[:, 0])
print('Learning error: ', error)
print('Classification error: ', test_error)

plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Train error')
plt.grid()
plt.show()