import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from GRU_RNN import GRU_RNN     # import RNN class


T = 50
epochs = 30
k1 = 10             # NOTE k1 should divide T
k2 = 13
weight_file = None  # None or 'weights.json'


def scale_0_1(x, min_x, max_x):
    if abs(max_x - min_x) > 1e-6:   # prevent dividing by 0
        return (x - min_x)/(max_x - min_x)
    else:
        print(min_x, max_x)
        print('Data hardly varies.')
        return x


# Load .csv file and sort it by date
df = pd.read_csv("data/stock_market_data-AAL.csv")
df = df.sort_values('Date')


# Choose price type
data = df['Close'].to_numpy() # options: 'Low', 'High', 'Close', 'Open'
# data = (df['High'].to_numpy() + df['Low].to_numpy()) / 2.0 # mid price


# The first 80% of the data is the training data, the remaining 20% is the test data
length = len(data)
train_length = int(length * 0.8)

train_data = data[:train_length]
test_data = data[train_length:]


# Scale the data to be between 0 and 1 (only depending on the training data set!)
min_x = min(train_data)
max_x = max(train_data)
train_data = scale_0_1(train_data, min_x, max_x)
test_data = scale_0_1(test_data, min_x, max_x)

XY_train = []
XY_test = []
# Bring the data in shape
for i in range(train_length-2*T+1):
    xy_tuple = (train_data[i:i+T], train_data[i+T:i+2*T])
    XY_train.append(xy_tuple)
for i in range(len(test_data)-2*T+1):
    xy_tuple = (test_data[i:i+T], test_data[i+T:i+2*T])
    XY_test.append(xy_tuple)


# Training
my_rnn = GRU_RNN(T, k1=k1, k2=k2, weight_file=weight_file)
losses = my_rnn.train_network(training_data=XY_train, epochs=epochs, testing_data=XY_test)


# Plot epochs vs losses
plt.figure()
plt.title('Training progress')
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.plot(range(epochs), losses)
plt.show()
plt.close()


# Plot some predicted sequences against the real ones (we predict sequences of 100 points, which is longer than the training sequences)
test_sequence = test_data[100:200]
real_continuation = test_data[200:300]
prediction = my_rnn.predict(test_sequence)

plt.figure()
plt.title('Prediction of American Airlines Stock Price (1)')
plt.xlabel('Days')
plt.ylabel('Stock price')
plt.plot(range(200), np.concatenate((test_sequence, real_continuation)), label='real')
plt.plot(range(100, 200), prediction, label='prediction')
plt.legend()
plt.show()
plt.close()

test_sequence = test_data[300:400]
real_continuation = test_data[400:500]
prediction = my_rnn.predict(test_sequence)

plt.figure()
plt.title('Prediction of American Airlines Stock Price (2)')
plt.xlabel('Days')
plt.ylabel('Stock price')
plt.plot(range(200), np.concatenate((test_sequence, real_continuation)), label='real')
plt.plot(range(100, 200), prediction, label='prediction')
plt.legend()
plt.show()
plt.close()