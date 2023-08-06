from first_steps import train_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

learning_rates = np.array([0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]).reshape(6,)
steps = np.array([10, 100, 200, 300, 400, 500, 1000]).reshape(7,)
batch_size = 1000
RMSES = np.linspace(160, 280, len(learning_rates)*len(steps))
Cols = np.linspace(0.0, 1.0, len(learning_rates)*len(steps)).reshape(1, len(learning_rates)*len(steps))
print(Cols)

combinations = [[lr, s] for lr in learning_rates for s in steps]

outcomes = []
for i, combination in enumerate(combinations):
    # outcomes.append([combination, train_model(combination[0], combination[1], batch_size=batch_size)])
    outcomes.append([combination, RMSES[i]])
    print("[{}/{}] Done for learning_rate: {}, steps: {}"
          .format(i+1, len(combinations), combination[0], combination[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for outcome in outcomes:
    x = outcome[0][0]       # Learning rate
    y = outcome[0][1]       # Steps
    z = outcome[1]          # RMSE
    ax.scatter(x, y, z, c=Cols)

ax.set_xlabel("Learning rate")
ax.set_ylabel("Steps")
ax.set_zlabel("RMSE")
plt.show()

print(cm.cmaps_listed)

