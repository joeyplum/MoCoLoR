# %% Load data (slow)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv(r"data/floret-osamri-20230901/motion_flow.txt", sep='\t', header=None)
data = np.array(data)
print(data.shape)


# %% Plot results

plt.figure()
start = int(1.5e7)
plt.plot(data[start:,1])

print(np.sum(data[start:,1] > 4.8))

# %% Extract the useful bits (manually)



