
acc =    [1,20,30,40,45,60,65,75, 85]
inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

import matplotlib.pyplot as plt


plt.plot(acc,inputs)
plt.xlabel("# inputs")
plt.ylabel("Accuracy [%] on 10 randomly chosen trials")
plt.grid()
plt.show()
