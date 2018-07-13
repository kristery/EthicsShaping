from drive import Driving
import numpy as np
dr = Driving()

for cnt in range(1):
    state = dr.reset()
    while True:
        action = np.random.randint(3)
        state, reward, done = dr.step(action)
        print(state, reward)
        if done:
            break
