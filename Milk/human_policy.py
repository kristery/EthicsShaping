from milk import FindMilk
import numpy as np
import pickle

np.random.seed(1234)

fm = FindMilk()

state = fm.reset()
rewards = []
trajectory = {}

for cnt in range(100):
    while True:
        ethical_state = state[2:]
        ### human policy ###
        probs = []
        for pos in ethical_state:
            if pos > 0: probs.append(20)
            elif pos < 0: probs.append(1)
            else: probs.append(5)
        # normalization
        total = sum(probs)
        probs = [p/total for p in probs]
        action = np.random.choice(4, 1, p=probs)[0]
        
        try:
            trajectory[(ethical_state, action)] += 1
        except:
            trajectory[(ethical_state, action)] = 1

        state, reward, done = fm.step(action)
        rewards.append(reward)
        if done:
            break

with open('hpolicy_milk.pkl', 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)
