import argparse
import pickle
from milk import FindMilk
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--ethical', action='store_true',
                    help='indicate whether learn from human trajectory or not')
parser.add_argument('--c', type=float, default=0.6,
                    help='a parameter to determine the human policy (default: 0.6)')
parser.add_argument('--cn', type=float, default=0.1,
                    help='scale of the additioal punishment (default: 0.1)')
parser.add_argument('--cp', type=float, default=0.1,
                    help='scale of the additional reward (default: 0.1)')
parser.add_argument('--taun', type=float, default=0.2,
                    help='threshold to determine negatively ethical behavior (default: 0.2)')
parser.add_argument('--taup', type=float, default=0.55,
                    help='threshold to determine positvely ethical behavior (default: 0.55)')
parser.add_argument('--temp', type=float, default=1,
                    help='the temperature parameter for Q learning policy (default: 1)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=500,
                    help='number of episdoes (default: 500)')
args = parser.parse_args()

if args.ethical:
    with open('hpolicy_milk.pkl', 'rb') as f:
        trajectory = pickle.load(f)
        
np.random.seed(args.seed)
hpolicy = {}
Q = {}
actions = range(4)

fm = FindMilk()
episode_rewards = []

for cnt in range(args.num_episodes):
    state = fm.reset()
    state = state[:2]
    rewards = 0.
    prev_pair = None
    prev_reward = None
    frame = 0
    
    while True:
        frame += 1
        probs = []
        for action in actions:
            try: 
                probs.append(np.e**(Q[(state, action)]/args.temp))
            except:
                Q[(state, action)] = np.random.randn()
                probs.append(np.e**(Q[(state, action)]/args.temp))

        total = sum(probs)
        probs = [p / total for p in probs]
        
        #print(probs)
        action = np.random.choice(4, 1, p=probs)[0]
        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = fm.step(action)

        prev_pair = (state, action)
        prev_reward = reward
        rewards += reward
        if done:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward - Q[prev_pair])
            break
        state = next_state[:2]
    
    episode_rewards.append(rewards)
    print('episode: {}, frame: {}, total reward: {}'.format(cnt + 1, frame, rewards))
for key in Q:
    if key[0][0] > 6 and key[0][1] > 6:
        print('key: {}, value: {}'.format(key, Q[key]))

df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./rewards.csv', index=False)
