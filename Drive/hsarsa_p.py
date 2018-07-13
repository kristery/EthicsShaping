import argparse
import pickle
from drive import Driving
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='ethical agent')
parser.add_argument('--ethical', action='store_true',
                    help='indicate whether learn from human trajectory or not')
parser.add_argument('--c', type=float, default=0.6,
                    help='a parameter to determine the human policy (default: 0.6)')
parser.add_argument('--cn', type=float, default=2,
                    help='scale of the additioal punishment (default: 2)')
parser.add_argument('--cp', type=float, default=2,
                    help='scale of the additional reward (default: 2)')
parser.add_argument('--taun', type=float, default=0.2,
                    help='threshold to determine negatively ethical behavior (default: 0.2)')
parser.add_argument('--taup', type=float, default=0.55,
                    help='threshold to determine positvely ethical behavior (default: 0.55)')
parser.add_argument('--temp', type=float, default=0.7,
                    help='the temperature parameter for Q learning policy (default: 0.7)')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=1000,
                    help='number of episdoes (default: 1000)')
parser.add_argument('--verbose', action='store_true',
                    help='show log')
parser.add_argument('--record_beg', type=int, default=600,
                    help='begin to record trajectories')
args = parser.parse_args()

actions = range(3)

np.random.seed(args.seed)
Q = {}

dr = Driving(ishuman_p=True)
trajectory = {}
episode_rewards = []
collisions = []
cat_hits = []

def kl_div(p1, p2):
    total = 0.
    for idx in range(len(p1)):
        total += -p1[idx]*np.log(p2[idx]/p1[idx])
    return total

for cnt in range(args.num_episodes):
    state = dr.reset()
    #state = state[:2]
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
        
        action = np.random.choice(3, 1, p=probs)[0]
        if args.verbose: print(probs, state, action)

        ethical_state = (state[2], state[4], state[6])
        if cnt > args.record_beg:
            try:
                trajectory[(ethical_state, action)] += 1
            except:
                trajectory[(ethical_state, action)] = 1


        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = dr.step(action)

        prev_pair = (state, action)
        prev_reward = reward
        rewards += reward
        if done:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward - Q[prev_pair])
            break
        state = next_state
    collision, cat_hit = dr.log()        
    collisions.append(collision)
    cat_hits.append(cat_hit)
    episode_rewards.append(rewards)
    
    if cnt % 100 == 0:
        print('episode: {}, frame: {}, total reward: {}'.format(cnt, frame, rewards))

label = 'human_p'

df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{:.2f}_{:.2f}_{}_steps.csv'.format(args.temp, args.gamma, label), index=False)
dfp = pd.DataFrame(np.array(collisions))
dfp.to_csv('./record/{:.2f}_{:.2f}_{}_collisions.csv'.format(args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(cat_hits))
dfn.to_csv('./record/{:.2f}_{:.2f}_{}_cat_hits.csv'.format(args.cn, args.taun, label), index=False)

with open('hpolicy_drive_p.pkl', 'wb') as f:
    pickle.dump(trajectory, f, pickle.HIGHEST_PROTOCOL)
