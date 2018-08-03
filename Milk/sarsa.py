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
parser.add_argument('--cn', type=float, default=2,
                    help='scale of the additioal punishment (default: 2)')
parser.add_argument('--cp', type=float, default=2,
                    help='scale of the additional reward (default: 2)')
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
parser.add_argument('--num_episodes', type=int, default=1000,
                    help='number of episdoes (default: 1000)')
args = parser.parse_args()

actions = range(4)
if args.ethical:
    hpolicy = {}
    with open('hpolicy_milk.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    for key in trajectory:
        if key[0] not in hpolicy:
            probs = []
            count = []
            for action in actions:
                try:
                    count.append(trajectory[(key[0], action)])
                except:
                    count.append(0)
            total_cnt = sum(count)
            probs = [args.c**count[action]*(1-args.c)**(total_cnt-count[action]) for action in actions]
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            hpolicy[key[0]] = probs


np.random.seed(args.seed)
Q = {}

fm = FindMilk()
episode_rewards = []
poss = []
negs = []

def kl_div(p1, p2):
    total = 0.
    for idx in range(len(p1)):
        total += -p1[idx]*np.log(p2[idx]/p1[idx])
    return total

for cnt in range(args.num_episodes):
    state = fm.reset()
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
        
        action = np.random.choice(4, 1, p=probs)[0]
        if prev_pair is not None:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward + args.gamma * Q[(state, action)] - Q[prev_pair])
        next_state, reward, done = fm.step(action)

        if args.ethical:
            if state[2:] in hpolicy:
                hprobs = hpolicy[state[2:]]
                if hprobs[action] < args.taun and hprobs[action] < probs[action]:
                    H = -args.cn * kl_div(probs, hprobs)
                elif hprobs[action] > args.taup and hprobs[action] > probs[action]:
                    H = args.cp * kl_div(probs, hprobs)
                else:
                    H = 0
            reward += H

        prev_pair = (state, action)
        prev_reward = reward
        rewards += reward
        if done:
            Q[prev_pair] = Q[prev_pair] + args.lr * (prev_reward - Q[prev_pair])
            break
        state = next_state
    neg_passed, pos_passed = fm.log()        
    poss.append(pos_passed)
    negs.append(neg_passed)
    episode_rewards.append(frame)
    if cnt % 100 == 0:
        print('episode: {}, frame: {}, total reward: {}, neg_passed: {}, pos_passed: {}'.format(cnt, frame, rewards, neg_passed, pos_passed))

if args.ethical:
    label = 'ethical'
else:
    label = 'normal'

df = pd.DataFrame(np.array(episode_rewards))
df.to_csv('./record/{}_steps.csv'.format(label), index=False)
dfp = pd.DataFrame(np.array(poss))
dfp.to_csv('./record/{:.2f}_{:.2f}_{}_pos_passed.csv'.format(args.cp, args.taup, label), index=False)
dfn = pd.DataFrame(np.array(negs))
dfn.to_csv('./record/{:.2f}_{:.2f}_{}_neg_passed.csv'.format(args.cn, args.taun, label), index=False)
