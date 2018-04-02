#Locate current script dir, then add to sys path
import sys
#import os,inspect
#current_dir = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ) )[0]))
current_dir = 'D:\\GitHub\RL'#manual set
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


#import random 
#random.seed(42)

#import param

import state
import reward
import param
import algo
import report 

#s = state.CartPoleRBFState()
s = state.CartPoleSimpleState()
r = reward.CartPoleReward().rewardFunction2
p = param.CartPoleParam()



for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.2,sn = str(i))
    a.execute()

for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.0,sn = str(i))
    a.execute()



explist = []
explist.append('LSTDQLamda rewardFunction2 D_8 ITE_6 EP_1 TS_6000 LD_0')
explist.append('LSTDQLamda rewardFunction2 D_8 ITE_7 EP_1 TS_6000 LD_0')
explist.append('LSTDQLamda rewardFunction2 D_8 ITE_8 EP_1 TS_6000 LD_0')
explist.append('LSTDQLamda rewardFunction2 D_8 ITE_9 EP_1 TS_6000 LD_0')

report.report(explist,prefix = 'test')
report.report(report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_20 TS_6000 LD_0'), prefix = "--")


s = state.CartPoleRBFState()
r = reward.CartPoleReward().rewardFunction2
p = param.CartPoleParam()



for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.2,sn = str(i))
    a.execute()



