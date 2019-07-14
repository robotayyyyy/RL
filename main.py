#Locate current script dir, then add to sys path
import sys
#import os,inspect
#current_dir = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ) )[0]))
current_dir = 'D:/GitHub/RL/'#manual set
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


    
import random 
random.seed(42)

#import param

import state
import reward
import param
import algo
import report 

#s = state.CartPoleRBFState()
s = state.CartPoleSimpleState()

r = reward.CartPoleReward().rewardFunction3V3
#r = reward.CartPoleReward().rewardFunction3
#r = reward.CartPoleReward().rewardFunctionMinus11
#r = reward.CartPoleReward().rewardFunction01 
#r = reward.CartPoleReward().rewardFunctionMinus10

p = param.CartPoleParam()
a = algo.RL(s,r,p)

# #----------------------
# import numpy as np
# p.maxTimeStep = 200
# a = algo.RL(s,r,p)
# a._RL__explorationRate = 0
# 
# w = np.matrix(([[0],[0],[1],[1],[0],[0],[-1],[0]]))*-1/np.linalg.norm(w)
# 
# for i in range(10):
#     ddd = a.renderPolicy(w)
#     print(len(ddd['samples']))
#     newW2 = a.LSTDQLamda( ddd['samples'], w, 0 )
#     w = newW2
# #----------------------


a.expName = 'reward3_newTest'
a.execute()


for i in range(5):
    a = algo.RL(s,r,p,sn = str(i))
    a.execute()
    
    
    
    
    
    
    

expList = report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_500 TS_6000 LD_0.2')

report.report(expList,prefix = 's100')

sam = a.collectSamples(s.dummyWeight)

import copy
sam2 = copy.deepcopy(sam)



# ss = s.dummyWeight
# ss[0:4] = -1
# ss[4:8] = 1
# 
# ss[3] = 1
# ss[4] = -1
# a.renderPolicy(ss)

s = state.FSimpleState()
r = reward.CartPoleReward().dummyReward
p = param.FParam()
a = algo.RL(s,r,p,env = 'a')

a.execute()

a.collectSamples(s.dummyWeight)

for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.2,sn = str(i))
    a.execute()

for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.0,sn = str(i))
    a.execute()



expList = []
expList.append('LSTDQLamda rewardFunction2 D_8 ITE_6 EP_1 TS_6000 LD_0')
expList.append('LSTDQLamda rewardFunction2 D_8 ITE_7 EP_1 TS_6000 LD_0')
expList.append('LSTDQLamda rewardFunction2 D_8 ITE_8 EP_1 TS_6000 LD_0')
expList.append('LSTDQLamda rewardFunction2 D_8 ITE_9 EP_1 TS_6000 LD_0')

expList = report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_100 TS_6000 LD_0.2')
expList = report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_500 TS_6000 LD_0.2')

report.report(expList,prefix = 's100')
report.report(expList,prefix = 's500')

report.report(report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_500 TS_6000 LD_0.2'), prefix = "s500")
report.report(report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_100 TS_6000 LD_0.2'), prefix = "s100",cof=False)
report.report(report.getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_20 TS_6000 LD_0.2'), prefix = "s20")


report.report(report.getExpSeriesName('reward3V3'), prefix = "test")


s = state.CartPoleRBFState()
r = reward.CartPoleReward().rewardFunction2
p = param.CartPoleParam()



for i in range(10):
    a = algo.RL(s,r,p,timestep = 6000, episode = 20,iteration = 200,lambdaV = 0.2,sn = str(i))
    a.execute()


#s = state.CartPoleRBFState()

#-----------------------------------
# init chart
expList = []
expList.append('init sample b1 (reward -1 fx2) NONFIX 6k')
expList.append('init sample b2 (reward -1 fx2) NONFIX 6k')
expList.append('init sample b2 b1 (reward -1 fx2) NONFIX 6k')
label = ['$s_1$','$s_2$','$s_3$']
report.report(expList, prefix = "expInit",label = label,cof = False,box = False)


#-----------------------------------
# reward chart
expList = []
expList.append('LSTDQLamda rewardFunctionMinus11 D_8 ITE_200 EP_100 TS_6000 LD_0')
expList.append('LSTDQLamda rewardFunction01 D_8 ITE_200 EP_100 TS_6000 LD_0')
expList.append('reward -1 fx2 Lamda 0.0 6k')
#expList.append(report.getExpSeriesName('LSTDQLamda rewardFunction3V2 D_8 ITE_200 EP_100 TS_6000 LD_0/')[2])
#expList.append('reward4V3')
expList.append('reward1')
expList.append(report.getExpSeriesName('LSTDQLamda rewardFunction5 D_8 ITE_200 EP_100 TS_6000 LD_0/')[0])
label = ['$R_1$','$R_2$','$R_3$','$R_4$','$R_5$']
report.report(expList, prefix = "expReward",label = label,cof = False,box = False)

#-----------------------------------
# lambda chart
expList = ["reward -1 fx2 Lamda 0.0 6k"]
expList.append("reward -1 fx2 Lamda 0.2 6k")
expList.append("reward -1 fx2 Lamda 0.4 6k")
expList.append("reward -1 fx2 Lamda 0.6 6k")
expList.append("reward -1 fx2 Lamda 0.8 6k")
expList.append("reward -1 fx2 Lamda 1.0 6k")
label = ['λ 0.0','λ 0.2','λ 0.4','λ 0.6','λ 0.8','λ 1.0']
report.report(expList, prefix = "expLambda",label = label, cof = False,box = False)

expList = []
expList.append('reward3V3')
label = ['$R_6$']
report.report(expList, prefix = "3v3",label = label,cof=False,box=False)
