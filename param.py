import numpy as np
import os

class FParam:
    def __init__(self):
        self.possibleAction = 2 #amount of possible action
        #constant for algorithm
        self.discountFactor = 0.9 # set abitary 0 to 1
        self.explorationRate = 0.01
        self.maxTimeStep = 50 #timestep for each EP if absorb state no found
        self.maxEpisode = 50 #EP for each
        self.maxIteration = 5 #max itteration of LSPI if distance not less than predefine threshold
        self.distanceThreshold = .1 #threshold to judge that policy is convert
        
        self.cofLabel = ['Out of money','No data remain','Time']# f***ing important
        self.saveDir = 'D:/save/'
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
            
    def causeOfFailure(self,state):#check cause of failure in case absorbing state
        #state of CartPole-v0 contain 4 attribute
        if state[0] <= 0:
            return self.cofLabel[0]
        else: # max of time step reach
            return self.cofLabel[2]
    def causeOfFailureSummary(self,cof):
        return {self.cofLabel[0] : [cof.count(self.cofLabel[0])/self.maxEpisode*100] ,self.cofLabel[1] : [cof.count(self.cofLabel[1])/self.maxEpisode*100] ,self.cofLabel[2] : [cof.count(self.cofLabel[2])/self.maxEpisode*100] }
        #return {self.cofLabel[0] : [cof.count(self.cofLabel[0])] ,self.cofLabel[1] : [cof.count(self.cofLabel[1])] ,self.cofLabel[2] : [cof.count(self.cofLabel[2])] }
        
        
class CartPoleParam:
    def __init__(self):
        #constant for pendulumn problem
        self.maxDegree = 12*2*np.math.pi/360 #max possible pole degree(radian)
        self.maxX = 2.4 #max distance in X axist
        self.possibleAction = 2 #amount of possible action
        
        #constant for algorithm
        self.discountFactor = 0.9 # set abitary 0 to 1
        self.explorationRate = 0.01
        self.maxTimeStep = 6000 #timestep for each EP if absorb state no found
        self.maxEpisode = 100 #EP for each
        self.maxIteration = 200 #max itteration of LSPI if distance not less than predefine threshold
        self.distanceThreshold = -0.1 #threshold to judge that policy is convert
        
        self.cofLabel = ['Out of track','Pole down','Time']# f***ing important
        self.saveDir = 'D:/save/'
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
            
    def causeOfFailure(self,state):#check cause of failure in case absorbing state
        #state of CartPole-v0 contain 4 attribute
        if state[0] > self.maxX or state[0] < -self.maxX:
            return self.cofLabel[0]
        elif state[2] > self.maxDegree or state[2] < -self.maxDegree:
            return self.cofLabel[1]
        else: # max of time step reach
            return self.cofLabel[2]
    def causeOfFailureSummary(self,cof):
        return {self.cofLabel[0] : [cof.count(self.cofLabel[0])/self.maxEpisode*100] ,self.cofLabel[1] : [cof.count(self.cofLabel[1])/self.maxEpisode*100] ,self.cofLabel[2] : [cof.count(self.cofLabel[2])/self.maxEpisode*100] }
        #return {self.cofLabel[0] : [cof.count(self.cofLabel[0])] ,self.cofLabel[1] : [cof.count(self.cofLabel[1])] ,self.cofLabel[2] : [cof.count(self.cofLabel[2])] }