import pandas as df
import numpy as np
from datetime import datetime

current_dir = 'D:/GitHub/RL/'#manual set
class fenv:
    def __init__(self):

        self.data = df.read_csv(current_dir+'fenv/set50_day.txt',header=None )
        self.data.columns  = ['date','open','high','low','close','volume']
        self.data.date = [datetime.strptime(self.data.date[i], '%Y.%m.%d %H:%M') for i in range(len(self.data.date))]
        #for directive (%-) https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        self.n = len(self.data.date)
        self.free = 10

    def reset(self):
        self.balance = 50
        self.balanceThreshold = 0
        self.index = np.random.randint(0,self.n/2)
        return self.getState(self.index)
    
    def getState(self,index):
        if index < self.n:
            return [self.balance,self.data.open[index],self.data.high[index],self.data.low[index],self.data.close[index],self.data.volume[index]]
        else:
            return np.nan
    
    def getReward(self,action):
        return (self.data.close[self.index+1] - self.data.close[self.index])*action -self.free
        
        
    def step(self,action):
        info = ''
        isAbsorb = False
        if self.index >= self.n-2:
            info = 'no more data'
            
        reward = self.getReward(action*2-1)
        self.balance += reward
        if self.balance <= self.balanceThreshold:
            isAbsorb = True
        self.index += 1
        nextState = self.getState(self.index)
        
        return ([nextState, reward, isAbsorb, info])
        
            
        