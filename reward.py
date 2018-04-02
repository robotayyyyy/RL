import numpy as np
import param

class CartPoleReward:
    def __init__(self):
        self.__param = param.CartPoleParam()
        
    def rewardFunction01(self,nextState, isAbsorb):
        
        if isAbsorb:
            reward = 0
        else:
            reward = 1
        return(reward)
    
    def rewardFunctionMinus10(self,nextState, isAbsorb):
        
        if isAbsorb:
            reward = -1
        else:
            reward = 0
        return(reward)
    
    def rewardFunctionMinus11(self,nextState, isAbsorb):
        
        if isAbsorb:
            reward = -1
        else:
            reward = 1
        return(reward)
    
    def rewardFunction1(self,nextState,isAbsorb):
        
        if isAbsorb:
            reward = -1
        else:
            rewX = 1-np.linalg.norm(nextState[0]/self.__param.maxX)
            rewXD = 1-np.linalg.norm(nextState[1]/self.__param.maxX)
            rewA = 1-np.linalg.norm(nextState[2]/self.__param.maxDegree)
            #rewAD = 1-np.linalg.norm(nextState[3]/2.5)
            rewAD = np.math.exp(-(nextState[3]*nextState[3]))
            reward = np.min([rewX ,rewA,rewAD ])
        return(reward)
    
    def rewardFunction1V2(self,nextState,isAbsorb):
        
    
        rewX = np.linalg.norm(nextState[0]/self.__param.maxX)
        rewXD = np.linalg.norm(nextState[1]/self.__param.maxX)
        rewA = np.linalg.norm(nextState[2]/self.__param.maxDegree)
        #rewAD = 1-np.linalg.norm(nextState[3]/2.5)
        rewAD =  np.linalg.norm(nextState[3])
        reward = -rewX -rewA -rewAD
        return(reward)
        
    def rewardFunction2(self,nextState,isAbsorb):#best
        
        if isAbsorb:
            reward = -1
        else:
            Angle = nextState[2]
            AngleV = nextState[3]
            
            if Angle <0 :
                if AngleV <0:
                    reward = 0
                else:
                    reward = 1
            else :
                if AngleV <0:
                    reward = 1
                else:
                    reward = 0
        return(reward)
    
    def rewardFunction3(self,nextState,isAbsorb):
        
        if isAbsorb:
            reward = -1
        else:
            x = np.linalg.norm(nextState[0]/self.__param.maxX)/2
            Angle = nextState[2]
            AngleV = nextState[3]
            
            if Angle <0 :
                if AngleV <0:
                    reward = 0
                else:
                    reward = 1
            else :
                if AngleV <0:
                    reward = 1
                else:
                    reward = 0
            reward = reward-x
        return(reward)
    
    def rewardFunction4(self,nextState,isAbsorb):
        positionX = nextState[0]
        Angle = nextState[2]
        reward = -np.math.pow(positionX,2) -np.math.pow(Angle,8)
        return(reward)
    
    def rewardFunction4V2(self,nextState,isAbsorb):
        positionX = nextState[0]/maxX
        Angle = nextState[2]/maxDegree
        # positionX = nextState[0]
        # Angle = nextState[2]
        reward =  np.min([-np.math.pow(positionX,128), -np.math.pow(Angle,16) ]) 
        return(reward)
        
    def rewardFunction5(self,nextState,isAbsorb):
        positionX = nextState[0]
        Angle = nextState[2]
        reward = -np.math.pow(Angle,2)
        return(reward)
    