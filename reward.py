import numpy as np
import param




class CartPoleReward:
    def __init__(self):
        self.__param = param.CartPoleParam()
        
    def dummyReward(self,nextState, reward, isAbsorb, info):
        return reward
    
    def rewardFunction01(self, nextState, reward, isAbsorb, info):
        
        if isAbsorb:
            reward = 0
        else:
            reward = 1
        return(reward)
    
    def rewardFunctionMinus10(self, nextState, reward, isAbsorb, info):
        
        if isAbsorb:
            reward = -1
        else:
            reward = 0
        return(reward)
    
    def rewardFunctionMinus11(self, nextState, reward, isAbsorb, info):
        
        if isAbsorb:
            reward = -1
        else:
            reward = 1
        return(reward)
    
    def rewardFunction1(self,nextState, reward, isAbsorb, info ):
        
        if isAbsorb:
            reward = -1
        else:
            rewX = 1-np.linalg.norm(nextState[0]/self.__param.maxX)
            
            rewA = 1-np.linalg.norm(nextState[2]/self.__param.maxDegree)

            
            reward = np.min([rewX ,rewA ])
        return(reward)
    
    def rewardFunction1V2(self,nextState, reward, isAbsorb, info):
        
    
        rewX = np.linalg.norm(nextState[0]/self.__param.maxX)
        rewXD = np.linalg.norm(nextState[1]/self.__param.maxX)
        rewA = np.linalg.norm(nextState[2]/self.__param.maxDegree)
        #rewAD = 1-np.linalg.norm(nextState[3]/2.5)
        rewAD =  np.linalg.norm(nextState[3])
        reward = -rewX -rewA -rewAD
        return(reward)
        
    def rewardFunction22(self, nextState, reward, isAbsorb, info):#best
        
        if isAbsorb:
            reward = -1
        else:
            Angle = nextState[2]
            AngleV = nextState[3]
            
            if Angle <0 :
                if AngleV <0:
                    reward = -0.1
                else:
                    reward = 1
            else :
                if AngleV <0:
                    reward = 1
                else:
                    reward = -0.1
        return(reward)
        
    def rewardFunction2(self, nextState, reward, isAbsorb, info):#best
        
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
    
    def rewardFunction3(self, nextState, reward, isAbsorb, info):
        x = np.linalg.norm(nextState[0]/self.__param.maxX)*0.1
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
            reward = reward-x
        return(reward)
    
    def rewardFunction3V2(self, nextState, reward, isAbsorb, info):
        x = self.gaussian(nextState[0], mu=0, sig=1)*0.1
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
            reward = reward+x
        return(reward)
    
    def rewardFunction3V3(self, nextState, reward, isAbsorb, info):
        positionX = self.gaussian(np.linalg.norm(nextState[0]), mu=2.4, sig=0.1) 
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
            reward = reward-positionX
        return(reward)
    


    def gaussian(self,x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
    def rewardFunction4(self, nextState, reward, isAbsorb, info):
        if isAbsorb:
            reward = -1
        else:
            positionX = np.linalg.norm(nextState[0]/self.__param.maxX)
            Angle = np.linalg.norm(nextState[2]/self.__param.maxDegree)
            reward = -0.5*Angle -0.5*positionX
        return(reward)
        
    
    
    def rewardFunction4V2(self, nextState, reward, isAbsorb, info):
        if isAbsorb:
            reward = -1
        else:
            positionX = self.gaussian(nextState[0], mu=0, sig=0.1) 
            Angle = self.gaussian(nextState[2], mu=0, sig=0.5)  
            reward = 0.5*Angle + 0.5*positionX
        return(reward)
        
    def rewardFunction5(self,nextState, reward, isAbsorb, info):
        if isAbsorb:
            reward = -1
        else:
            positionX = self.gaussian(np.linalg.norm(nextState[0]), mu=2.4, sig=0.01) 
            Angle = self.gaussian(nextState[2], mu=0, sig=0.01)  
            reward = Angle - positionX
        return(reward)
        

    

    