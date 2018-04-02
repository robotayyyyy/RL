import gym
import fileRecord
import numpy as np
from collections import namedtuple
mySample = namedtuple("mySample", "state action reward nextState isAbsorb")

class RL:
    def __init__(self, state, rewardF, param, **args):
        p = param#param.CartPoleParam()

        self.__basis = state.basis
        self.__D = state.dimension
        self.__reward = rewardF
        
        #default
        self.__envInput = 'CartPole-v0'
        self.__algo = self.LSTDQLamda
        self.__possibleAction = p.possibleAction
        self.__timestep = p.maxTimeStep
        self.__episode = p.maxEpisode
        self.__iteration = p.maxIteration
        self.__explorationRate = p.explorationRate
        self.__distanceThreshold = p.distanceThreshold
        self.__discountFactor = p.discountFactor
        self.__p = p
        
        self.__lambdaV = 0
        self.__fix = False
        self.__initSample = False
        self.__sn = '' #sn is serial number for each exp, none is default
        
        self.validateInput(args)
        
        self.__env = gym.make(self.__envInput)
        self.__env._max_episode_steps = 50000 #set max of timestep as big number

#        print(self.expName)
        
    def validateInput(self,args):
        #check is exist
        if 'env' in args:
            self.__envInput = args['env']
        if 'algo' in args:
            self.__algo = args['algo']
        if 'possibleAction' in args:
            self.__possibleAction = args['possibleAction']
        if 'timestep' in args:
            self.__timestep = args['timestep']
        if 'episode' in args:
            self.__episode = args['episode']
        if 'iteration' in args:
            self.__iteration = args['iteration']
        if 'explorationRate' in args:
            self.__explorationRate = args['explorationRate']
        if 'distanceThreshold' in args:
            self.__distanceThreshold = args['distanceThreshold']  
        if 'discountFactor' in args:
            self.__discountFactor = args['discountFactor']  
            
        if 'lambdaV' in args:
            self.__lambdaV = args['lambdaV']      
        if 'fix' in args:
            self.__fix = args['fix']      
        if 'initSample' in args:
            self.__initSample = args['initSample']  
        if 'sn' in args:
            self.__sn = args['sn']  
        
        if self.__sn != '':
            self.expName = " ".join( (self.__algo.__name__,self.__reward.__name__,'D_'+str(self.__D),'ITE_'+str(self.__iteration),'EP_'+str(self.__episode),'TS_'+str(self.__timestep),'LD_'+str(self.__lambdaV)+'/',self.__sn) )
        else:    
            self.expName = " ".join( (self.__algo.__name__,self.__reward.__name__,'D_'+str(self.__D),'ITE_'+str(self.__iteration),'EP_'+str(self.__episode),'TS_'+str(self.__timestep),'LD_'+str(self.__lambdaV)) )
            
        if 'expName' in args:
            self.expName = args['expName']       
        
    def policyFunction(self,policyWeight, state):#get the action that maximize utility(value function) for given state

        if np.random.random() < self.__explorationRate:#random an action
            selectedAction = not np.random.randint(self.__possibleAction) #random range [0,possibleAction)
        else:#get the best action
            temp = [float(np.transpose(policyWeight) * self.__basis(state,a)) for a in range(self.__possibleAction)]
            selectedAction = np.argmax(temp)
            
        return(selectedAction)
            
    def collectSamples(self,policyWeight):
        samples = []
        accReward = []
        cof = []
        for i in range(self.__episode):
            Time = True
            state = self.__env.reset() #reset simulation to start position
            for j in range(self.__timestep):
                action = self.policyFunction(policyWeight, state)
                #action = int(np.round( random.random())) #pure random
                nextState, reward, isAbsorb, info = self.__env.step(action) #do action
                
                reward = self.__reward(nextState,isAbsorb)
                
                #record
                samples.append( mySample(state,action,reward,nextState,isAbsorb) )
                accReward.append(reward)
                state = nextState
      
                if isAbsorb: #the game end befor max timestep reached
                    Time = False
                    #add cause of failure here
                    cof.append( self.__p.causeOfFailure(nextState) )
                    #then stop collecting sample
                    break
            if Time:
                cof.append( self.__p.causeOfFailure(nextState) )
        return(  {'samples':samples,'avgReward':np.average(accReward), 'cof':self.__p.causeOfFailureSummary(cof) }  )
        
    def renderPolicy(self,policyWeight ):
        samples = []
        sumreward = []
        state = self.__env.reset() #reset simulation to start position
        
        for j in range(self.__timestep):
            self.__env.render()
            action = self.policyFunction(policyWeight, state)
            #action = int(np.round( random.random())) #pure random policy
            nextState, reward, isAbsorb, info = self.__env.step(action) #do action
            
            reward = self.__reward(nextState,isAbsorb)
            
            #record
            samples.append( mySample(state,action,reward,nextState,isAbsorb) )
            sumreward.append(reward)
            state = nextState
                     
            if isAbsorb: #the game end befor max timestep reached
                break
    
        
        print('timestep : ',j+1)
        input('press any key to exit : ')
        self.__env.render(close=True)
        return(  {'samples':samples,'avgReward':np.average(sumreward)}  )
    
    def execute(self):
        self.LSPI(self.__algo)
        
    def LSPI(self,algo,**args):
        initSample = self.__initSample
        fix = self.__fix
        lambdaV = self.__lambdaV
        
        newSample = False
        allPolicyWeight = []
        allMeanTimestep = []
        allDistance = []
        allMeanReward = []
        
        policyWeight = np.matrix(np.zeros((self.__D,1))) #initialize policy as zero vector
        
        if initSample != False: #init sample exist
            samples = initSample
        else: #no init sample
            samples = self.collectSamples(policyWeight) 
        
        distance = np.math.inf #initialize distance as inf
        allPolicyWeight.append(policyWeight)
        allMeanTimestep.append(len(samples['samples'])/self.__episode)
        allMeanReward.append(samples['avgReward'])
        allDistance.append(np.nan)
        cof = samples['cof']
        iteration = 0
        while iteration < self.__iteration and distance > self.__distanceThreshold:
            if not fix and newSample != False: #in case not fix sample and there are newSample
                samples = newSample
            
            print("input---------",len(samples['samples'])/self.__episode,"--------------")
            
            #policyWeight = LSQ(samples['samples'], policyWeight)
            newPolicyWeight = algo(samples['samples'], policyWeight,lambdaV)
            
            distance = np.linalg.norm(newPolicyWeight-allPolicyWeight[iteration])
            newSample = self.collectSamples(newPolicyWeight) #for measure performance
            policyWeight = newPolicyWeight
            print(iteration,"average time steps :",len(newSample['samples'])/self.__episode,"distance",distance)
            
            iteration +=1
            
            #record-----------------------------
            allPolicyWeight.append(newPolicyWeight)
            allMeanTimestep.append(len(newSample['samples'])/self.__episode)
            allDistance.append(distance)
            allMeanReward.append(newSample['avgReward'])
            for key in cof:
                cof[key] += newSample['cof'][key]
        fileRecord.ExpSaveLoad().saveExp(self.expName,[0,1,1,1,1],allPolicyWeight=allPolicyWeight, allMeanTimestep=allMeanTimestep, allDistance=allDistance, allMeanReward=allMeanReward,cof=cof)
    
    def LSTDQLamda(self,samples, policyWeight,lambdaV):
        A = np.matrix(np.zeros((self.__D,self.__D)))
        B = np.matrix(np.zeros((self.__D,1)))
        n = len(samples)
        z = np.zeros((self.__D,1))
        for i in range(n): #start from i=0 to i=n-1
            phi = self.__basis(samples[i].state,samples[i].action)
            z = (lambdaV)*z + phi
            if samples[i].isAbsorb != True: #check if next state is not absorb state
                nextAction = self.policyFunction(policyWeight, samples[i].nextState)
                nextPhi = self.__basis(samples[i].nextState, nextAction)
            else:
                nextPhi = np.zeros((self.__D,1))
            
            A = A + z * np.transpose(phi - self.__discountFactor*nextPhi)
            B = B + z * samples[i].reward
        
        return(np.linalg.pinv(A)*B)    