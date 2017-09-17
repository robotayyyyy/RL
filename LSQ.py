import pickle
import os#, errno
import gym
import numpy as np
import random 
import itertools
from collections import namedtuple
import matplotlib.pyplot as plt
random.seed(42)
env = gym.make('CartPole-v0')

#constant for pendulumn problem
maxDegree = 12*2*np.math.pi/360 #max possible pole degree(radian)
maxX = 2.4 #max distance in X axist
possibleAction = 2 #amount of possible action

c1 = [-maxX, maxX]
c2 = [-1,  1]
c3 = [-maxDegree,-maxDegree/2,maxDegree/2, maxDegree]
c4 = [-1,-0.1,0.1, 1]
D = 8 #dimension of state
#D = (len(c1) * len(c2) * len(c3) * len(c4) +1)*2



#constant for algorithm
discountFactor = 0.9 # set abitary 0 to 1
explorationRate = 0.01
lamda = 0.0

maxTimeStep = 1000 #timestep for each EP if absorb state no found
maxEpisode = 100 #EP for each
maxIteration = 200 #max itteration of LSPI if distance not less than predefine threshold
distanceThreshold = .1 #threshold to judge that policy is convert

#defined....
mySample = namedtuple("mySample", "state action reward nextState isAbsorb")
rootDir = "D:\\exp result\\"

b2 = np.matrix( np.zeros((D,1)) )
b2[0:int(D/2),0] = -1
b2[int(D/2):D,0] = 1
b1 =b2*-1

relativeVectors = []
for w in c1 :
    for x in c2 :
        for y in c3 :
            for z in c4 :
                relativeVectors.append([w,x,y,z])


#function
def basis(state, action): #only support 4D raw state and 0,1 action
    phi = np.zeros((D,1))
    if(action == 0):
        phi[0,0]= state[0]/maxX
        phi[1,0]= state[1]/maxX
        phi[2,0]= state[2]/maxDegree
        phi[3,0]= state[3]/maxDegree
    else:
        phi[4,0]= state[0]/maxX
        phi[5,0]= state[1]/maxX
        phi[6,0]= state[2]/maxDegree
        phi[7,0]= state[3]/maxDegree
    return(phi)


'''
def basis(state,action):
    temp = np.zeros((D,1))
    if(action == 0):
        base = 0
    else:
        base = D/2
    count = int(base)
    
    temp[count,0] = 1
    count +=1
    
    dist = (np.matrix(relativeVectors) - np.matrix(state)).tolist()
    xValue = np.linalg.norm( dist, axis = 1)
    yValue = np.exp( -(xValue * xValue) )
    n=len(yValue)
    temp[count:count + n,0] = yValue
                 
    return(temp)
'''

def policyFunction(policyWeight, state):#get the action that maximize utility(value function) for given state

    for cadidateAction in range(possibleAction):
        temp = float( np.transpose(np.matrix(policyWeight)) * basis(state, cadidateAction) )
        
        if cadidateAction == 0: #for first time
            maxUtility = temp
            selectedAction = 0
            
        if temp > maxUtility: #found better action
            maxUtility = temp
            selectedAction = cadidateAction
    
    if np.random.random() < explorationRate:
        selectedAction = not selectedAction
        
    return(selectedAction)

def rawardFunction(nextState):
    rewX = 1-np.linalg.norm(nextState[0]/maxX)
    rewXD = 1-np.linalg.norm(nextState[1]/maxX)
    rewA = 1-np.linalg.norm(nextState[2]/maxDegree)
    rewAD = 1-np.linalg.norm(nextState[3])
    reward = min(rewX ,rewA,rewAD ) 
    #reward = min(rewX ,rewA ) 
    #reward = (3*rewX + 0*rewXD + 3*rewA + 1*rewAD)/7 
    return(reward)

def rawardFunction2(nextState):
    
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
    
def collectSamples(policyWeight):
    samples = []
    sumreward = []
    for i in range(maxEpisode):
        state = env.reset() #reset simulation to start position
        for j in range(maxTimeStep):
            action = policyFunction(policyWeight, state)
            #action = int(np.round( random.random())) #pure random policy
            nextState, reward, isAbsorb, info = env.step(action) #do action
            
            if reward == 1:
                reward = rawardFunction2(nextState)
            # if reward == 1:
            #     reward = 0
                
            if isAbsorb: #the game end befor max timestep reached
                reward = -1 # 0 or -1
                samples.append( mySample(state,action,reward,nextState,isAbsorb) )
                break

            #record
            samples.append( mySample(state,action,reward,nextState,isAbsorb) )
            sumreward.append(reward)
            state = nextState
    return(  {'samples':samples,'avgReward':np.average(sumreward)}  )

def renderPolicy(policyWeight):
    samples = []
    state = env.reset() #reset simulation to start position
    for j in range(maxTimeStep):
        env.render()
        action = policyFunction(policyWeight, state)
        #action = int(np.round( random.random())) #pure random policy
        nextState, reward, isAbsorb, info = env.step(action) #do action
        
        if reward == 1:
            reward = rawardFunction2(nextState)
                 
        if isAbsorb: #the game end befor max timestep reached
            samples.append( mySample(state,action,reward,nextState,isAbsorb) )
            break

        #record
        samples.append( mySample(state,action,reward,nextState,isAbsorb) )
        state = nextState
    env.close()
    print(j+1)
    return(samples)

def LSQ(samples, policyWeight):
    A = np.matrix(np.zeros((D,D)))
    B = np.matrix(np.zeros((D,1)))
    n = len(samples)
    
    for i in range(n): #start from i=0 to i=n-1
    
        phi = basis(samples[i].state,samples[i].action)
        if samples[i].isAbsorb != True: #check if next state is not absorb state
            nextAction = policyFunction(policyWeight, samples[i].nextState)
            nextPhi = basis(samples[i].nextState, nextAction)
        else:
            nextPhi = np.zeros((D,1))
        
        A = A + phi * np.transpose(phi - discountFactor*nextPhi)
        B = B + phi * samples[i].reward
    
    return(np.linalg.pinv(A)*B)

def LSQLamda(samples, policyWeight,lamda):
    A = np.matrix(np.zeros((D,D)))
    B = np.matrix(np.zeros((D,1)))
    n = len(samples)
    z = np.zeros((D,1))
    for i in range(n): #start from i=0 to i=n-1
        phi = basis(samples[i].state,samples[i].action)
        z = (lamda)*z + phi
        if samples[i].isAbsorb != True: #check if next state is not absorb state
            nextAction = policyFunction(policyWeight, samples[i].nextState)
            nextPhi = basis(samples[i].nextState, nextAction)
        else:
            nextPhi = np.zeros((D,1))
        
        A = A + z * np.transpose(phi - discountFactor*nextPhi)
        B = B + z * samples[i].reward
    
    return(np.linalg.pinv(A)*B)

def printSamples(samples):    
    for i in range(len(samples)):
        print(samples[i].state, samples[i].action, samples[i].reward, samples[i].isAbsorb)

def saveFile(filename,var):

    with open(filename, 'wb') as fp:
        pickle.dump(var, fp)
        
def loadFile(filename):        
    with open (filename, 'rb') as fp:
        return(pickle.load(fp))

def saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward):
    outputDir = rootDir+expName+"\\"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    desc = "State Dimension : " + str(D) + "\nReward : {0,1}" + "\nPolicy evaluate finction : LSQ" 
    desc += "\n--------------------------------------------------------------"
    desc += "\nLamda : "+str(lamda) 
    desc += "\nDiscount factor : "+ str(discountFactor) 
    desc += "\nexploration rate : "+str(explorationRate)
    desc += "\n--------------------------------------------------------------"
    desc += "\nTimeStep : "+str(maxTimeStep) 
    desc += "\nEpisode : "+ str(maxEpisode) 
    desc += "\nIteration : "+str(maxIteration)
    desc += "\ndistanceThreshold : "+str(distanceThreshold)
    
    saveFile(outputDir+"allPolicyWeight",allPolicyWeight)
    saveFile(outputDir+"allMeanTimestep",allMeanTimestep)
    saveFile(outputDir+"allDistance",allDistance)
    saveFile(outputDir+"desc",desc)
    saveFile(outputDir+"allMeanReward",allMeanReward)
    
    plt.plot(allMeanTimestep,label=expName,color = "red")
    plt.ylabel('timestep per EP')
    plt.xlabel('iteration')
    plt.legend(bbox_to_anchor=(0.7, 1.1), loc=2, borderaxespad=0.)
    plt.savefig(outputDir+"MeanTimestep.png")
    plt.close()
    
    plt.plot(allDistance,label=expName,color = "red")
    plt.ylabel('Distance per iteration')
    plt.xlabel('iteration')
    plt.legend(bbox_to_anchor=(0.7, 1.1), loc=2, borderaxespad=0.)
    plt.savefig(outputDir+"Distance.png")
    plt.close()
    
    plt.plot(allMeanReward,label=expName,color = "red")
    plt.ylabel('MeanReward per iteration')
    plt.xlabel('iteration')
    plt.legend(bbox_to_anchor=(0.7, 1.1), loc=2, borderaxespad=0.)
    plt.savefig(outputDir+"MeanReward.png")
    plt.close()
    
def loadExp(expName):
    outputDir = rootDir+expName+"\\"
    if not os.path.exists(outputDir):
        print("error path not exist")
        return(0)
        
    allPolicyWeight = loadFile(rootDir+expName+"\\allPolicyWeight")
    allMeanTimestep = loadFile(rootDir+expName+"\\allMeanTimestep")
    allDistance = loadFile(rootDir+expName+"\\allDistance")
    allMeanReward = loadFile(rootDir+expName+"\\allMeanReward")
    
    return [allPolicyWeight,allMeanTimestep,allDistance,allMeanReward]
    
maxTimeStep = 6000
maxIteration = 200
def LSPI():
    allPolicyWeight = []
    allMeanTimestep = []
    allDistance = []
    allSamples = []
    allMeanReward = []
    
    policyWeight = np.matrix(np.zeros((D,1))) #initial policy
    allPolicyWeight.append(policyWeight)
    
    distance = np.math.inf
    iteration = 0
    while iteration < maxIteration and distance > distanceThreshold:
        
        obj = collectSamples(policyWeight) #optional
        samples = obj['samples']
        avgReward = obj['avgReward']
        policyWeight = LSQ(samples, policyWeight)
        distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
        print(iteration,"average time steps :",len(samples)/maxEpisode,"distance",distance)
        iteration +=1
        
        #record-----------------------------
        allPolicyWeight.append(policyWeight)
        allMeanTimestep.append(len(samples)/maxEpisode)
        allDistance.append(distance)
        allMeanReward.append(avgReward)
    
    print(policyWeight,len(collectSamples(policyWeight)['samples'])/maxEpisode)
    
    #expName = "reward 0 1"
    #expName = "reward -1 1"
    #expName = "reward -1 fx1"
    #expName = "reward -1 fx2"
    expName = "reward -1 fx2 6000 TimeStep"
    #expName = "reward -1 0"
    saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
    
#maxIteration = 100 #max itteration of LSPI if distance not less than predefine threshold
#distanceThreshold = 0 #threshold to judge that policy is convert
def LSPI2():
    allPolicyWeight = []
    allMeanTimestep = []
    allDistance = []
    allSamples = []
    allMeanReward = []
    
    policyWeight = np.matrix(np.zeros((D,1))) #initial policy
    allPolicyWeight.append(policyWeight)
    distance = np.math.inf
    
    iteration = 0
    samples = collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b2)['samples']
    
    while iteration < maxIteration and distance > distanceThreshold:
        
        obj = collectSamples(policyWeight) #optional
        tempsamples = obj['samples']
        avgReward = obj['avgReward']
        policyWeight = LSQ(samples, policyWeight)
        distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
        print(iteration,"average time steps :",len(tempsamples)/maxEpisode,"distance",distance)
        iteration +=1

        #record-----------------------------
        allPolicyWeight.append(policyWeight)
        allMeanTimestep.append(len(tempsamples)/maxEpisode)
        allDistance.append(distance)
        allMeanReward.append(avgReward)
    
    print(policyWeight,len(collectSamples(policyWeight)['samples'])/maxEpisode)
    
    #expName = "init sample b1 (reward -1 1)"
    #expName = "init sample b2 (reward -1 1)"
    #expName = "init sample b1 b2 (reward -1 1)"
    
    #expName = "init sample b1 (reward -1 fx2)"
    #expName = "init sample b2 (reward -1 fx2)"
    #expName = "init sample b1 b2 (reward -1 fx2)"
    #expName = "init sample b1 b1 b1 b1 b2 (reward -1 fx2)"
    expName = "test2"
    saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
    
    
#allPolicyWeight,allMeanTimestep,allDistance,allMeanReward = loadExp(expName) #for load option

# expList = ["init sample b1 (reward -1 1)"]
# expList.append("init sample b2 (reward -1 1)")
# expList.append("init sample b1 b2 (reward -1 1)")
# 
# expList = ["init sample b1 (reward -1 fx2)"]
# # expList.append("init sample b1 (reward -1 fx2)")
# expList.append("init sample b2 (reward -1 fx2)")
# expList.append("init sample b1 b2 (reward -1 fx2)")

# expList = ["reward 0 1"]
# expList.append("reward -1 1")
# expList.append("reward -1 fx1")
# expList.append("reward -1 fx2")
# 
# expList = ["reward -1 fx2 6000 TimeStep"]
# expList.append("reward -1 fx2")
# 
# 
# plt.close()
# 
# plt.figure( figsize=(20, 9))
# plt.ylim(0,6100)
# for expName in expList:
#     allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(expName) 
#     plt.plot(meanTimeStep,label=expName,linewidth = 2 )#,linestyle = "dashed")#,color = "red")
# plt.title("Experiment 3 and 2",loc = "center")
# plt.grid('on')
# 
# lgd = plt.legend(bbox_to_anchor=(0.5,-.1), loc=9, borderaxespad=.1,ncol = 3)
# for line in lgd.get_lines():
#     line.set_linewidth(3)
# plt.savefig(rootDir+'exp3and2.png',additional_artists = lgd,bbox_inches='tight')
# 
# plt.show()
# 
# 
# policyWeight = allPolicyWeight[110]
# s1 = renderPolicy(policyWeight)
# 
# policyWeight = allPolicyWeight[160]
# s2 = renderPolicy(policyWeight)
# 
# reward = []
# reward2 = []
# n = len(s1)
# for i in range(n):
#     reward.append(s1[i].reward)
#     
# n = len(s2)
# for i in range(n):    
#     reward2.append(s2[i].reward)
#     
#     
# policyWeight = allPolicyWeight[90]
# s3 = renderPolicy(policyWeight)
# 
# policyWeight = allPolicyWeight2[90]
# s4 = renderPolicy(policyWeight)
# 
# reward3 = []
# reward4 = []
# n = len(s3)
# for i in range(n):
#     reward3.append(s3[i].reward)
#     
# n = len(s4)
# for i in range(n):    
#     reward4.append(s4[i].reward)
