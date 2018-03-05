import pickle
import os#, errno
import gym
#env._max_episode_steps = 50000 #new version of gym is very ugly with default _max_episode_steps, if _max_episode_steps is reach, isAbsorb is return
import numpy as np
import random 
import itertools
from collections import namedtuple

random.seed(42)
env = gym.make('CartPole-v0')

#constant for pendulumn problem
maxDegree = 12*2*np.math.pi/360 #max possible pole degree(radian)
maxX = 2.4 #max distance in X axist
possibleAction = 2 #amount of possible action

cX = [-maxX,0, maxX]
cA = [-maxDegree, 0 , maxDegree]
cAD = [-1,0, 1]
#D = 8 #dimension of state
D = ( len(cX) * len(cA) * len(cAD) +1 ) * 2

relativeVectors = []
for w in cX :
    #for x in cXD :
    for y in cA :
        for z in cAD :
            relativeVectors.append([w,y,z])

#constant for algorithm
discountFactor = 0.9 # set abitary 0 to 1
explorationRate = 0.01
lamda = 0.0

maxTimeStep = 6000 #timestep for each EP if absorb state no found
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

w1 = b2.copy()
w1[1] = 0
w1[5] = 0

w2 = w1.copy()
w2[0] = 0
w2[4] = 0

w3 = w1.copy()
w3[3] = 0
w3[7] = 0 

w4 = w1.copy()
w4[0] = -0.5
w4[4] = 0.5

w5 = w4.copy()
w5[0] = -0.5
w5[1] = 0.5
w5[4] = 0.5
w5[5] = -0.5

#function
'''
def basis(state, action): #best
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
    state = [state[0],state[2],state[3]]
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
    temp[count:(count + n),0] = yValue
                 
    return(temp)


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


def rewardFunction01(nextState, isAbsorb):
    
    if isAbsorb:
        reward = 0
    else:
        reward = 1
    return(reward)

def rewardFunctionMinus10(nextState, isAbsorb):
    
    if isAbsorb:
        reward = -1
    else:
        reward = 0
    return(reward)

def rewardFunctionMinus11(nextState, isAbsorb):
    
    if isAbsorb:
        reward = -1
    else:
        reward = 1
    return(reward)

def rewardFunction1(nextState,isAbsorb):
    
    if isAbsorb:
        reward = -1
    else:
        rewX = 1-np.linalg.norm(nextState[0]/maxX)
        rewXD = 1-np.linalg.norm(nextState[1]/maxX)
        rewA = 1-np.linalg.norm(nextState[2]/maxDegree)
        #rewAD = 1-np.linalg.norm(nextState[3]/2.5)
        rewAD = np.math.exp(-(nextState[3]*nextState[3]))
        reward = np.min([rewX ,rewA,rewAD ])
    return(reward)

def rewardFunction1V2(nextState,isAbsorb):
    

    rewX = np.linalg.norm(nextState[0]/maxX)
    rewXD = np.linalg.norm(nextState[1]/maxX)
    rewA = np.linalg.norm(nextState[2]/maxDegree)
    #rewAD = 1-np.linalg.norm(nextState[3]/2.5)
    rewAD =  np.linalg.norm(nextState[3])
    reward = -rewX -rewA -rewAD
    return(reward)
    
def rewardFunction2(nextState,isAbsorb):#best
    
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

def rewardFunction3(nextState,isAbsorb):
    
    if isAbsorb:
        reward = -1
    else:
        x = np.linalg.norm(nextState[0]/maxX)/2
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

def rewardFunction4(nextState,isAbsorb):
    positionX = nextState[0]
    Angle = nextState[2]
    reward = -np.math.pow(positionX,2) -np.math.pow(Angle,8)
    return(reward)

def rewardFunction4V2(nextState,isAbsorb):
    positionX = nextState[0]/maxX
    Angle = nextState[2]/maxDegree
    # positionX = nextState[0]
    # Angle = nextState[2]
    reward =  np.min([-np.math.pow(positionX,128), -np.math.pow(Angle,16) ]) 
    return(reward)
    
def rewardFunction5(nextState,isAbsorb):
    positionX = nextState[0]
    Angle = nextState[2]
    reward = -np.math.pow(Angle,2)
    return(reward)
    
    
def collectSamples(policyWeight,rewardFunction = rewardFunction2):
    env._max_episode_steps = 50000
    samples = []
    sumreward = []
    for i in range(maxEpisode):

        state = env.reset() #reset simulation to start position
        for j in range(maxTimeStep):
            action = policyFunction(policyWeight, state)
            #action = int(np.round( random.random())) #pure random
            nextState, reward, isAbsorb, info = env.step(action) #do action
            
            reward = rewardFunction(nextState,isAbsorb)
            
            #record
            samples.append( mySample(state,action,reward,nextState,isAbsorb) )
            sumreward.append(reward)
            state = nextState
  
            if isAbsorb: #the game end befor max timestep reached
                # reward = -1 # 0 or -1
                break

    return(  {'samples':samples,'avgReward':np.average(sumreward)}  )

def renderPolicy(policyWeight,rewardFunction = rewardFunction2):
    samples = []
    sumreward = []
    env._max_episode_steps = 50000
    state = env.reset() #reset simulation to start position
    
    for j in range(maxTimeStep):
        env.render()
        action = policyFunction(policyWeight, state)
        #action = int(np.round( random.random())) #pure random policy
        nextState, reward, isAbsorb, info = env.step(action) #do action
        
        reward = rewardFunction(nextState,isAbsorb)
        
        #record
        samples.append( mySample(state,action,reward,nextState,isAbsorb) )
        sumreward.append(reward)
        state = nextState
                 
        if isAbsorb: #the game end befor max timestep reached
            break

    env.close()
    print(j+1)
    return(  {'samples':samples,'avgReward':np.average(sumreward)}  )

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

def SARSALamda(samples, policyWeight,lamda):
    A = np.matrix(np.zeros((D,D)))
    B = np.matrix(np.zeros((D,1)))
    n = len(samples)
    z = np.zeros((D,1))
    for i in range(n-1): #start from i=0 to i=n-1
        phi = basis(samples[i].state,samples[i].action)
        z = (lamda)*z + phi
        if samples[i].isAbsorb != True: #check if next state is not absorb state
            nextAction = samples[i+1].action
            nextPhi = basis(samples[i].nextState, nextAction)
        else:
            nextPhi = np.zeros((D,1))
        
        A = A + z * np.transpose(phi - discountFactor * nextPhi)
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
    

def LSPI(rewardFunction,expName,initSample = False,fix = False):
    newSample = False
    allPolicyWeight = []
    allMeanTimestep = []
    allDistance = []
    allSamples = []
    allMeanReward = []
    
    policyWeight = np.matrix(np.zeros((D,1))) #initial policy
    allPolicyWeight.append(policyWeight)
    distance = np.math.inf
    iteration = 0
    
    if initSample != False:#if init sample exist
        samples = initSample
    else:#no init sample
        samples = collectSamples(policyWeight,rewardFunction) 
    while iteration < maxIteration and distance > distanceThreshold:
        if not fix and newSample != False:#in case not fix sample and there are newSample
            samples = newSample
        
        print("input---------",len(samples['samples'])/maxEpisode,"--------------")
        policyWeight = LSQ(samples['samples'], policyWeight)
        distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
        newSample = collectSamples(policyWeight,rewardFunction) #for measure performance
        print(iteration,"average time steps :",len(newSample['samples'])/maxEpisode,"distance",distance)
        iteration +=1
        
        #record-----------------------------
        allPolicyWeight.append(policyWeight)
        allMeanTimestep.append(len(newSample['samples'])/maxEpisode)
        allDistance.append(distance)
        allMeanReward.append(newSample['avgReward'])

    saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
    

def LSPILamda(rewardFunction,expName,lamda,algo):
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
        
        obj = collectSamples(policyWeight,rewardFunction) #optional
        samples = obj['samples']
        avgReward = obj['avgReward']
        policyWeight = algo(samples, policyWeight, lamda)
        distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
        print(iteration,"average time steps :",len(samples)/maxEpisode,"distance",distance)
        iteration +=1
        
        #record-----------------------------
        allPolicyWeight.append(policyWeight)
        allMeanTimestep.append(len(samples)/maxEpisode)
        allDistance.append(distance)
        allMeanReward.append(avgReward)
    
    saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
    
# def LSPILamda2(rewardFunction,expName,lamda):
#     allPolicyWeight = []
#     allMeanTimestep = []
#     allDistance = []
#     allSamples = []
#     allMeanReward = []
#     
#     policyWeight = np.matrix(np.zeros((D,1))) #initial policy
#     allPolicyWeight.append(policyWeight)
#     
#     distance = np.math.inf
#     iteration = 0
#     while iteration < maxIteration and distance > distanceThreshold:
#         
#         obj = collectSamples(policyWeight,rewardFunction) #optional
#         samples = obj['samples']
#         avgReward = obj['avgReward']
#         policyWeight = SARSALamda(samples, policyWeight, lamda)
#         distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
#         print(iteration,"average time steps :",len(samples)/maxEpisode,"distance",distance)
#         iteration +=1
#         
#         #record-----------------------------
#         allPolicyWeight.append(policyWeight)
#         allMeanTimestep.append(len(samples)/maxEpisode)
#         allDistance.append(distance)
#         allMeanReward.append(avgReward)
#     
#     print(policyWeight,len(collectSamples(policyWeight,rewardFunction)['samples'])/maxEpisode)
#     
#     #expName = "reward -1 fx2 6000 TimeStep"
#     saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
    
# lamdaList = [0.0,.2,.4,.6,.8,1.0]
# 
# for lamda in lamdaList:
#     expName = "reward -1 fx2 Lamda "+ str(lamda)+" SARSA"
#     LSPILamda(rewardFunction2,expName,lamda)
#     #LSPILamda(rewardFunctionMinus11,expName,lamda)

# lamda = 0.2
# 
# expName = "state 4D rewardFunction02 Lamda "+ str(lamda)
# LSPILamda(rewardFunction2,expName,lamda)

# [a,b,c,d] = loadExp("rewardFunction03 Lamda 0.2")
# renderPolicy(a[150])

#LSPILamda(rewardFunction01,expName,lamda)

# obj = collectSamples(b1,rewardFunction2)
# #renderPolicy(b2,rewardFunction2)
# s = obj['samples']
# n = len(s)
# min = np.inf
# max = 0
# for i in range(n):
#     if s[i][0][3]< min:
#         min = s[i][0][3]
#     if s[i][0][3]> max:
#         max = s[i][0][3]

import pandas
import matplotlib.pyplot as plt

def getColorList(num):
    color = ["red"]
    color.append("blue")
    color.append("green")
    color.append("orange")
    color.append("purple")
    color.append("#00ff00")
    color.append("#00ffff")
    color.append("#ffffff")
    color.append("#0000ff")
    color.append("#ff00ff")
    return color[0:num]
    
def getNEP(data,ep=50):
    d = data
    n = len(d['samples'])
    c = 0
    for i in range(n):
        if(d['samples'][i].isAbsorb):
            c = c+1
        if(c == ep):
            break
    return {'samples':d['samples'][0:(i+1)],'avgReward':d['avgReward']}