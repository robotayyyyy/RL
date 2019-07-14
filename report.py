import fileRecord,plot,os,param

f = fileRecord.ExpSaveLoad()
p = plot.expPlot()


def dictToDataAndLabel(d):#for extract only data and label from dict (as 2 arrays)
    return ({'data':[d[i] for i in d], 'label':[i for i in d]})

def getListInFolder(path):
    return(os.listdir(param.CartPoleParam().saveDir+path))

def getExpSeriesName( expname):
    return ([expname+'/'+s for s in  getListInFolder(expname)])
    
def report( expList, **args):
    prefix = ''
    label = 'NA'
    cof = True
    box = True
    
    if 'prefix' in args:
        prefix = args['prefix']
    if 'label' in args:
        label = args['label']
    if 'cof' in args:
        cof = args['cof']
    if 'box' in args:
        box = args['box']
        
    data = [ f.loadExp(e,cof = cof) for e in expList ]
    n = len(data)
    
    reward = [data[i]['reward'] for i in range(n)]
    distance = [data[i]['distance'] for i in range(n)]
    timeStep = [data[i]['timestep'] for i in range(n)]
    
    p.plot(reward,label,fileName=prefix+'reward',labelY='Reward per decision')
    p.plot(distance,label,fileName=prefix+'distance',labelY='distance')
    #p.plot(distance,label,boxPlot = True, fileName=prefix+'distanceB')
    p.plot(timeStep,label,fileName=prefix+'timeStep',labelY='Time step per episode')
    
    if box:
        p.plot(reward,label,boxPlot = True, fileName=prefix+'rewardB',labelY='Reward per decision')
        p.plot(timeStep,label,boxPlot = True, fileName=prefix+'timeStepB',labelY='Time step per episode')
    
    if cof:
        for key in data[0]['cof']:
            #print(key)   
            cofi = [ data[i]['cof'][key]  for i in range(n)]     
            p.plot(cofi,key,fileName=prefix+" cof "+key)
            p.plot(cofi,key,boxPlot = True, fileName=prefix+" cof "+key+'B')
            
        


#expList= getExpSeriesName('LSTDQLamda rewardFunction2 D_8 ITE_200 EP_20 TS_6000 LD_0')






#expName=[]
#LSPI(rewardFunction3,"reward -1 fx3 6k")
#
#LSPI(rewardFunction4V2,"reward fx4V2 6k")
#LSPI(rewardFunction4,"reward fx4 6k")
#LSPI(rewardFunction5,"reward fx5 6k")
#
#LSPI(rewardFunction1V2,"reward fx1V2 6k")
#
#LSPI(rewardFunctionMinus10,"reward -1 0 6k")
#
#initSampleA0 = collectSamples(b2,rewardFunction5)
#initSampleA1 = collectSamples(b1,rewardFunction5)
#LSPI(rewardFunction5,"init sample A0 (reward fx5) FIX 6k",initSampleA0,fix = True)
#
#allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp("init sample b2 b1 (reward -1 1) FIX 6k") 
#
#
## def LSPI2(rewardFunction,expName,initSample): # fixed sample
##     allPolicyWeight = []
##     allMeanTimestep = []
##     allDistance = []
##     allSamples = []
##     allMeanReward = []
##     
##     policyWeight = np.matrix(np.zeros((D,1))) #initial policy
##     allPolicyWeight.append(policyWeight)
##     distance = np.math.inf
##     
##     iteration = 0
##     samples = initSample['samples']#collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b1)['samples'] + collectSamples(b2)['samples']
##     
##     while iteration < maxIteration and distance > distanceThreshold:
##         
##         obj = collectSamples(policyWeight,rewardFunction) #optional
##         tempsamples = obj['samples'] #temp sample just for measure timestep
##         avgReward = obj['avgReward']
##         policyWeight = LSQ(samples, policyWeight)
##         distance = np.linalg.norm(policyWeight-allPolicyWeight[iteration])
##         print(iteration,"average time steps :",len(tempsamples)/maxEpisode,"distance",distance)
##         iteration +=1
## 
##         #record-----------------------------
##         allPolicyWeight.append(policyWeight)
##         allMeanTimestep.append(len(tempsamples)/maxEpisode)
##         allDistance.append(distance)
##         allMeanReward.append(avgReward)
##     
##     print(policyWeight,len(collectSamples(policyWeight,rewardFunction)['samples'])/maxEpisode)
##     saveExp(expName,allPolicyWeight,allMeanTimestep,allDistance,allMeanReward)
#
#distanceThreshold = 0
#maxTimeStep = 6000
#explorationRate
## # #reward -1 1
#outputDir = rootDir+"\\initsample_rewardminus11\\"
## if not os.path.exists(outputDir):
##     os.makedirs(outputDir)
## initSampleB1 = collectSamples(b1,rewardFunctionMinus11)
## initSampleB2 = collectSamples(b2,rewardFunctionMinus11)
## 
## samples = getNEP(initSampleB1,50)['samples'] + getNEP(initSampleB2,50)['samples']
## avgReward = (initSampleB1['avgReward'] + initSampleB2['avgReward'])/2
## initSampleB1B2 = {'samples':samples,'avgReward':avgReward}
## 
## samples = getNEP(initSampleB2,50)['samples'] + getNEP(initSampleB1,50)['samples']
## avgReward = (initSampleB1['avgReward'] + initSampleB2['avgReward'])/2
## initSampleB2B1 = {'samples':samples,'avgReward':avgReward}
## 
## #----- save var
## saveFile(outputDir+"initSampleB1",initSampleB1)
## saveFile(outputDir+"initSampleB2",initSampleB2)
## saveFile(outputDir+"initSampleB1B2",initSampleB1B2)
## saveFile(outputDir+"initSampleB2B1",initSampleB2B1)
## #----- load var
#initSampleB1 = loadFile(outputDir+"\\initSampleB1")
#initSampleB2 = loadFile(outputDir+"\\initSampleB2")
#initSampleB1B2 = loadFile(outputDir+"\\initSampleB1B2")
#initSampleB2B1 = loadFile(outputDir+"\\initSampleB2B1")
## 
## # #reward -1 fx2
#
#outputDir = rootDir+"\\initsample_rewardfx2\\"
#
## if not os.path.exists(outputDir):
##     os.makedirs(outputDir)
## initSampleB1X = collectSamples(b1,rewardFunction2)
## initSampleB2X = collectSamples(b2,rewardFunction2)
## 
## samples = getNEP(initSampleB1X,50)['samples'] + getNEP(initSampleB2X,50)['samples']
## avgReward = (initSampleB1['avgReward'] + initSampleB2['avgReward'])/2
## initSampleB1B2X = {'samples':samples,'avgReward':avgReward}
## 
## samples = getNEP(initSampleB2X,50)['samples'] + getNEP(initSampleB1X,50)['samples']
## avgReward = (initSampleB1['avgReward'] + initSampleB2['avgReward'])/2
## initSampleB2B1X = {'samples':samples,'avgReward':avgReward}
## 
##w = np.random.random((8, 1))-0.5
##initSampleBR = collectSamples(w,rewardFunction2)
## #----- save var
## saveFile(outputDir+"initSampleB1X",initSampleB1X)
## saveFile(outputDir+"initSampleB2X",initSampleB2X)
## saveFile(outputDir+"initSampleB1B2X",initSampleB1B2X)
## saveFile(outputDir+"initSampleB2B1X",initSampleB2B1X)
## saveFile(outputDir+"initSampleBR",initSampleBR)
#
## #----- load var
#initSampleB1X = loadFile(outputDir+"\\initSampleB1X")
#initSampleB2X = loadFile(outputDir+"\\initSampleB2X")
#initSampleB1B2X = loadFile(outputDir+"\\initSampleB1B2X")
#initSampleB2B1X = loadFile(outputDir+"\\initSampleB2B1X")
#initSampleBR = loadFile(outputDir+"\\initSampleBR")
## 
## 
#LSPI(rewardFunctionMinus11,"init sample b1 (reward -1 1) FIX 6k",initSampleB1,fix = True)
#LSPI(rewardFunctionMinus11,"init sample b2 (reward -1 1) FIX 6k",initSampleB2,fix = True)
#LSPI(rewardFunctionMinus11,"init sample b1 b2 (reward -1 1) FIX 6k",initSampleB1B2,fix = True)
#LSPI(rewardFunctionMinus11,"init sample b2 b1 (reward -1 1) FIX 6k",initSampleB2B1,fix = True)
#
#LSPI(rewardFunctionMinus11,"init sample b1 (reward -1 1) NONFIX 6k",initSampleB1)
#LSPI(rewardFunctionMinus11,"init sample b2 (reward -1 1) NONFIX 6k",initSampleB2)
#LSPI(rewardFunctionMinus11,"init sample b1 b2 (reward -1 1) NONFIX 6k",initSampleB1B2)
#LSPI(rewardFunctionMinus11,"init sample b2 b1 (reward -1 1) NONFIX 6k",initSampleB2B1)
#
#LSPI(rewardFunction2,"init sample b1 (reward -1 fx2) FIX 6k",initSampleB1X,fix = True)
#LSPI(rewardFunction2,"init sample b2 (reward -1 fx2) FIX 6k",initSampleB2X,fix = True)
#LSPI(rewardFunction2,"init sample b1 b2 (reward -1 fx2) FIX 6k",initSampleB1B2X,fix = True)
#LSPI(rewardFunction2,"init sample b2 b1 (reward -1 fx2) FIX 6k",initSampleB2B1X,fix = True)
#
#LSPI(rewardFunction2,"init sample b1 (reward -1 fx2) NONFIX 6k",initSampleB1X)
#LSPI(rewardFunction2,"init sample b2 (reward -1 fx2) NONFIX 6k",initSampleB2X)
#LSPI(rewardFunction2,"init sample b1 b2 (reward -1 fx2) NONFIX 6k",initSampleB1B2X)
#LSPI(rewardFunction2,"init sample b2 b1 (reward -1 fx2) NONFIX 6k",initSampleB2B1X)
#
#LSPI(rewardFunction2,"init sample br (reward -1 fx2) FIX 6k",initSampleBR,fix = True)
#LSPI(rewardFunction2,"init sample br (reward -1 fx2) NONFIX 6k",initSampleBR)
## 
## 
## LSPI(rewardFunction01,"reward 0 1 6k")
## LSPI(rewardFunctionMinus10,"reward -1 0 6k")
## LSPI(rewardFunctionMinus11,"reward -1 1 6k")
## LSPI(rewardFunction1,"reward -1 fx1 6k")
## LSPI(rewardFunction2,"reward -1 fx2 6k")
## 
## 
#
#lamdaList = [0.0,.2,.4,.6,.8,1.0]
#for lamda in lamdaList:
#    # expName = "reward -1 fx2 Lamda "+ str(lamda)+" SARSA 6k"
#    # LSPILamda(rewardFunction2,expName,lamda,SARSALamda)
#    expName = "reward -1 fx2 Lamda "+ str(lamda)+" 6k"
#    LSPILamda(rewardFunction2,expName,lamda,LSQLamda)
#    
#    
#maxTimeStep = 6000
#maxEpisode = 300
#maxIteration = 10  
#explorationRate = 0
#expName = "reward -1 fx2 Lamda 6k RBF"
#LSPILamda(rewardFunction2,expName,lamda,LSQLamda)
#
#allPolicyWeight,meanTimeStep,distance,allMeanReward = loadExp(expName) 
#
#b=[]
#for i in range(10):
#    a = renderPolicy(allPolicyWeight[1])
#    print(i,len(a['samples']))
#    b.append(len(a['samples']))
#
#
## # 
## # #reward -1 fx2
## # initSampleB1 = collectSamples(b1,rawardFunction2)['samples']
## # LSPI2(rawardFunction2,"init sample b1 (reward -1 fx2)",initSampleB1)
## # 
## # initSampleB2 = collectSamples(b2,rawardFunction2)['samples']
## # LSPI2(rawardFunction2,"init sample b2 (reward -1 fx2)",initSampleB2)
## # 
## # initSampleB1B2 = initSampleB1+initSampleB2
## # LSPI2(rawardFunction2,"init sample b1 b2 (reward -1 fx2)",initSampleB1B2)
## 
## 
## 
## #allPolicyWeight,allMeanTimestep,allDistance,allMeanReward = loadExp(expName) #for load option
## 
## # expList = ["init sample b2 (reward -1 1)"]
## # expList.append("init sample b2 (reward -1 1)")
## # expList.append("init sample b1 b2 (reward -1 1)")
## # 
## # expList = ["reward -1 fx2 6000 TimeStep"]
## # expList.append("reward -1 fx2")
## 
## # expList = ["reward -1 fx2 Lamda 0.0"]
## # expList.append("reward -1 fx2 Lamda 0.2")
## # expList.append("reward -1 fx2 Lamda 0.4")
## # expList.append("reward -1 fx2 Lamda 0.6")
## # expList.append("reward -1 fx2 Lamda 1.0")
## # 
## # expList = ["reward -1 fx2 Lamda 0.0 SARSA"]
## # expList.append("reward -1 fx2 Lamda 0.2 SARSA")
## # expList.append("reward -1 fx2 Lamda 0.4 SARSA")
## # expList.append("reward -1 fx2 Lamda 0.6 SARSA")
## # expList.append("reward -1 fx2 Lamda 1.0 SARSA")
## 
## 
## # expName = ["init sample b1 (reward -1 fx2) NONFIX 6k"]
## # expName.append("init sample b2 (reward -1 fx2) NONFIX 6k")
## # expName.append("init sample b1 b2 (reward -1 fx2) NONFIX 6k")
## # 
## # expName.append("init sample b1 (reward -1 fx2) FIX 6k")
## # expName.append("init sample b2 (reward -1 fx2) FIX 6k")
## # expName.append("init sample b1 b2 (reward -1 fx2) FIX 6k")
#
##--------------------------
## exp 01
##--------------------------
#expName = ["init sample b2 (reward -1 fx2) NONFIX 6k"]
#expName.append("init sample b2 (reward -1 fx2) FIX 6k")
#
#
#expName = ["init sample b1 (reward -1 fx2) NONFIX 6k"]
#expName.append("init sample b1 (reward -1 fx2) FIX 6k")
#
#
# expName = ["init sample br (reward -1 fx2) NONFIX 6k"]
# expName.append("init sample br (reward -1 fx2) FIX 6k")
# 
# expLabel = expName #by default
# expLabel = ["Updated sample"]
# expLabel.append("Fixed sample")
# 
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(2)})
# expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":['blue','orange']})
# 
# plt.close()
# plt.rcParams.update({'font.size': 20})
# plt.figure( figsize=(10, 7))
# plt.ylim(0,6100)
# for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    meanTimeStep[0] = meanTimeStep[1] 
#    if index == 0 :
#        meanTimeStep[142]=6000
#        meanTimeStep[143]=6000
#        meanTimeStep[145]=6000
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color)
#    #plt.semilogy(meanTimeStep,linewidth = 2 ,color = row.color)
# plt.grid('on')
# 
# lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
# plt.xlabel("iteration")
# plt.ylabel("Average timestep")
# 
# plt.savefig(rootDir+'expT01.png',additional_artists = (lgd,), bbox_inches='tight')
# plt.show()
#
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(0,1)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    allMeanReward[0] = allMeanReward[2]
#
#    plt.plot(allMeanReward,label=row.expLabel,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average reward")
#plt.savefig(rootDir+'expR01.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(-5,100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(distance,label=row.expLabel,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Polocy distance")
#plt.savefig(rootDir+'expD01.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
##--------------------------
## exp 02
##--------------------------
#expName = ["init sample b1 (reward -1 fx2) NONFIX 6k"]
#expName.append("init sample b2 (reward -1 fx2) NONFIX 6k")
#expName.append("init sample b1 b2 (reward -1 fx2) NONFIX 6k")
#
#
#expName = ["init sample b1 (reward -1 1) NONFIX 6k"]
#expName.append("init sample b2 (reward -1 1) NONFIX 6k")
#expName.append("init sample b1 b2 (reward -1 1) NONFIX 6k")
#
#expLabel = expName #by default
#expLabel = ["s1"]
#expLabel.append("s2")
#expLabel.append("s3")
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(3)})
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average timestep")
#plt.savefig(rootDir+'expT02.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(-0.15,1)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    if index == 2: #dataset of index2 contain epx2 !!!!!error fix!!!!!
#        allMeanReward[0] = allMeanReward[0]/3
#    plt.plot(allMeanReward,label=row.expLabel,linewidth = 2 ,color = row.color)
#plt.grid('on')
#
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average reward")
#plt.savefig(rootDir+'expR02.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(-5,150)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot( np.asarray(distance),label=row.expLabel,linewidth = 2 ,color = row.color)
#    
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Polocy distance")
#plt.savefig(rootDir+'expD02.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#
##--------------------------
## exp 2
##--------------------------
#expName = ["reward -1 fx2 Lamda 0.0 6k"]
#expName.append("reward -1 1 6k") 
#expName.append("reward 0 1 6k")
##expName.append("reward -1 fx1 6k")
##expName.append("reward -1 fx2 6k")
##expName.append("reward -1 fx2 Lamda 0.0 6k")
#
#expLabel = expName #by default
#expLabel = ["$R_{1}$"]
#expLabel.append("$R_{2}$")
#expLabel.append("$R_{3}$")
##expLabel.append("$R_{4}$")
#
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel})
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(3)})
#
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
##plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color)
#    #plt.plot(np.log(np.asarray(meanTimeStep)),label=row.expLabel,linewidth = 2 ,color = row.color)
#    plt.semilogy(meanTimeStep,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average timestep in log scale")
#
#plt.savefig(rootDir+'exp2L.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
##plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(allMeanReward,label=row.expLabel,linewidth = 2 ,color = row.color)
#    #plt.semilogy(allMeanReward,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average reward in log scale")
#
#plt.savefig(rootDir+'exp2RL.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
##--------------------------
## exp 2V2
##--------------------------
#expName = ["reward -1 0 6k"]
#expName.append("reward fx5 6k")
#expName.append("reward fx4 6k")
#expName.append("reward -1 fx1 6k")
#expName.append("reward -1 fx2 Lamda 0.0 6k")
#
#expLabel = ["$R_{1}$"]
#expLabel.append("$R_{2}$")
#expLabel.append("$R_{3}$")
#expLabel.append("$R_{4}$")
#expLabel.append("$R_{5}$")
#
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel})
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(5)})
#
#
#lineS = [ '--','-' , ':','-.','--']
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
##plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    LS = lineS.pop()
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color,linestyle =LS)
#
#    #plt.semilogy(meanTimeStep,linewidth = 2 ,color = row.color,linestyle = LS)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average timestep in log scale")
#
#plt.savefig(rootDir+'exp2V2L.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
##plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(allMeanReward,label=row.expLabel,linewidth = 2 ,color = row.color)
#    #plt.semilogy(allMeanReward,linewidth = 2 ,color = row.color)
#plt.grid('on')
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average reward in log scale")
#
#plt.savefig(rootDir+'exp2V2RL.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
##--------------------------
## exp 3
##--------------------------
#
#expName = ["reward -1 fx2 Lamda 0.0 6k"]
##expName = ["reward -1 fx2 6k"]
#expName.append("reward -1 fx2 Lamda 0.2 6k")
#expName.append("reward -1 fx2 Lamda 0.4 6k")
#expName.append("reward -1 fx2 Lamda 0.6 6k")
#expName.append("reward -1 fx2 Lamda 0.8 6k")
#expName.append("reward -1 fx2 Lamda 1.0 6k")
#
## expName = ["reward -1 fx2 Lamda 0.0 SARSA 6k"]
## #expName = ["reward -1 fx2 6k"]
## expName.append("reward -1 fx2 Lamda 0.2 SARSA 6k")
## expName.append("reward -1 fx2 Lamda 0.4 SARSA 6k")
## expName.append("reward -1 fx2 Lamda 0.6 SARSA 6k")
## expName.append("reward -1 fx2 Lamda 0.8 SARSA 6k")
## expName.append("reward -1 fx2 Lamda 1.0 SARSA 6k")
#
#
#expLabel = expName #by default
#expLabel = ["λ 0.0"]
#expLabel.append("λ 0.2")
#expLabel.append("λ 0.4")
#expLabel.append("λ 0.6")
#expLabel.append("λ 0.8")
#expLabel.append("λ 1.0")
#
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel})
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(6)})
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color)
#plt.grid('on')
#
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.1, 0.5) ,ncol = 1, mode="none",fontsize = 'small')
#plt.xlabel("iteration")
#plt.ylabel("Average timestep")
#plt.savefig(rootDir+'exp3.png',additional_artists = (lgd,), bbox_inches='tight')
#plt.show()
#
##--------------------------
## exp 4
##--------------------------
##expName = ["reward -1 fx2 Lamda 0.0"]
#expName = [("reward -1 fx2 Lamda 0.2 6k")]
#expName.append("reward -1 fx2 Lamda 0.2 SARSA 6k")
#
#
#expLabel = expName #by default
#expLabel = ["Q-learning"]
#expLabel.append("SARSA")
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel})
#expIndex = pandas.DataFrame({"expName":expName,"expLabel":expLabel,"color":getColorList(2)})
#
#plt.close()
#plt.rcParams.update({'font.size': 20})
#plt.figure( figsize=(10, 7))
#plt.ylim(0,6100)
#for index,row in expIndex.iterrows():
#    allPolicyWeight2,meanTimeStep,distance,allMeanReward = loadExp(row.expName) 
#    plt.plot(meanTimeStep,label=row.expLabel,linewidth = 2 ,color = row.color)#,linestyle = "dashed")#,color = "red")
#plt.grid('on')
#
#lgd = plt.legend(loc =10 ,bbox_to_anchor=(1.15, 0.5) ,ncol = 1, mode="none",fontsize = 'small')#loc=(0,-.25)
## for line in lgd.get_lines():
##     line.set_linewidth(3)
#plt.xlabel("iteration")
#plt.ylabel("Average timestep")
##plt.ylabel("Average reward")
#
#plt.savefig(rootDir+'exp4.png',additional_artists = (lgd,), bbox_inches='tight')
#
#plt.show()
