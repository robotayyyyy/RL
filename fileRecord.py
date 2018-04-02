import pickle
import os
import param
import plot


#os.listdir( param.CartPoleParam().saveDir+'LSTDQLamda rewardFunction2 D_8 ITE_10 EP_300 TS_6000')

class ExpSaveLoad:
    def __init__(self, **args):
        
        self.__rootDir = param.CartPoleParam().saveDir
        if 'rootDir' in args:
            self.__rootDir = args['rootDir']
        
    def saveFile(self,filename,var):
        with open(filename, 'wb') as fp:
            pickle.dump(var, fp)   

    def saveExp(self,expName,plotList, **args):
        outputDir = self.__rootDir+expName+"/"
        print(outputDir)
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            
        #plotList = args.popitem()[1] # graph is always the last element of args. When pop it return [key,value], select value then
        for key in args:
            self.saveFile(outputDir+key,args[key])#save variable
            if type(args[key]) == dict:
                if(plotList.pop(0) == 1):#pop first element then plot
                    subKey = [i for i in args[key]] #subKey is label for now
                    data = [args[key][i] for i in args[key]]
                    plot.expPlot().plot(data, subKey, fileName = key, rootDir = outputDir)
#                for subKey in args[key]:#save each series as 1 file
#                    self.saveFile(outputDir+subKey,args[key][subKey])    
            else:                    
                self.saveFile(outputDir+key,args[key])
                if(plotList.pop(0) == 1):#pop first element then plot
                    plot.expPlot().plot([args[key]], [key], fileName = key, rootDir =outputDir)

    def loadFile(self,filename):        
        with open (filename, 'rb') as fp:
            return(pickle.load(fp))
            
    def loadExp(self,expName):#not perfect but robust
        outputDir = self.__rootDir+expName+"\\"
        if not os.path.exists(outputDir):
            print("error path not exist")
            return(0)
        
        if os.path.exists(self.__rootDir+expName+"\\allPolicyWeight"):
            allPolicyWeight = self.loadFile(self.__rootDir+expName+"\\allPolicyWeight")
        else:
            allPolicyWeight = 'none'
        
        allMeanTimestep = self.loadFile(self.__rootDir+expName+"\\allMeanTimestep")
        allDistance = self.loadFile(self.__rootDir+expName+"\\allDistance")
        allMeanReward = self.loadFile(self.__rootDir+expName+"\\allMeanReward")
        cof = self.loadFile(self.__rootDir+expName+"\\cof")
#        cof['Out of track'] = self.loadFile(self.__rootDir+expName+"\\Out of track")
#        cof['Pole down'] = self.loadFile(self.__rootDir+expName+"\\Pole down")
#        cof['Time'] = self.loadFile(self.__rootDir+expName+"\\Time")
#        
        return {'weight':allPolicyWeight,'timestep':allMeanTimestep,'distance':allDistance,'reward':allMeanReward,'cof':cof}
    




    
#def getNEP(data,ep=50):
#    d = data
#    n = len(d['samples'])
#    c = 0
#    for i in range(n):
#        if(d['samples'][i].isAbsorb):
#            c = c+1
#        if(c == ep):
#            break
#    return {'samples':d['samples'][0:(i+1)],'avgReward':d['avgReward']}



        

#    
#def printSamples(samples):    
#        for i in range(len(samples)):
#            print(samples[i].state, samples[i].action, samples[i].reward, samples[i].isAbsorb)