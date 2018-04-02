import pandas
import matplotlib.pyplot as plt
import param
import numpy as np

class expPlot:
    
    def getColorList(self,num):
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
    
    def plot(self,data,label,**args): #data must be [array] or [array of array], label element must equal n
        boxPlot = False
        n = len(data) #in is number if series
        if len(label) != n: #in case amount label not equal to series of data
            label = list(range(n))
            
        print('n = ',n)
        color = self.getColorList(n)
        lineS = np.repeat('-',n)
        title = False
        
        gap = np.nanmax(np.concatenate(data)) - np.nanmin(np.concatenate(data))
        maxV = np.nanmax(np.concatenate(data)) + gap*0.05
        minV = np.nanmin(np.concatenate(data)) - gap*0.05
            
        logSacle = False
        lx = 1.1
        ly = 0.5
        lcol = 1
        labelX = 'Iteration'
        labelY = 'Average time step'
        fileName = 'temp'
        rootDir = param.CartPoleParam().saveDir
        
        if 'boxPlot' in args:
            boxPlot = args['boxPlot']
        if 'color' in args:
            color = args['color']
        if 'lineS' in args:
            lineS = args['lineS']  
        if 'title' in args:
            title = args['title'] 
        if 'max' in args:
            maxV = args['max']
        if 'min' in args:
            minV = args['min']
        if 'logSacle' in args:
            logSacle = args['logSacle']
        if 'lx' in args:
            lx = args['lx']
        if 'ly' in args:
            ly = args['ly']
        if 'lcol' in args:
            lcol = args['lcol']
        if 'labelX' in args:
            labelX = args['labelX']
        if 'labelY' in args:
            labelY = args['labelY']
        if 'fileName' in args:
            fileName = args['fileName']
        if 'rootDir' in args:
            rootDir = args['rootDir']
        
        if np.min(np.concatenate(data)) <0: #can't input <0 to log function
            logSacle = False
        if logSacle != False:
            labelY += ' in log scale'
        expIndex = pandas.DataFrame({"data":data,"label":label,"color":color,"lineS":lineS})
        
        
        plt.close()
        plt.rcParams.update({'font.size': 20})
        plt.figure( figsize=(10, 7))
        plt.ylim(minV,maxV)
        plt.xlim(0, max([len(data[i]) for i in range(len(data))]) -1)#make sure that x axis start at 0
        
        if boxPlot:
            tdata = list(np.array(np.matrix.transpose(np.matrix(data))) )
            plt.plot([np.nan]+[np.mean(tdata[i]) for i in range(len(tdata))],label = 'mean')
            plt.plot([np.nan]+[np.max(tdata[i]) for i in range(len(tdata))],label = 'max')
            plt.plot([np.nan]+[np.min(tdata[i]) for i in range(len(tdata))],label = 'min')
            plt.plot([np.nan]+[np.percentile(tdata[i],25) for i in range(len(tdata))],label = 'Q1')
            plt.plot([np.nan]+[np.percentile(tdata[i],50) for i in range(len(tdata))],label = 'Q2')
            plt.plot([np.nan]+[np.percentile(tdata[i],75) for i in range(len(tdata))],label = 'Q3')
            #plt.boxplot(tdata)
        else:
            for index,row in expIndex.iterrows():
                plt.plot( row['data'], label = row.label, linewidth = 2, color = row.color, linestyle = row.lineS)
                if(logSacle != False):
                    plt.semilogy(row.data, linewidth = 2, color = row.color, inestyle = row.lineS)
                
        plt.grid('on')
        lgd = plt.legend(loc =10 ,bbox_to_anchor=(lx, ly) ,ncol = lcol, mode="none",fontsize = 'small')
        plt.xlabel(labelX)
        plt.ylabel(labelY)
        if title != False:
            plt.title(title)
        
        plt.savefig(rootDir+fileName+'.png',additional_artists = (lgd,), bbox_inches='tight')
        plt.show()
        