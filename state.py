import numpy as np
import param

class FSimpleState:
    def __init__(self):

        basisSimpleD = 12
        self.dimension = basisSimpleD
        #dummy weight for testing
        self.dummyWeight = np.matrix( np.random.rand(self.dimension).reshape(self.dimension,1)-0.5)
        
    def basis(self,state, action): #best
        D = self.dimension
        phi = np.zeros((D,1))
        if(action == 0):
            
            phi[0,0]= state[0]
            phi[1,0]= state[1]
            phi[2,0]= state[2]
            phi[3,0]= state[3]
            phi[4,0]= state[4]
            phi[5,0]= state[5]

        else:
            phi[6,0]= state[0]
            phi[7,0]= state[1]
            phi[8,0]= state[2]
            phi[9,0]= state[3]
            phi[10,0]= state[4]
            phi[11,0]= state[5]

        return(phi)
        
        
class CartPoleSimpleState:
    def __init__(self):
        self.__param = param.CartPoleParam()
        basisSimpleD = 8
        self.dimension = basisSimpleD
        #dummy weight for testing
        self.dummyWeight = np.matrix( np.random.rand(self.dimension).reshape(self.dimension,1)-0.5)
        
    def basis(self,state, action): #best
        D = self.dimension
        phi = np.zeros((D,1))
        if(action == 0):
            
            phi[0,0]= state[0]/self.__param.maxX
            phi[1,0]= state[1]/self.__param.maxX
            phi[2,0]= state[2]/self.__param.maxDegree
            phi[3,0]= state[3]/self.__param.maxDegree
        else:
            phi[4,0]= state[0]/self.__param.maxX
            phi[5,0]= state[1]/self.__param.maxX
            phi[6,0]= state[2]/self.__param.maxDegree
            phi[7,0]= state[3]/self.__param.maxDegree
        return(phi)
    

class CartPoleRBFState:
    def __init__(self):
        self.__param = param.CartPoleParam()
        cX = [-self.__param.maxX,0, self.__param.maxX]
        cA = [-self.__param.maxDegree, 0 , self.__param.maxDegree]
        cAD = [-1,0, 1]
        relativeVectors = []
        for w in cX :
            #for x in cXD :
            for y in cA :
                for z in cAD :
                    relativeVectors.append([w,y,z])

        basisRBFD = ( len(cX) * len(cA) * len(cAD) +1 ) * 2
        
        self.dimension = basisRBFD
        self.__relativeVectors = relativeVectors
        #dummy weight for testing
        self.dummyWeight = np.matrix( np.random.rand(self.dimension).reshape(self.dimension,1)-0.5)
        
    
    def basis(self,state,action):
        D = self.dimension
        state = [state[0],state[2],state[3]]
        temp = np.zeros((D,1))
        
        if(action == 0):
            base = 0
        else:
            base = D/2
        count = int(base)
        
        temp[count,0] = 1
        count +=1
        
        dist = (np.matrix(self.__relativeVectors) - np.matrix(state)).tolist()
        xValue = np.linalg.norm( dist, axis = 1)
        yValue = np.exp( -(xValue * xValue) )
        n=len(yValue)
        temp[count:(count + n),0] = yValue
                     
        return(temp)

#class CartPoleRBFState:
#    def __init__(self):
#        self.__param = param.CartPoleParam()
#        cX = [-self.__param.maxX,0, self.__param.maxX]
#        cA = [-self.__param.maxDegree, 0 , self.__param.maxDegree]
#        cAD = [-1,0, 1]
#        relativeVectors = []
#        for w in cX :
#            #for x in cXD :
#            for y in cA :
#                for z in cAD :
#                    relativeVectors.append([w,y,z])
#        basisSimpleD = 8
#        basisRBFD = ( len(cX) * len(cA) * len(cAD) +1 ) * 2
#        
#        self.__basisSimpleD = basisSimpleD
#        self.__basisRBFD = basisRBFD
#        self.__relativeVectors = relativeVectors
#        #dummy weight for testing
#        self.dummyWeightSimple = np.matrix( np.random.rand(self.__basisSimpleD).reshape(self.__basisSimpleD,1)-0.5)
#        self.dummyWeightRBF = np.matrix( np.random.rand(self.__basisRBFD).reshape(self.__basisRBFD,1)-0.5)
#        
#    def basisSimple(self,state, action): #best
#        D = self.__basisSimpleD
#        phi = np.zeros((D,1))
#        if(action == 0):
#            
#            phi[0,0]= state[0]/self.__param.maxX
#            phi[1,0]= state[1]/self.__param.maxX
#            phi[2,0]= state[2]/self.__param.maxDegree
#            phi[3,0]= state[3]/self.__param.maxDegree
#        else:
#            phi[4,0]= state[0]/self.__param.maxX
#            phi[5,0]= state[1]/self.__param.maxX
#            phi[6,0]= state[2]/self.__param.maxDegree
#            phi[7,0]= state[3]/self.__param.maxDegree
#        return(phi)
#    
#    def basisRBF(self,state,action):
#        D = self.__basisRBFD
#        state = [state[0],state[2],state[3]]
#        temp = np.zeros((D,1))
#        
#        if(action == 0):
#            base = 0
#        else:
#            base = D/2
#        count = int(base)
#        
#        temp[count,0] = 1
#        count +=1
#        
#        dist = (np.matrix(self.__relativeVectors) - np.matrix(state)).tolist()
#        xValue = np.linalg.norm( dist, axis = 1)
#        yValue = np.exp( -(xValue * xValue) )
#        n=len(yValue)
#        temp[count:(count + n),0] = yValue
#                     
#        return(temp)
