import openturns as ot
import matplotlib.pyplot as plt
from RegressionTree import RegressionTree
from scipy import stats


class RandomForestClassificationAlgorithm(object):
    """
    Class RandomForestRegressionAlgorithm
    
    Defines a randomized forest of regression trees
    """
    def __init__(self, inputSample, outputSample ):
        # set attribut
        self.inputSample = inputSample
        self.outputSample = outputSample
        self.treeNumber = 20
        self.exploredDimensionsRatio = 3
        self.dataSetSize = inputSample.getSize()
        self.bootstrapTrainingData = True
        self.treeList = None
        
    def run(self):
        """
        Method to train the random forest
        """
        
        treeList = []    
        for i in range(self.treeNumber):
            if self.bootstrapTrainingData:
                bootstrapIndices = ot.BootstrapExperiment_GenerateSelection(self.dataSetSize, self.dataSetSize)
                inputSample = self.inputSample[bootstrapIndices]
                outputSample = self.outputSample[bootstrapIndices]
                
            else:
                inputSample = self.inputSample
                outputSample = self.outputSample
                
            tree = RegressionTree(inputSample, outputSample, self.exploredDimensionsRatio)
            tree.fit()
            treeList.append(tree)
        self.treeList = treeList
        
    def resetTreeList(self):
        """
        Method to reset the random forest
        """
        self.treeList = None

    def setBoostrapTrainingData(self,bootstrapTrainingData):
        """
        Boostrap training data accessor
        """
        self.bootstrapTrainingData = bootstrapTrainingData
        self.resetTreeList()

    def setTreeNUmber(self,treeNumber):
        """
        Number of trees accessor
        """
        self.treeNumber = treeNumber
        self.resetTreeList()
        
    def setExploredDimensionsRatio(self,exploredDimensionsRatio):
        """
        Ratio of explored cutting dimesnions accessor
        """
        self.exploredDimensionsRatio = exploredDimensionsRatio
        self.resetTreeList()

    def predict(self, predictInput):
        """
        Random forest prediction
        """
        predictOutput = ot.Sample(predictInput.getSize(),self.treeNumber)
        for i in self.treeNumber:
            predictOutput[:,i] = self.tree[i].computePrediction(predictInput)
        return predictOutput/self.treeNumbe
    
    def getResult(self):
        """
        Result accessor
        """
        return RandomForestResult(self)
        

class RandomForestResult(object):
    """
    Class RandomForestResult
    
    Result associated to the RandomForestAlgorithm class
    """
    
    def __init__(self,RandomForestAlgorithm):
        self.RandomForestAlgorithm = RandomForestAlgorithm
        
    def getMetaModel(self):
        """
        Meta model accessor
        """
        return self.RandomForestAlgorithm.predict
    
    def getTreelList(self):
        """
        Tree list accessor
        """    
        return self.RandomForestAlgorithm.treeList
    
    
    
dist = ot.ComposedDistribution([ot.Uniform()])
inputSample = dist.getSample(100)
f = ot.SymbolicFunction(['x1'],['(x1*3)^2'])
outputSample = f(inputSample)

forest = RandomForestClassificationAlgorithm(inputSample, outputSample)

self = forest

forest.run()

plt.figure()
plt.plot(inputSample[:,0],outputSample,'*')

pred = forest.predict(inputSample)
plt.plot(inputSample[:,0],pred,'*')

# pred = forest.treeList[0].predict(inputSample)
# plt.plot(inputSample[:,0],pred,'*')

# tree = RegressionTree(inputSample, outputSample)
# tree.fit()
# pred2 = tree.predict(inputSample[:,0])
# plt.plot(inputSample[:,0],pred2,'*')
