import numpy as np
import openturns as ot
import matplotlib.pyplot as plt

class RegressionTree(object):
    """
    Class RegressionTree
    """
    def __init__(self, inputSample, outputSample, exploredDimensionsRatio):
        # set attribut
        self.inputSample = inputSample
        self.outputSample = outputSample
        self.inputDimension = self.inputSample.getDimension()
        self.cellList = [ot.Interval([-1],[1])]
        self.minNodeSize = 5
        self.exploredDimensionsRatio = exploredDimensionsRatio
        self.testedDimSize = int(np.ceil(self.inputDimension/self.exploredDimensionsRatio))
        self.cellProbabilities = []
        self.classes = ot.Point(np.unique(self.outputSample))
        self.nClasses = self.classes.getSize()

    def fit(self):
        """
        Training method
        """
        
        ok = 1
        nullCounter = 0

        while ok:
            cell = self.cellList[0]
            contained_bool = cell.contains(self.inputSample)

            if sum(contained_bool) < self.minNodeSize*2:
				#  We move the cell to the end of the list
                temp = self.cellList[0]
                self.cellList.remove(self.cellList[0])
                self.cellList.append(temp)
                nullCounter += 1
            else:

                contained_indices = [idx for idx, v in enumerate(contained_bool) if v]
                contained_input = self.inputSample[contained_indices]
                selected_dims = np.random.choice(np.arange(self.inputDimension),self.testedDimSize, replace = False)
                optVal = 0.
                for dim in selected_dims:
                    for i in range(contained_input.getSize()-1):
                        if contained_input.getMarginal(int(dim))[i] == contained_input.getMarginal(int(dim))[i+1]:
                            optVal = 0
                        else:
                            x_cut =  (contained_input.getMarginal(int(dim))[i]+contained_input.getMarginal(int(dim))[i+1])/2
                            crit = self.computeCriterion(cell,int(dim),x_cut[0])
                            if crit > optVal:
                                optVal = crit
                                optCut = [dim,x_cut[0]]
   			             
                if optVal == 0:
   					#  We move the cell to the end of the list
                   temp = self.cellList[0]
                   self.cellList.remove(self.cellList[0])
                   self.cellList.append(temp)
                   nullCounter += 1
                   
                else:
                   lowerbound_left = cell.getLowerBound()
                   lowerbound_right = cell.getLowerBound()
                   upperbound_left = cell.getUpperBound()
                   upperbound_right = cell.getUpperBound()
 		
                   upperbound_left[optCut[0]] = optCut[1]
                   lowerbound_right[optCut[0]] = optCut[1]
 			        
                   newcell_left = ot.Interval(lowerbound_left, upperbound_left)
                   newcell_right = ot.Interval(lowerbound_right, upperbound_right)
                   self.cellList.remove(self.cellList[0])
                   self.cellList.append(newcell_left)
                   self.cellList.append(newcell_right)
                   
                   nullCounter = 0   
                
            if nullCounter == len(self.cellList):
                ok = 0
                for cell in self.cellList:
                    contained_bool = cell.contains(self.inputSample)
                    contained_indices = [idx for idx, v in enumerate(contained_bool) if v]
                    contained_output = self.outputSample[contained_indices]
                    n_contained = len(contained_indices)
                    classProbabilities = ot.Point(self.nClasses)
                    for i in range(self.nClasses):
                        prob = np.count_nonzero(np.array(contained_output)==self.classes[i])/n_contained
                        classProbabilities[i] = prob
                
                    self.cellProbabilities.append(classProbabilities)


    def computeCriterion(self, cell, dim_cut, coord_cut):
        """
        Computes the criterion associated to a given cut (location + dimension)
        """
        dim_cut = int(dim_cut)
        contained_bool = cell.contains(self.inputSample)
        contained_indices = [idx for idx, v in enumerate(contained_bool) if v]
        n_contained = len(contained_indices)

        lowerbound_left = cell.getLowerBound()
        lowerbound_right = cell.getLowerBound()
        upperbound_left = cell.getUpperBound()
        upperbound_right = cell.getUpperBound()

        upperbound_left[dim_cut] = coord_cut
        lowerbound_right[dim_cut] = coord_cut
        
        newcell_left = ot.Interval(lowerbound_left, upperbound_left)
        newcell_right = ot.Interval(lowerbound_right, upperbound_right)

        contained_bool_left = newcell_left.contains(self.inputSample)
        contained_bool_right = newcell_right.contains(self.inputSample)
        contained_indices_left = [idx for idx, v in enumerate(contained_bool_left) if v]
        contained_indices_right = [idx for idx, v in enumerate(contained_bool_right) if v]
        n_contained_left = len(contained_indices_left)
        n_contained_right = len(contained_indices_right)

        
        contained_output = self.outputSample[contained_indices]
        contained_output_left = self.outputSample[contained_indices_left]
        contained_output_right = self.outputSample[contained_indices_right]                        

        if(sum(contained_bool_left) < self.minNodeSize or sum(contained_bool_right) < self.minNodeSize):
            criterion = 0

        else :
            criterion_upper = 0
            criterion_left = 0
            criterion_right = 0
            for i in range(self.nClasses):
                prob_k_upper = np.count_nonzero(np.array(contained_output)==self.classes[i])/n_contained
                prob_k_left = np.count_nonzero(np.array(contained_output_left)==self.classes[i])/n_contained_left
                prob_k_right = np.count_nonzero(np.array(contained_output_right)==self.classes[i])/n_contained_right
                
                criterion_upper += prob_k_upper*(1-prob_k_upper)
                criterion_left += prob_k_left*(1-prob_k_left)
                criterion_right += prob_k_right*(1-prob_k_right)
            

            # criterion = criterion_upper - n_contained_left/n_contained*criterion_left - n_contained_right/n_contained*criterion_right
            criterion = criterion_upper - criterion_left - criterion_right

        return criterion
    
    
    def computePrediction(self,predictInput):
        """
        Computes the regression tree prediction
        """
        predictOutput = ot.Sample(predictInput.getSize(),self.nClasses)
        for i in range(len(self.cellList)):
            contained_bool = self.cellList[i].contains(predictInput)
            contained_indices = [idx for idx, v in enumerate(contained_bool) if v]
            predictOutput[contained_indices] = [self.cellProbabilities[i]]*len(contained_indices)
        return np.argmax(predictOutput, axis = 1)
 
    
dist = ot.ComposedDistribution([ot.Uniform()])
inputSample = dist.getSample(50)
# inputSample.add(inputSample[-1])
# f = ot.SymbolicFunction(['x1'],['(x1*3)^2'])

def f(x):
    
    X = x[0]
    X+= ot.Normal(0,0.1).getSample(1)[0][0]
    if X < -0.3:
        y = 0
    elif X >= -0.3 and X< 0.4:
        y = 1
    else:
        y = 2
    return [y]

f = ot.PythonFunction(1,1,f)

outputSample = f(inputSample)

# bootstrapIndices = ot.BootstrapExperiment_GenerateSelection(100,100)
# inputSample = inputSample[bootstrapIndices]
# outputSample = outputSample[bootstrapIndices]
            
tree = RegressionTree(inputSample, outputSample, 3)

tree.fit()

bds = []
for cell in tree.cellList:
    # print(sum(cell.contains(inputSample)))
    bds.append(cell.getLowerBound()[0])
    
  
plt.close('all')
plt.figure()
plt.plot(bds,-np.ones(len(bds)),'*')
plt.plot([-0.3,0.4],[-1,-1],'o')
plt.plot(inputSample[:,0],outputSample,'r*')


pred = tree.computePrediction(inputSample[:,0])
plt.plot(inputSample[:,0],pred,'b*')

print(pred)
