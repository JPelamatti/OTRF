import numpy as np
import openturns as ot

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
        self.cellMeans = []
        

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
                # selected_dims = np.arange(self.inputDimension)
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
                    self.cellMeans.append(contained_output.computeMean())


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
        
        contained_output = self.outputSample[contained_indices]
        contained_output_left = self.outputSample[contained_indices_left]
        contained_output_right = self.outputSample[contained_indices_right]            


        if(sum(contained_bool_left) < self.minNodeSize or sum(contained_bool_right) < self.minNodeSize):
            criterion = 0
            
        else :
            criterionA = np.array(contained_output-contained_output.computeMean()[0])**2
            criterionB = np.array(contained_output_left-contained_output_left.computeMean()[0])**2
            criterionC = np.array(contained_output_right-contained_output_right.computeMean()[0])**2
            criterion = np.mean(criterionA)-1/n_contained*(np.sum(criterionB)+np.sum(criterionC))
                
        return criterion
    
    
    def computePrediction(self,predictInput):
        """
        Computes the regression tree prediction
        """
        predictOutput = ot.Sample(predictInput.getSize(),1)
        for i in range(len(self.cellList)):
            contained_bool = self.cellList[i].contains(predictInput)
            contained_indices = [idx for idx, v in enumerate(contained_bool) if v]
            predictOutput[contained_indices] = [self.cellMeans[i]]*len(contained_indices)
        return(predictOutput)
    
    
# dist = ot.ComposedDistribution([ot.Uniform()])
# inputSample = dist.getSample(100)
# inputSample.add(inputSample[-1])
# f = ot.SymbolicFunction(['x1'],['(x1*3)^2'])
# outputSample = f(inputSample)

# bootstrapIndices = ot.BootstrapExperiment_GenerateSelection(100,100)
# inputSample = inputSample[bootstrapIndices]
# outputSample = outputSample[bootstrapIndices]
            
# tree = RegressionTree(inputSample, outputSample)

# tree.fit()

# bds = []
# for cell in tree.cellList:
#     # print(sum(cell.contains(inputSample)))
#     bds.append(cell.getLowerBound()[0])
    
    
# plt.figure()
# plt.plot(bds,np.zeros(len(bds)),'*')
# plt.plot(inputSample[:,0],outputSample,'*')

# for cell in tree.cellList:
        
#     contained_bool = cell.contains(inputSample)
#     contained_indices = [idx*v for idx, v in enumerate(contained_bool) if v]
#     contained_output = outputSample[contained_indices]
#     contained_input = inputSample[contained_indices]
#     # plt.plot(contained_input.getMarginal(0),[contained_output.computeMean()]*contained_output.getSize(),'*')
#     plt.plot(inputSample[:,0],tree.predict(inputSample[:,0]),'*')

# test = dist.getSample(5)
# print(test)
# print(tree.predict(test))