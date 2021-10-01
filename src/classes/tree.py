import numpy as np
from classes.node import Node

class TreeClassifier():
    def __init__(self, minSamplesSplit, maxDepth):

            # initialize the root node
            self.root = None
            # stopping criterion
            self.minSamplesSplit = minSamplesSplit
            self.maxDepth = maxDepth

    # build tree recursively
    def BuildTree(self, dataset, curDepth = 0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        numSamples, numFeatures = np.shape(X)
        
        # split until stopping criterion is satisfied
        if numSamples >= self.minSamplesSplit and curDepth <= self.maxDepth:
            # get best split
            bestSplit = self.GetBestSplit(dataset, numSamples, numFeatures)
            # check if there is any information gain
            if bestSplit["info"] > 0:
                # left subtree + increment curDepth
                l_subtree = self.BuildTree(bestSplit["l_dataset"], curDepth + 1)
                # right subtree + increment curDepth
                r_subtree = self.BuildTree(bestSplit["r_dataset"], curDepth + 1)
                # return decision node
                return Node(bestSplit["featureIndex"], bestSplit["threshold"], 
                            l_subtree, r_subtree, bestSplit["info"])
        
        # compute leaf node
        leafVal = self.CalculateLeafVal(Y)
        return Node(val = leafVal)
    
    # compute best split using gini index to find highest info gain
    def GetBestSplit(self, dataset, numSamples, numFeatures):
        
        # dictionary to store the best split
        bestSplit = {}
        # maxInfo initialised to value less than all others
        maxInfo = -float("inf")
        
        # loop over all the features
        for featureIndex in range(numFeatures):
            featureVals = dataset[:, featureIndex]
            maxThresholds = np.unique(featureVals)
            # loop over all the feature values in the data
            for threshold in maxThresholds:
                # get current split
                l_dataset, r_dataset = self.Split(dataset, featureIndex, threshold)
                # check if children are present
                if len(l_dataset) > 0 and len(r_dataset) > 0:
                    data, left_data, right_data = dataset[:, -1], l_dataset[:, -1], r_dataset[:, -1]
                    # compute information gain based on Gini index
                    curInfo = self.InformationGain(data, left_data, right_data)
                    # update the best split if needed a higher info gain found
                    if curInfo > maxInfo:
                        bestSplit["featureIndex"] = featureIndex
                        bestSplit["threshold"] = threshold
                        bestSplit["l_dataset"] = l_dataset
                        bestSplit["r_dataset"] = r_dataset
                        bestSplit["info"] = curInfo
                        maxInfo = curInfo
                        
        # return best split based on highest info gain
        return bestSplit
    
    def Split(self, dataset, featureIndex, threshold):
        
        # splits on either side of the threshold (less than or equal to and greater than the threshold)
        l_dataset = np.array([row for row in dataset if row[featureIndex] <= threshold])
        r_dataset = np.array([row for row in dataset if row[featureIndex] > threshold])
        return l_dataset, r_dataset
    
    def InformationGain(self, parent, l_child, r_child):
        
        l_Weight = len(l_child) / len(parent)
        r_Weight = len(r_child) / len(parent)
        # calculate gain based on gini index
        gain = self.GiniIndex(parent) - (l_Weight * self.GiniIndex(l_child) + r_Weight * self.GiniIndex(r_child))
        return gain
    
    def GiniIndex(self, data):
        
        labels = np.unique(data)
        gini = 0
        for i in labels:
            p_i = len(data[data == i]) / len(data)
            gini += p_i**2
        # 1 - gini gives impurity
        return 1 - gini
        
    def CalculateLeafVal(self, Y):
        
        # return the most commonly occuring data point in Y
        Y = list(Y)
        return max(Y, key = Y.count)
    
    # print out the tree to console
    def PrintTree(self, tree = None, indent = " "):
        
        if not tree:
            tree = self.root

        # if leaf node
        if tree.val is not None:
            print(str(tree.val))

        else:
            print("is X_" + str(tree.featureIndex), "<=", str(tree.threshold) + "?", "info gain: " + str(tree.info))
            print("%sleft: " % (indent), end = "")
            self.PrintTree(tree.left, indent * 2)
            print("%sright: " % (indent), end = "")
            self.PrintTree(tree.right, indent * 2)
    
    def Fit(self, X, Y):
        # train the tree
        dataset = np.concatenate((X, Y), axis = 1)
        self.root = self.BuildTree(dataset)
    
    def Predict(self, X):
        
        predictions = [self.MakePrediction(x, self.root) for x in X]
        return predictions
    
    def MakePrediction(self, x, tree):
    
        if tree.val != None: return tree.val
        featureVal = x[tree.featureIndex]
        if featureVal <= tree.threshold:
            return self.MakePrediction(x, tree.left)
        else:
            return self.MakePrediction(x, tree.right)