class Node():
    def __init__(self, featureIndex = None, 
                 threshold = None, 
                 left = None, 
                 right = None, 
                 info = None, 
                 val = None):
      # for decision nodes
      self.featureIndex = featureIndex
      self.threshold = threshold
      self.left = left
      self.right = right
      self.info = info
      # value for leaf nodes
      self.val = val