# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:45:53 2021

@author: ankok
"""

class DecisionTreeClassifier:

    def __init__(self, max_depth ,min_size = 10):
        self.max_depth = max_depth
        self.min_size = min_size
        
    def fit(self, X, y):
        self.tree = decisiontree(X,y,self.max_depth,self.min_size)
        
    def predict(self, X):
        predictions = list()
        for row in test:
            prediction = predict(self.tree, row)
            predictions.append(prediction)   
        return(predictions)
    
#beginning the algorithm    
def decisiontree(train, test, max_depth, min_size):
	 tree = build(train, max_depth, min_size)
	 return tree

#build tree
def build(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
    
#classification 
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

#test index and value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

#gini index formula
def gini_index(groups, classes):

	n_instances = float(sum([len(group) for group in groups]))

	gini = 0.0
	for group in groups:
		size = float(len(group))

		if size == 0:
			continue
		score = 0.0

		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p

		gini += (1.0 - score) * (size / n_instances)
	return gini

#for max output
def node_t(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

#tree
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])

	if not left or not right:
		node['left'] = node['right'] = node_t(left + right)
		return

	if depth >= max_depth:
		node['left'], node['right'] = node_t(left), node_t(right)
		return

	if len(left) <= min_size:
		node['left'] = node_t(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	
	if len(right) <= min_size:
		node['right'] = node_t(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)


#prediction left or right
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

#main
clf = DecisionTreeClassifier(max_depth= 5)
clf.fit(train,test)
pred = clf.predict(test)
true_val = [row[-1] for row in test]
print(pred)
