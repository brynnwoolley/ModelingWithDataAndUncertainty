"""
Random Forest Lab

Brynn :)
Section
Date
"""
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        if sample[self.column] >= self.value:
            return True
        else:
            return False
        
    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # initialize
    left, right = [], []
    n = len(data[0])
    # iterate
    for i in data:
        if question.match(i):
            left.append(i)
        else:
            right.append(i)

    left = np.array(left).reshape(-1, data.shape[1])
    right = np.array(right).reshape(-1, data.shape[1])
     
    return left, right

# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    # initialize 
    opt_gain, opt_question = 0, None
    features = feature_names[:-1]
    n = len(features)

    # problem 6
    if random_subset:
        n_sqrt = int(np.floor(np.sqrt(n)))
        indices = np.random.randint(low=0, high=len(features), size=n_sqrt)

    for i in range(n):
        if random_subset and i not in indices:
            continue
        unique_vals = list(set(data[:,i]))
        for val in unique_vals:
            question = Question(column=i,value=val,feature_names=feature_names)
            left, right = partition(data,question)
            if num_rows(left) < min_samples_leaf or num_rows(right) < min_samples_leaf:
                continue
            ig = info_gain(data, left,right)
            if ig > opt_gain: # update gain
                opt_gain = ig
                opt_question = question

    if opt_question == None:
        print('No valid split could be found due to the minimum leaf size constraint.')
    
    return opt_gain, opt_question

# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        # initialize
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        # initialize
        self.question = question
        self.left = left_branch
        self.right = right_branch

# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    # base case: check # of rows is less than 2x min_samples_leaf
    if num_rows(data) < 2*min_samples_leaf:
        return Leaf(data)
    # base case: all samples are the same class
    elif len(np.unique(data[:, -1])) == 1:
        return Leaf(data)
    # base case: no more features to split on
    elif len(feature_names) == 0:
        return Leaf(data)
    # base case: tree has reached the maximum depth
    elif current_depth == max_depth:
        return Leaf(data)
    else:
        # find optimal gain & question for splitting
        opt_gain, opt_question = find_best_split(data, feature_names, min_samples_leaf, random_subset)
    
        # now if node is not leaf it's a decision node
        left, right = partition(data, opt_question)
        left_branch = build_tree(left, feature_names, min_samples_leaf, max_depth,
                                current_depth=current_depth+1, random_subset=random_subset)
        
        right_branch = build_tree(right, feature_names, min_samples_leaf, max_depth,
                                current_depth=current_depth+1, random_subset=random_subset)

        return Decision_Node(opt_question, left_branch, right_branch)

# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # check if node is a leaf
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)
    
    # else if node is not a leaf
    else:
        if my_tree.question.match(sample):
            return predict_tree(sample, my_tree.left)
        else:
            return predict_tree(sample, my_tree.right)

def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct_predictions = 0   # start with zero

    for sample in dataset:
        # check if it predicts correctly (int(True) = 1 & int(False) = 0)
        correct_predictions += int(predict_tree(sample, my_tree) == sample[-1])
    # return proportion of correctly classified data
    return correct_predictions/len(dataset)


# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    # initialize 
    predictions = []
    for tree in forest:
        # add prediction
        predictions.append(predict_tree(sample, tree))
    # return label predicted by the majority of the trees.
    return max(predictions, key=predictions.count)


def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    correct_predictions = 0   # start with zero

    for sample in dataset:
        # store label
        label = sample[-1]
        # check if it predicts correctly (int(True) = 1 & int(False) = 0)
        correct_predictions += int(predict_forest(sample, forest) == sample[-1])
        correct_predictions += int(predict_forest(sample, forest) == label)

    # return proportion of correctly classified data
    return correct_predictions/len(dataset)
    
# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    # read in the data and remove the first column
    parkinsons_data = np.loadtxt('parkinsons.csv',dtype=str,delimiter=',')
    parkinsons_features = np.loadtxt('parkinsons_features.csv', dtype=str, delimiter=',')
    parkinsons_data = parkinsons_data[:, 1:]


## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if not hasattr(my_tree, "question"):#isinstance(my_tree, leaf_class):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

def test(problem=0):
    """     Problem 1     """
    # Load in the data
    animals = np.loadtxt('animals.csv', delimiter=',')
    # Load in feature names
    features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
    # Load in sample names
    names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)
    
    prob = 1 if problem == 0 else problem
    if prob == 1:    
        # initialize question and test partition function
        question = Question(column=1, value=3, feature_names=features)
        left, right = partition(animals, question)
        print(len(left), len(right))
        print(f'{62==len(left)}, {38==len(right)}\n')
        #62 38
        question = Question(column=1, value=75, feature_names=features)
        left, right = partition(animals, question)
        print(len(left), len(right))
        #0 100 
        print(f'{0==len(left)}, {100==len(right)}\n')

    """     Problem 2     """
    prob = 2 if problem == 0 else problem
    if prob == 2:
        print(find_best_split(animals, features) )
        #(0.12259833679833687, Is # legs/tentacles >= 2.0?)
        print(str(find_best_split(animals, features))=="(0.12259833679833687, Is # legs/tentacles >= 2.0?)")
    
    """     Problem 4     """
    prob = 4 if problem == 0 else problem
    if prob == 4:
        my_tree = build_tree(animals, features)
        # draw_tree(my_tree)
    """     Problem 5     """
    prob = 5 if problem == 0 else problem
    if prob == 5:
        shuffled = np.copy(animals)
        np.random.shuffle(shuffled)
        train = shuffled[:int(len(shuffled) * 0.8)]
        test = shuffled[int(len(shuffled) * 0.8):]
        my_tree = build_tree(train, features)
        print(analyze_tree(test, my_tree))


