import numpy as np
import scipy.stats as st


def entropy(attribute_data):

    _, val_freqs = np.unique(attribute_data, return_counts=True)
    # probabilities for each unique attribute value
    val_probs = val_freqs / len(attribute_data)
    return -val_probs.dot(np.log(val_probs))


def info_gain(attribute_data, labels):

    attr_val_counts = get_count_dict(attribute_data)
    total_count = len(labels)
    EA = 0.0
    for attr_val, attr_val_count in attr_val_counts.items():
        EA += attr_val_count * entropy(labels[attribute_data == attr_val])
    
    # Issue #1: Take entropy/information on global labels not on attribute data
    return entropy(labels) - EA / total_count


def get_count_dict(data):

    data_values, data_freqs = np.unique(data, return_counts=True)
    return dict(zip(data_values, data_freqs))


def hypothesis_test(attribute_data, labels, p_threshold=None, return_p_value=False):
    
    
    label_counts = get_count_dict(labels)
   
    attr_val_counts = get_count_dict(attribute_data)
   
    total_count = len(labels)
    
   
    k = len(label_counts)
  
    m = len(attr_val_counts)
    
    statistic = 0.0
    for attr_val, attr_val_count in attr_val_counts.items():
        attr_val_ratio = attr_val_count / total_count
        
        label_counts_attr_val = get_count_dict(labels[attribute_data == attr_val])
        for label_attr_val, label_count_attr_val in label_counts_attr_val.items():
            
            exp_label_count_attr_val = attr_val_ratio * label_counts[label_attr_val]
           
            statistic += (label_count_attr_val - exp_label_count_attr_val)**2 / exp_label_count_attr_val

# Calculate the p value from the chi-square distribution CDF
p_value = 1 - st.chi2.cdf(statistic, df=(m-1)*(k-1))

if return_p_value:
    return p_value < p_threshold, p_value
    else:
        return p_value < p_threshold


# Main decision tree class. There'll be one instance of the class per node.
class DecisionTree:
    
    label = None
    # Split attribute for the children
    attribute = None
    
    attribute_value = None
    # A list of child nodes (DecisionTree)
    children = None
    
    p_value = None
    
    p_threshold = None
    parent = None
    
    level = None
    # max depth, for pruning
    max_level = 10000000
    
    def __init__(self, data, labels, attributes, fitness_func=info_gain, value=None, parent=None, p_threshold=1.0, max_level=None, old_level=0):
        
        self.level = old_level + 1
        self.p_threshold = p_threshold
        
        if max_level is not None:
            self.max_level = max_level
        
        if value is not None:
            self.attribute_value = value
        
        if parent is not None:
            self.parent = parent

        if data.size == 0 or not attributes or self.level == self.max_level:
            try:
                # self.label = st.mode(labels)[0][0][0]
                self.label = st.mode(labels)[0][0]
            except:
                self.label = labels[len(labels) - 1]
            return
        
        # If labels are all the same, set label and return
        if np.all(labels[:] == labels[0]):
            self.label = labels[0]
            return
        

        examples_all_same = True
        for i in range(1, data.shape[0]):
            for j in range(data.shape[1]):
                if data[0, j] != data[i, j]:
                    examples_all_same = False
                    break
            if not examples_all_same:
                break
        if examples_all_same:
            # Choose the last label
            self.label = labels[len(labels) - 1]
            return
        
        # Build the tree by splitting the data and adding child trees
        self.build(data, labels, attributes, fitness_func)
        return
    
    def __repr__(self):
        if self.children is None:
            return "x[{0}]={1}, y={2}".format(self.parent.attribute, self.attribute_value, self.label)
        else:
            if self.parent is not None:
                return "x[{0}]={1}, p={2}".format(self.parent.attribute, self.attribute_value, self.p_value)
            else:
                return "p={0}".format(self.p_value)

def build(self, data, labels, attributes, fitness_func):

            self.choose_best_attribute(data, labels, attributes, fitness_func)
            best_attribute_column = attributes.index(self.attribute)
            # Attribute data is the single column with attribute values for the best attribute
            attribute_data = data[:, best_attribute_column]
            
            # Prune if hypothesis test fails
            no_prune, self.p_value = hypothesis_test(attribute_data, labels, return_p_value=True, p_threshold=self.p_threshold)
            
            if not no_prune:
            # The try-return is probably not required here and above
            try:
                self.label = st.mode(labels)[0][0]
            except:
                self.label = labels[len(labels) - 1]
return
    
    # The child trees will be passed data for all attributes except the split attribute
    child_attributes = attributes[:]
    child_attributes.remove(self.attribute)
    
    self.children = []
        for val in np.unique(attribute_data):
            # Create children for data where the split attribute == val for each unique value for the attribute
            child_data = np.delete(data[attribute_data == val,:], best_attribute_column,1)
            child_labels = labels[attribute_data == val]
            self.children.append(DecisionTree(child_data, child_labels, child_attributes, value=val, parent=self,
                                              old_level=self.level, max_level=self.max_level))

def choose_best_attribute(self, data, labels, attributes, fitness):

            best_gain = float('-inf')
            for attribute in attributes:
            attribute_data = data[:, attributes.index(attribute)]
            gain = fitness(attribute_data, labels)
            if gain > best_gain:
                best_gain = gain
                self.attribute = attribute
                    return

def classify(self, data):

            if data.size == 0:
            return
                
                # If we're down to one record then convert it back to a 2-D array
                if len(data.shape) == 1:
                    data = np.reshape(data, (1,len(data)))
                        
                        if self.children is None:
                            # If we're at the bottom of the tree then return the labels for all records as the tree node label
                            labels = np.ones(len(data)) * self.label
                            return labels
                                
                                labels = np.zeros(len(data))
                                
                                for child in self.children:
                                    # Get the array indexes where the split attibute value  = child attribute value
                                    child_attr_val_idx = data[:,self.attribute] == child.attribute_value
                                    # pass the array subsets to child trees for classification
                                        labels[child_attr_val_idx] = child.classify(data[child_attr_val_idx])
                                            
                                            return labels
