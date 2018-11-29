import collections
import pandas as pd
import pdb
import math
import numpy as np
from scipy import stats

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def create_node(df):
    
    if len(df.iloc[:, -1].unique()) == 1:
        node = Node("")
        node.answer = df.iloc[:, -1].unique()[0]
        return node
    
    gain_ratios = []
    ent = stats.entropy(collections.Counter(df.iloc[:, -1]).values(), base=2)
    for col in xrange(len(df.columns) - 1):
        ig = ent
        iv = 0.0
        for val, cnt in collections.Counter(df.iloc[:, col]).iteritems():
            tmp = df.loc[df.iloc[:, col] == val]
            ig -= float(cnt)/len(df) * stats.entropy(collections.Counter(tmp.iloc[:, -1]).values(), base=2)
            iv -= float(cnt)/len(df) * np.log2(float(cnt)/len(df))
        gain_ratios.append(ig/iv)
    
    split = np.argmax(gain_ratios)
    items = df.iloc[:, split].unique()
    node = Node(df.columns[split])
    
    for x in range(len(items)):
        new_df = df.loc[df.iloc[:, split] == items[x]]
        del new_df[df.columns[split]]
        child = create_node(new_df)
        node.children.append((items[x], child))
    
    return node



def print_tree(node, level):
    if node.answer != "":
        print empty(level), node.answer
        return
        
    print empty(level), node.attribute
    
    for value, n in node.children:
        print empty(level + 1), value
        print_tree(n, level + 2)
        

df = pd.read_csv("./stock.data")
print(df)
print()

node = create_node(df)
print_tree(node, 0)

