import numpy as np

#Author: Nick Steelman
#Date: 5/29/18
#gns126@gmail.com
#cleanestmink.com
def symm_matrix(confusion_matrix):
    '''An input confusion matrix is a directed graph, but if we want to easily find
    clusters in the graph it is easier if it is undirected. So we sum the weights
    in both directions between each vertex to get the undirected connection'''
    for i in range(len(confusion_matrix)):
        for j in range(i,len(confusion_matrix)):
            tot = confusion_matrix[i][j] + confusion_matrix[j][i]
            confusion_matrix[j][i],confusion_matrix[i][j] = tot/2, tot/2

def find_thresholds(confusion_matrix, step = .001):
    '''
    Find the number of groups found at a variety of thresholds
    '''
    num_groups = {}
    max_value = 1
    i = 0
    while i < max_value:
        groups = return_groups(confusion_matrix, i)
        if len(groups) not in num_groups:
            num_groups[len(groups)] = groups
        if len(groups) == len(confusion_matrix):
            break
        i += step
    return num_groups

def return_groups(confusion_matrix, threshold):
    '''The purpose of this function is to take in a confusion matrix and return
    the maximum groups of categories in which none of the members of a group has
    a confusion rate above threshold.'''
    groups = []
    visited = set()
    # Itertate through all values
    num_classes = len(confusion_matrix[0])
    for i in range(num_classes):
        if i not in visited:
            groups.append(search_neighbors(confusion_matrix,threshold,visited,i))
    return groups

def search_neighbors(confusion_matrix, threshold, visited, index):
    '''Return all reachable valules from a given vertex (if you think of the
    matrix as a graph)'''
    queue = []#I know this is not a real queue
    current_group = []
    # print(index)
    queue.append(index)
    num_classes = len(confusion_matrix[0])
    while len(queue):
        val = queue.pop(0)
        if val not in visited:
            visited.add(val)
            current_group.append(val)
            for j in range(num_classes):
                # print(val)
                # print(j)
                if confusion_matrix[val][j] > threshold and j not in visited:
                    queue.append(j)
    return current_group


if __name__ == "__main__":
    matrix = [
    [1,1,0,0,0],
    [1,1,0,0,0],
    [0,0,1,0,1],
    [0,0,0,1,0],
    [0,0,1,0,1]
    ]
    print(return_groups(matrix,0.5))
