import numpy as np

#Author: Nick Steelman
#Date: 5/29/18
#gns126@gmail.com
#cleanestmink.com


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
