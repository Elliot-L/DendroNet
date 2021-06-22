
from build_parent_child_mat import build_pc_mat
from parse_patric_tree import load_tree_and_leaves
import numpy as np
from queue import Queue
import os


new_pc, new_topo_order, new_leaves = build_pc_mat(genome_file='data_files/genome_lineage.csv', label_file='data_files/betalactam_firmicutes_samples.csv')

print(new_pc[6])

data_tree, leaves = load_tree_and_leaves(os.path.join('data_files', 'patric_tree_storage', 'betalactam'))

q = Queue(maxsize=0)
topo_order = []
q.put(data_tree)  # inputing the root in the queue
while not q.empty():
    curr = q.get()
    topo_order.append(curr)
    if len(curr.descendants) > 0:
        for des in curr.descendants:
            q.put(des)

parent_child = np.zeros(shape=(len(topo_order), len(topo_order)), dtype=np.int)

mapping = []

X = []
y = []
feature_index = 0

# Filling the X matrix and the y vector with features and target values, respectively, from the leaves
for index, node in enumerate(topo_order):
    if node in leaves:
        y.append(node.y)
        X.append(node.x)
        mapping.append((feature_index, index))
        feature_index += 1
    for child in node.descendants:
        parent_child[index][topo_order.index(child)] = 1

print(parent_child[7])




