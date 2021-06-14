
from build_parent_child_mat import build_pc_mat
from parse_patric_tree import load_tree_and_leaves
from queue import Queue
import numpy as np

new_pc, new_topo_order, new_leaves = build_pc_mat(genome_file='data_files/genome_lineage.csv', label_file='data_files/erythromycin_firmicutes_samples.csv')

print("done with first")

old_root, old_leaves = load_tree_and_leaves("data_files/patric_tree_storage/erythromycin")

print("tree created")

q = Queue(maxsize=0)
topo_order = []
q.put(old_root)  # inputing the root in the queue
while not q.empty():
    curr = q.get()
    topo_order.append(curr)
    if len(curr.descendants) > 0:
        for des in curr.descendants:
            q.put(des)
c = 0
for node in topo_order:
    if node.name != 'root':
        for des in node.descendants:
            if new_pc[new_topo_order.index(node.name)][new_topo_order.index(des.name)] == 1:
                c += 1
            else:
                print("ERROR ERROR ERROR")
print(c)

parent_child = np.zeros(shape=(len(topo_order), len(topo_order)), dtype=np.int)

for index, node in enumerate(topo_order):
    for child in node.descendants:
        parent_child[index][topo_order.index(child)] = 1

print("done with second")

new_count = 0
for row in new_pc:
    for col in row:
        if col == 1:
            new_count += 1

old_count = 0

for row in parent_child:
    for col in row:
        if col == 1:
            old_count += 1

print(new_count)
print(old_count)




