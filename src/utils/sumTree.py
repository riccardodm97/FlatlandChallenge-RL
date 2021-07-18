import numpy as np


# Smart way of storing data with weights for each experience 
class SumTree(object):
    
    data_pointer = 0

    def __init__(self, capacity):
        
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object) # Contains the experiences (so the size of data is capacity)
        self.n_entries = 0


    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (overwrite)
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1


    def update(self, tree_index, priority):
        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:    

            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change


    def get_leaf(self, v):
      
        parent_index = 0

        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # if we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else: # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node