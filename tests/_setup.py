import sys
sys.path.append('..')

'''
 - We can have a global Graph object, any operation occuring will add to it
 - During each forward pass the graph has to be rebuilt
 - After each backward pass the graph has to be reset and all memory should be freed, all tensors except Params should be destroyed
 - Tensor will only do the tensor work
 - Node will handle all the connections and stuff
 - Maybe a weakref to the node from the tensor for backward pass
'''