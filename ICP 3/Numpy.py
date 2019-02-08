import statistics
from statistics import mode
import numpy as np #we import numpy as np


randnums = np.random.randint(0,21,15)#create a variable names randnums and set it to np.random.rand
#here 0 is inclusive 21 is exclusive



print(randnums)# Printing out the entire list
print("Most repeated value is:",statistics.mode(randnums))