import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
      
      shifted = z - np.max(z) #to prevent overflow
      exp = np.exp(shifted) #defining shited exp

      #constructing softmax formula 

      softmax  = np.round( (exp/(np.sum(exp))), 4)
      return softmax 