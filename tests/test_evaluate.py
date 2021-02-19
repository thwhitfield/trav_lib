import pandas as pd
import numpy as np

from trav_lib.evaluate import get_confusion_matrix

def test_get_confusion_matrix():
    y_true = [0,0,0,1,1,1,1,1,1,1]
    y_pred = [0,1,1,0,0,0,1,1,1,1]
    matrix = get_confusion_matrix(y_true, y_pred)
    
    matrix2 = pd.DataFrame({0:[1,3],1:[2,4]})
    
    assert matrix.equals(matrix2)
