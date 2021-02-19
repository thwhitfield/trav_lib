import pandas as pd
import numpy as np

from trav_lib.data_prep import reduce_memory

def test_reduce_memory():

    df = pd.DataFrame({'ints':[1,2,3,4],'floats':[.1,.2,.3,.4],'strings':['a','b','c','d']})

    df2 = reduce_memory(df)
    
    assert df2['ints'].dtype == np.dtype('int8')
    assert df2['floats'].dtype == np.dtype('float32')
    assert df2['strings'].dtype == np.dtype('O')

    df = pd.DataFrame({'ints':[1,2,3,4],'floats':[.1,.2,.3,.4],'strings':['a','b','c','d']})

    df3 = reduce_memory(df, cat_cols = ['strings'])
    
    assert df3['ints'].dtype == np.dtype('int8')
    assert df3['floats'].dtype == np.dtype('float32')
    assert df3['strings'].dtype.name == 'category' 
    