# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 13:43:04 2018

@author: mchen
"""
import pandas as pd

class TimeSeriesRolling():
    
    """
    create time series iterator on rolling basis
    
    Args:
    
        time_series (numpy.ndarray):  array of date object/datetime. e.g. df.index.value
        step (int): the step size of iteration, as well as the size of test_interval
        window (int): rolling window length
        min_periods (int): min window size
        lag (int): the leakage gap between train and predict. 1 for most cases
        
    Returns:
    
        train_interval, test_interval      
    
    """
    
    def __init__(self, 
                 time_series,
                 window,
                 min_periods,
                 step,
                 lag):
    
        self.time_series = time_series
        self.window = window
        self.min_periods = min_periods
        self.step  = step
        self.end = self.min_periods # end of interval in each iteration, starts with min_periods
        self.lag = lag
    
        return
        
    def __iter__(self):
        
        return self

    def next(self): # for version 3, use __next__
        
        if len(self.time_series)-self.end < self.lag:
            
            raise StopIteration
        
        else:        
    
            if self.end-self.window < 0:
                
                interval_length = self.end
            
            else:
            
                interval_length = self.window
            
            train_interval=self.time_series[self.end-interval_length : self.end]
            test_interval=self.time_series[self.end+self.lag-1 : self.end+self.lag-1+self.step]
            
            self.end = self.end+self.step
                    
            return train_interval, test_interval
        
if __name__=="__main__":
    
    pass
    
#    df_pool=pd.read_csv(r'\\farmnas\FARM\_PoolDump\SP500_CONSOLIDATED\LATEST\_poolIndicatorF.csv', index_col=0)
#    
#    ts_rolling=TimeSeriesRolling(df_pool.index,
#                                 window=120,
#                                 min_periods=60,
#                                 step=1,
#                                 lag=1)
#                                 
#    for i, (train_interval, test_interval) in  enumerate(ts_rolling):
#
#        print i
#        print 'train interval', train_interval[0], train_interval[-1], len(train_interval)                
#        print 'test interval', test_interval[0], test_interval[-1], len(test_interval) 
#        print 'lag', df_pool.index.tolist().index(test_interval[0]) - df_pool.index.tolist().index(train_interval[-1])
#        print '\n'

        
                                     
    
    