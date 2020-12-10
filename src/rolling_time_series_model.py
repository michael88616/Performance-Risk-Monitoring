# -*- coding: utf-8 -*-
"""
Created on Mon Dec 03 13:43:04 2018

@author: mchen
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from TimeSeriesRolling import TimeSeriesRolling
import os

class RollingLinearRegression():
    
    """
    rolling linear regression
        
    Args:
        
        df_x (pandas dataframe): time series dataframe, T (samples) x N (features)
        df_y (pandas dataframe): time series dataframe, T x 1 
        standardized (bool): stardardize the feature matrix
    
    Attributes:
        
        df_predict (pandas dataframe): prediction series
        df_coef (pandas dataframe): rolling fit coef time series 
        df_feature_imp (pandas dataframe): normalized feature importance series from RandomForestRegressor rolling fit 
        score (float): R^2 of rolling prediction against actual target
        
    """
    
    def __init__(self, df_x, 
                 df_y, 
                 standardized=False):
    
        self.df_x=df_x
        self.df_y=df_y
        self.standardized=standardized
        
        if self.standardized:
            
            print "data will be standardized"
            self.scaler = StandardScaler(with_mean=True, with_std=True)

        if list(self.df_x.index)!=list(self.df_y.index):
                               
            self.index_=list((self.df_x.index.intersection(self.df_y.index)).sort_values())
            self.df_x=self.df_x.ix[self.index_]
            self.df_y=self.df_y.ix[self.index_]
        
        else:
        
            self.index_=list(self.df_x.index)
        
        self.list_predictors=df_x.columns.tolist()
        self.target=df_y.columns
        
        return
        
    def calculate_vif(self, x):
                
        """
        measure collinearity 
        
        Args:
        
            x (numpy.ndarray): feature/predictor matrix
        
        Return:
        
            df_vif (pandas dataframe): vif,  index: feature names, columns: vif
        """        
            
        list_vif=[self.variance_inflation_factor(x, i).round(4) for i in range(len(self.list_predictors))]    
            
        df_vif=pd.DataFrame(index = self.list_predictors, 
                            columns = ['vif'], 
                            data = list_vif)
                   
        return df_vif
    
    def calculate_pval(self, x, y):
        
        """
        measure p-value 
        
        f_regression implementation can be found at 
        http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm

        Args:
        
            x (numpy.ndarray): actual
            y (numpy.ndarray): predict  
            
        Return:
        
            df_pval (pandas dataframe): p value, index: feature names, columns: pval
        """                
        
        _, pval=f_regression(x, y, center=True)
        df_pval=pd.DataFrame(index = self.list_predictors, 
                             columns = ['pval'], 
                             data = pval.round(4))
        return df_pval
    
    def global_fit(self):
        
        """
        fit the entire data set
        
        Return:

              df_stats_summary (pandas dataframe): summary table
        """
        
        x_=self.df_x.as_matrix()
        y_=self.df_y.as_matrix()   
        
        if self.standardized:
            
            x_=self.scaler.fit_transform(x_)
    
        lr=LinearRegression(fit_intercept=True)
        lr.fit(x_, y_)      
                    
        lr_coefficients = np.concatenate([lr.intercept_, lr.coef_[0]], axis=0)   
        df_coef=pd.DataFrame(data=lr_coefficients.round(4), index=['intercept']+self.list_predictors, columns=['coef'])
        df_vif=self.calculate_vif(x_)
        df_pval=self.calculate_pval(x_, y_)
        
        model=RandomForestRegressor()
        model.fit(x_, y_)  
        df_feature_imp=pd.DataFrame(data=model.feature_importances_.round(4), index=self.list_predictors, columns=['feature_imp'])
        df_stats_summary=df_coef.join(df_vif).join(df_pval).join(df_feature_imp)
        
        return df_stats_summary
    
    def rolling_fit(self,
                    window, 
                    min_periods,
                    step,
                    lag):       
        
        """
        rolling fit
        
        window (int): rolling window
        min_periods (int): minimal periods
        step (int): rolling step size     
        lag (int): number of days to lag to remove time machine in multiple-HD 
                     prediction
        """
        
        time_series=self.df_x.index.values
        
        ts_rolling=TimeSeriesRolling(time_series,
                                     window=window,
                                     min_periods=min_periods,
                                     step=step,
                                     lag=lag)
                          
        self.df_predict=pd.DataFrame(index=self.index_, columns=['predict'])
        self.df_coef=pd.DataFrame()
        self.df_feature_imp=pd.DataFrame(index=self.index_, columns=self.list_predictors)
        
        print "\napplying linear rolling regressor\n"
        
        for idx, (train_interval, test_interval) in enumerate(ts_rolling):
                        
            if idx%100==0:
                
                print "processing", idx
                print "train interval", train_interval[0], train_interval[-1], len(train_interval)
                print "test interval", test_interval[0], test_interval[-1], len(test_interval)
            
            x_train=self.df_x.ix[train_interval].as_matrix()
            y_train_=self.df_y.ix[train_interval].as_matrix()   

            x_test=self.df_x.ix[test_interval].as_matrix()
            
            if self.standardized:
                
                x_train=self.scaler.fit_transform(x_train)
                x_test=self.scaler.fit_transform(x_test)
        
            lr=LinearRegression(fit_intercept=True)
            lr.fit(x_train, y_train_)      
                    
            lr_coefficients = np.append([lr.intercept_], [lr.coef_[0]])        
                        
            df_coef_temp=pd.DataFrame(index=test_interval, 
                                      columns=['intercept']+self.list_predictors, 
                                      data=lr_coefficients.reshape(1, -1)).fillna(method='ffill')
            self.df_coef=self.df_coef.append(df_coef_temp)
            
            df_predict_temp=pd.DataFrame(index=test_interval, columns=['predict'], data=lr.predict(x_test))
            self.df_predict=self.df_predict.append(df_predict_temp)
            
            # fit RF 
            model=RandomForestRegressor()
            model.fit(x_train, y_train_)  
            df_feature_imp_temp=pd.DataFrame(index=test_interval, 
                                             columns=self.list_predictors, 
                                             data= model.feature_importances_.reshape(1, -1)).fillna(method='ffill')
            self.df_feature_imp=self.df_feature_imp.append(df_feature_imp_temp)
            
        self.df_predict=self.df_predict.join(self.df_y, how='left').dropna().astype(float)
        self.score=r2_score(self.df_predict[self.target].as_matrix(), self.df_predict['predict'].as_matrix())
            
        return 
    
    def variance_inflation_factor(self, exog, exog_idx):
        
        """
        variance inflation factor, VIF, for one exogenous variable
        The variance inflation factor is a measure for the increase of the
        variance of the parameter estimates if an additional variable, given by
        exog_idx is added to the linear regression. It is a measure for
        multicollinearity of the design matrix, exog.
        One recommendation is that if VIF is greater than 5, then the explanatory
        variable given by exog_idx is highly collinear with the other explanatory
        variables, and the parameter estimates will have large standard errors
        because of this.
        Parameters
        
        Args:
        
            exog (pandas df, ndarray):
                design matrix with all explanatory variables, as for example used in
                regression
                
            exog_idx (int):
                index of the exogenous variable in the columns of exog

        Returns:

            vif (float): variance inflation factor
            
        """
        
        k_vars = exog.shape[1]
        
        if 'DataFrame' in str(type(exog)):
            
            exog=exog.as_matrix()
            
        assert ('ndarray' in str(type(exog)))   
        assert (k_vars>1)
        
        x_i = exog[:, exog_idx]
        mask = np.arange(k_vars) != exog_idx
        x_noti = exog[:, mask]
        
        lr = LinearRegression(fit_intercept=True)
        lr.fit(x_noti, x_i)  
        r_squared_i=r2_score(x_i, lr.predict(x_noti))
    
        vif = 1. / (1. - r_squared_i)
        
        return vif
        
if __name__=="__main__":
    
    pass
#    
#    if os.name=='nt':
#    
#        dir_farm='//farmnas/FARM/'
#    
#    else:
#        
#        dir_farm='//media/farmshare/'
#    
#    df_x=pd.read_csv(dir_farm+'Research/mchen/reports/20181128_risk_model/example/' \
#                '20190116_ML2/20190116.csv',index_col=0).dropna()
#
#    df_c2c=pd.read_csv(dir_farm+'_JIRA/PAP/PAP-23/Reports/2019-01-12/Reports/' \
#                       'ML2_FHD15_TH0.75_Scale_Beta1_Mcap4_BetaTarget0.2/20190112_000411_C2C_report_NET_Tab.csv',index_col=0)
#    df_y=pd.DataFrame(((df_c2c['Return WITHOUT_COST']-df_c2c['Benchmark Return CloseToClose'])/100).dropna())
#    df_y.columns=['relret']
#    
#    model=RollingLinearRegression(df_x, 
#                                  df_y, 
#                                  standardized=False)
#                                  
#    model.rolling_fit(window=120, 
#                      min_periods=60,
#                      step=1,
#                      lag=1)
        
    
    
    
    
    
    
    
    
    
    