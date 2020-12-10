# Factor Model

This Python library can be used to analyse the contribution of each factor to portfolio performance and risk, such as return, volatility, drawdown, etc. 

Factor risks are not random fluctuations of noise, instead, they are associated with exposure to specific factors.  Therefore, factor risks can not be diversified away. On the other hand, alphas are returns that are not related to known factors. Alpha risk (noise) can be reduced by increasing diversification. More strategies with positive alpha included in one portfolio, higher return to risk ratio. Alphas are usually portable and additive.

## Factors

Factors related to portfolio performance/risk

> MSCI factor categories are Volatility, Yield, Quality, Momentum, Value, Size. 
> [link](https://www.msci.com/factor-investing)

**1, 1D time-series factors**

No transformation.

Factors such as:

​	VIX, GDP (quarterly), CPI (quarterly), Market Return (SP500, RUSSELL), Macro Factors

**2, 2D cross-sectional time-series factors**

Factors are converted to 1D time-series using the method applied in Fama-French model, as described below.

Factors are ranked cross-sectionally and a hedge portfolio is constructed by long top 30%  and short bottom 30%. Performance of this hedge portfolio is used to represent the factor. As a result,

1. factors in various formats are normalized and market neutralized. return of such portfolio represents factor premium.
2. factors with different distributions and frequencies are represented by a return series at same scale. 
3. hedge portfolio can be observed and traded to add/hedge specific factor risks.
4. coefficients are relatively more stable. panel regression coefficients can be quite unstable given low information to noise ratio in most financial data.

Factors such as:

​	Market Cap, Book Value / Market Value, Stock returns/alphas, Beta, Short Interest (raw, days to cover), PE ratio, Option (Ratio, volatility), other fundamental factors in financial statements.


## Linear Model

    Return (t)=α (t)+β_1*Factor_1+β_2*Factor_2+...+ε(t)

Features:

	Colinearity:
	
		VIF (variance inflation factor, fitted on whole dataset) is used in this project with default threshold 10.
		
		Previous work handles this issue by removing factors if their correlation is above 0.7 which is a common way 
		but it will not capture multi-collinear factors. Alternatively, VIF can be used to test and remove factors with high VIF, 
		popular choice is VIF 5-10. Other options are PCA or step-wise regression, ridge regression, etc.
	
	Standardization:
	
	    Hedge portfolio returns generally follow similar distribution. Standardization is implemented using sklearn.preprocessing.StandardScaler.
	
	Model complexity:
	
	    factor importance

## Non-Linear Model (WIP)

    Return (t)=f(Factor_1+Factor_2+...)+ε(t)

* RF MDI 


