#import ImportLibraries
from ImportLibraries import *
from DataCleaning import Standardize

df_std=Standardize('cs-training.csv')

predictor_var=df_std[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','TotalPastDueDays']]
response_var=df_std['SeriousDlqin2yrs']


predictor_train,predictor_test,response_train,response_test = train_test_split(predictor_var,response_var,test_size=0.3,random_state=0)




predictor_var=sm.add_constant(predictor_var)
model=sm.OLS(response_var,predictor_var).fit()
print(model.summary())

