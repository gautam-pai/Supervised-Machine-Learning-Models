#import ImportLibraries
from ImportLibraries import *
from DataCleaning import Standardize

df_std=Standardize('cs-training.csv')

predictor_var=df_std[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','TotalPastDueDays']]
response_var=df_std['SeriousDlqin2yrs']


predictor_train,predictor_test,response_train,response_test = train_test_split(predictor_var,response_var,test_size=0.3,random_state=0)

log_regression=LogisticRegression(max_iter=2000)
log_regression.fit(predictor_train,response_train)

response_pred=log_regression.predict(predictor_test)

# Confusion Matrix

print(metrics.confusion_matrix(response_test,response_pred))

#Accuracy
print("Accuracy in Training Dataset=",metrics.accuracy_score(response_test,response_pred))