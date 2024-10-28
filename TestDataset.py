#import ImportLibraries
from ImportLibraries import *
from DataCleaning import Standardize

df_std=Standardize('cs-training.csv')

predictor_var=df_std[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents','TotalPastDueDays']]
response_var=df_std['SeriousDlqin2yrs']


#### Fetch Testing File & Data Preparation
test_df=pd.read_csv('cs-test.csv')
test_df['TotalPastDueDays']=test_df['NumberOfTimes90DaysLate']+test_df['NumberOfTime30-59DaysPastDueNotWorse']+test_df['NumberOfTime60-89DaysPastDueNotWorse']
test_df.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse'],axis=1,inplace=True)
test_df.drop(['Unnamed: 0','SeriousDlqin2yrs'],axis=1,inplace=True)
test_df['NumberOfDependents']=test_df['NumberOfDependents'].fillna(test_df['NumberOfDependents'].median())
test_df['MonthlyIncome']=test_df['MonthlyIncome'].fillna(test_df['MonthlyIncome'].median())


#### Logistic Regression

log1=LogisticRegression(max_iter=2000)
log1.fit(predictor_var,response_var)
response_pred=log1.predict(test_df[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents','TotalPastDueDays']])


#### Comparing the prediction with probability 
prob=pd.read_csv('sampleEntry.csv')

def  encoding(x):
    if x>=0.5:
        return 1
    return 0
prob=(prob['Probability']).tolist()
prob=[encoding(x) for x in prob]
probability=pd.Series(prob)

#### COnfusion Matrix
print(metrics.confusion_matrix(probability,response_pred))

#### Accuracy check
print("Accuracy on Testing Dataset=",metrics.accuracy_score(probability,response_pred))