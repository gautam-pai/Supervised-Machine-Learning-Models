#import 1_ImportLibraries
from ImportLibraries import *
#path=cs-training

def initial_cleaner(temp):
    temp.drop(temp[temp['age']<18].index,inplace=True)
    #temp.drop(temp[temp['age']>65].index,inplace=True)
    temp.drop(temp[temp['MonthlyIncome']>1000000].index,inplace=True)
    temp.drop(temp[temp["DebtRatio"]>23000].index,inplace=True)
    temp.drop(temp[temp['RevolvingUtilizationOfUnsecuredLines']>15000].index,inplace=True)
    return temp

def scaling(temp):
    scaler = StandardScaler()
    temp_scaled = pd.DataFrame(scaler.fit_transform(temp[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents','TotalPastDueDays']]),columns=['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents','TotalPastDueDays'])
    return temp_scaled

def Standardize(path):
    df=pd.read_csv(path)
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents']=df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    df_cleaned=initial_cleaner(df)
    df_cleaned.reset_index(inplace= True)
    x=df_cleaned['SeriousDlqin2yrs']
    df_cleaned['TotalPastDueDays']=df_cleaned['NumberOfTimes90DaysLate']+df_cleaned['NumberOfTime30-59DaysPastDueNotWorse']+df_cleaned['NumberOfTime60-89DaysPastDueNotWorse']
    df_cleaned.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse'],axis=1,inplace=True)

    df_standard_scaled=scaling(df_cleaned)

    df_standard_scaled['SeriousDlqin2yrs']=x


    return df_standard_scaled