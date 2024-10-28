#import 1_ImportLibraries
from ImportLibraries import *

def Standardize():
    df=pd.read_csv('cs-training.csv')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['MonthlyIncome']=df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents']=df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    df.drop(df[df['age']<18].index,inplace=True)
    df.drop(df[df['age']>65].index,inplace=True)
    df.drop(df[df['MonthlyIncome']>1000000].index,inplace=True)
    df.drop(df[df["DebtRatio"]>23000].index,inplace=True)
    df.drop(df[df['RevolvingUtilizationOfUnsecuredLines']>15000].index,inplace=True)

    df_cleaned=df.copy()
    df_cleaned.reset_index(inplace= True)
    df_cleaned['TotalPastDueDays']=df_cleaned['NumberOfTimes90DaysLate']+df_cleaned['NumberOfTime30-59DaysPastDueNotWorse']+df_cleaned['NumberOfTime60-89DaysPastDueNotWorse']
    df_cleaned.drop(['NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse'],axis=1,inplace=True)

    x=df_cleaned['SeriousDlqin2yrs']
    scaler = StandardScaler()
    df_standard_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','TotalPastDueDays']]),columns=['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','TotalPastDueDays'])
    df_standard_scaled['SeriousDlqin2yrs']=x


    return df_standard_scaled