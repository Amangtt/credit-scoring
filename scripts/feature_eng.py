import pandas as pd
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('./data/data.csv')
def per_transaction(df):
    df['Debit'] = df['Amount'].apply(lambda x: x if x > 0 else 0)  # Positive amounts
    df['Credit'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0)
    df= df.groupby('CustomerId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('Amount', 'count'),
        StdDevTransactionAmount=('Amount', 'std'),
        TotalDebit=('Debit', 'sum'),
        TotalCredit=('Credit', 'sum'),
        FirstTransaction=('TransactionStartTime', 'min'),  # Earliest transaction date
        LastTransaction=('TransactionStartTime', 'max'),
        PricingStrategyMode=('PricingStrategy', lambda x: x.mode()[0] if not x.mode().empty else None),
        FraudCount=('FraudResult', 'sum')

    ).reset_index()
    df['FirstTransaction']= pd.to_datetime(df['FirstTransaction'])
    df['start_year']= df['FirstTransaction'].dt.year
    df['start_month']= df['FirstTransaction'].dt.month
    df['start_day'] =df['FirstTransaction'].dt.day
    df['start_hour'] =df['FirstTransaction'].dt.hour

    df['LastTransaction']= pd.to_datetime(df['LastTransaction'])
    df['last_year']= df['LastTransaction'].dt.year
    df['last_month']= df['LastTransaction'].dt.month
    df['last_day'] =df['LastTransaction'].dt.day
    df['last_hour'] =df['LastTransaction'].dt.hour
    
    df=pd.get_dummies(df)
    scaler= StandardScaler()
    df_new= scaler.fit_transform(df)
    print(df_new)
per_transaction(df)



