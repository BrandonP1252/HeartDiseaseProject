import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def setup_sample():
    filename = "Heart_Disease_Pred.csv"
    df = pd.read_csv(filename)
    return df

def setup_test():
    filename = "Heart_Disease_New_Sample.csv"
    filename2 = "Heart_Disease_Pred.csv"
    df1 = pd.read_csv(filename)
    df2 = pd.read_csv(filename2)
    df = df1.loc[df1.index.difference(df2.index)]
    return df

def blood_pressure(df):
    blood_pressure_df = df[["trestbps", "target"]]
    blood_pressure_df = blood_pressure_df.sort_values(["trestbps"], ascending=False)
    return blood_pressure_df


def display(df):
   df["HD"] = df["Heart Disease"].apply(lambda x:1 if x == "Presence" else 0)

   plt.plot(df["BP"], df["HD"], marker=".", label="Blood Pressure", linestyle="none")
   plt.legend(loc="upper left")
   plt.xlabel("Blood Pressure")
   plt.ylabel("Heart Disease")
   plt.show()

def reformat_pred():
    filename = "Heart_Disease_Pred.csv"
    df = pd.read_csv(filename)
    df = df.drop(['ca', 'thal'], axis=1)
    df = df.rename(columns={'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain type', 'trestbps': 'BP', 'chol': 'Cholesterol', 'fbs': 'FBS over 120', 'restecg': 'EKG results',
                            'thalach': 'Max HR', 'exang': 'Exercise angina', 'oldpeak': 'ST depression', 'slope': 'Slope of ST', 'target': 'Heart Disease'})
    df['Heart Disease'] = df['Heart Disease'].apply(lambda x:'Presence' if x == 1 else 'Absence')
    return df


setup_test()