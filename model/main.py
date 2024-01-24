import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib



def create_model(data):
    y=data.diagnosis#target varaible
    X=data.drop(['diagnosis'], axis=1)
    
    #split the data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    
    #scale the data
    scaler=StandardScaler()
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    #train
    lr_model=LogisticRegression()
    lr_model.fit(X_train,y_train)
    
    #test
    y_pred=lr_model.predict(X_test)
    accuracy= accuracy_score(y_test,y_pred)
    print('Accuracy of our model:', accuracy_score)
    print('Classification report:', classification_report(y_test,y_pred))
    
    return lr_model, scaler

    

def get_clean_data():
    data=pd.read_csv(r'C:\Users\hp\Victor-Files\Supervised_Learning\Cancer-Prediction\data\data.csv')
    
    data=data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    
    return data

def main():
    data= get_clean_data()
    
    lr_model, scaler=create_model(data)

#save the model to a file in binary format
    joblib.dump(lr_model,'model/model.joblib')
#save the sclaer to a file in binary format 
    joblib.dump(scaler,'model/scaler.joblib')
    
    
if __name__ == '__main__':
    main()
    

