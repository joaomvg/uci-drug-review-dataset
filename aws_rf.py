import argparse
import pickle
import os

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

def data_load(data_dir):
    files=['cv_train.pkl','train_y.pkl']
    files=[os.path.join(data_dir,file) for file in files]
        
    data=[pickle.load(open(file,'rb')) for file in files]
     
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Forests')
    
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_depth', type=int)
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    cv_train,train_y=data_load(args.train)
    print('Data loaded successfully.')
    
    print('Initiating Random Forest Classifier')
    RFmodel = RandomForestClassifier(criterion="gini",n_estimators=args.n_estimators, max_depth=args.max_depth,random_state=36, verbose=1, n_jobs=-1)
    RFmodel.fit(cv_train, train_y)
    
    joblib.dump(RFmodel, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf