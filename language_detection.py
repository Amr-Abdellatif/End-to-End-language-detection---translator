import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

"""this is loading the trained model on language detection"""
def load_model_and_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        saved_objects = pickle.load(f)

    model = saved_objects['model']
    cv = CountVectorizer(vocabulary=saved_objects['cv'].vocabulary_)
    le = LabelEncoder()
    le.classes_ = saved_objects['le'].classes_
    
    return model, cv, le