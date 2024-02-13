import pickle

def  load_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)