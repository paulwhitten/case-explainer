import os
import sys
import csv
import json
import argparse
import pickle
from datetime import datetime
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from pympler import asizeof

# gets the rows from a csv file as strings
def load_rows(file):
    row_count = 0
    rows = []
    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row_count > 0: # ignore the header
                rows.append(row)
            row_count += 1
    return rows

# gets the rows from a csv file as integers
def load_int_rows(file):
    file_rows = load_rows(file)
    rows = []
    for row in file_rows:
        int_r = []
        for v in  row:
            int_r.append(int(v))
        rows.append(int_r)
    return rows

# Obtains an array of euclidean distances to other data points
# point - the point of interest
# data - all of the other points
def get_distances(point, data):
    distances = []
    for d in data:
        distances.append(distance.euclidean(point, d))
    return distances

TRAINING_INDEX_FILE = "training_index.json"
TRAINING_INDEX_PICKLE_FILE = "training_index.pickle"
TRAINING_FILE = "train.csv"

class TrainingIndex:

    # train_rows - list of training rows
    # scaled_rows - list of scaled training rows
    # train_labels - list of training labels
    # row_to_indexes - dictionary that maps from each row as a string (key) to an array of indexes
    # row_to_distances - dictionary that goes from each row as a string (key) to an array of the euclidean distance to other elements
    # scaled_row_to_indexes - 
    # scaled_row_to_distance -
    
    def __init__(self, index_folder, training_folder):
        #self.index_file = os.path.join(index_folder, TRAINING_INDEX_FILE)
        self.index_pickle_file = os.path.join(index_folder, TRAINING_INDEX_PICKLE_FILE)
        self.training_folder = training_folder
        temp_time = datetime.now()
        self.distance_misses = 0
        self.toal_distance_compute_duration = temp_time - temp_time
        if not os.path.exists(self.index_pickle_file): # self.index_file
            self.build_index()
        else:
            start_time = datetime.now()
            with open(self.index_pickle_file, "rb") as infile:
                self.training_index = pickle.load(infile)
            # with open(self.index_file) as infile:
            #     self.training_index = json.load(infile)
            end_time = datetime.now()
            print("Deserialized index in:", end_time - start_time)
            
    
    def build_index(self):
        start_time = datetime.now()
        self.training_index = {}
        train_rows = load_int_rows(os.path.join(self.training_folder, "train.csv"))

        # remove the labels
        train_labels = []
        for r in train_rows:
            train_labels.append(r.pop())
        self.training_index["train_rows"] = train_rows
        self.training_index["train_labels"] = train_labels

        # scaled data
        scaler = StandardScaler()
        scaled_rows = scaler.fit_transform(train_rows).tolist()
        self.training_index["scaled_rows"] = scaled_rows

        # build row to indexes
        row_to_indices = {}
        scaled_row_to_indices = {}
        for ix, r in enumerate(train_rows):
            key = str(r)
            scaled_key = str(scaled_rows[ix])
            if key not in row_to_indices:
                row_to_indices[key] = [ix]
                scaled_row_to_indices[scaled_key] = [ix]
            else:
                row_to_indices[key].append(ix)
                scaled_row_to_indices[scaled_key].append(ix)
        self.training_index["row_to_indexes"] = row_to_indices
        self.training_index["scaled_row_to_indexes"] = scaled_row_to_indices

        # load the full data #TODO save space by removing duplicates
        full_training = load_rows(os.path.join(self.training_folder, "train_full.csv"))
        full_test = load_rows(os.path.join(self.training_folder, "test_full.csv"))
        training_origin = []
        for row in full_training:
            training_origin.append(row[:6])
        self.training_index["train_origin"] = training_origin
        test_origin = []
        for row in full_test:
            test_origin.append(row[:6])
        self.training_index["test_origin"] = test_origin


        # build distance indexes
        row_to_distances = {}
        scaled_row_to_distances = {}
        for ix, r in enumerate(train_rows):
            key = str(r)
            scaled_key = str(scaled_rows[ix])
            if key not in row_to_distances:
                row_distances = []
                for jx, s in enumerate(train_rows):
                    if jx != ix:
                        row_distances.append(distance.euclidean(r, s))
                    else:
                        row_distances.append(0)
                row_to_distances[key] = row_distances
                scaled_row_distances = []
                scaled_row = scaled_rows[ix]
                for jx, s in enumerate(scaled_rows):
                    if jx != ix:
                        scaled_row_distances.append(distance.euclidean(scaled_row, s))
                    else:
                        scaled_row_distances.append(0)
                scaled_row_to_distances[scaled_key] = scaled_row_distances
        self.training_index["row_to_distances"] = row_to_distances
        self.training_index["scaled_row_to_distances"] = scaled_row_to_distances
        
        # write the index file
        #with open(self.index_file, 'w') as outfile:
        #    json.dump(self.training_index, outfile)
        with open(self.index_pickle_file, 'wb') as outfile:
            pickle.dump(self.training_index, outfile)
        end_time = datetime.now()
        print("Built index in:", end_time - start_time)

    def get_distances(self, row):
        if str(row) in self.training_index["row_to_distances"]:
            return self.training_index["row_to_distances"][str(row)]
        else:
            distances = get_distances(row, self.training_index["train_rows"])
            self.training_index["row_to_distances"][str(row)] = distances
            return distances
        
    def get_scaled_distances(self, row):
        if str(row) in self.training_index["scaled_row_to_distances"]:
            return self.training_index["scaled_row_to_distances"][str(row)]
        else:
            self.distance_misses += 1
            start_time = datetime.now()
            distances = get_distances(row, self.training_index["scaled_rows"])
            end_time = datetime.now()
            self.training_index["scaled_row_to_distances"][str(row)] = distances
            self.toal_distance_compute_duration += end_time - start_time
            return distances
        
    def get_row(self, row_num):
        return self.training_index["train_rows"][row_num]
    
    def get_scaled_row(self, row_num):
        return self.training_index["scaled_rows"][row_num]
    
    def get_label(self, row_num):
        return self.training_index["train_labels"][row_num]
    
    def get_training_origin(self, row_num):
        return self.training_index["train_origin"][row_num]
    
    def get_misses(self):
        return self.distance_misses
    
    def get_distance_compute_time(self):
        return self.toal_distance_compute_duration
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='builds a training index',
                description='',
                epilog='')
    parser.add_argument('-i', '--index_folder', required=True)
    parser.add_argument('-t', '--training_folder', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.index_folder):
        os.makedirs(args.index_folder)

    training_index = TrainingIndex(args.index_folder, args.training_folder)

    print("Training index size:", sys.getsizeof(training_index))

    print(asizeof.asized(training_index, detail=1).format())

    print("done")




        


            


