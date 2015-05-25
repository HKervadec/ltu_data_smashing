import numpy as np
import scipy.io as scio
from data_smashing import Datasmashing
import csv

def load_data(file_name):
    ''' 
    function load data from mat file with name file_name
    param: file_name - name of file to load
    '''
    return scio.loadmat(file_name)

def load_data_for_group(data, group_name, column):
    ''' 
    function takes data for one group of vehicles from data dictionary
    param: group_name - name of obtained grop
    param: data - dictionary with data
    param: column - desired column in data (3 - unnormailized, 4 - normailzed)
    '''
    return  data[group_name][:, column]

def quantizate(data, quantization_levels):
    '''
    fucntion quantizatizate one single level
    param: data - number to be quantizate
    param: quantization_levels - array of border values in quantization
    '''
    for i in range(len(quantization_levels)):
        if quantization_levels[i] >= data:
            return i + 1
    return len(quantization_levels)

def quantization(data, quantization_levels):
    '''
    function quantizate data depending on quantization_levels 
    param: data - matrix to quantizate
    param: quantization_levels - array of border values in quantization
    return: result - matrix where every column is one vehilce and rows are data
    '''

    result = np.empty([len(data[0]), len(data)])
    for i in range(len(data)):
        for j in range(len(data[0])):
            result[j, i] = quantizate(data[i][j][0], quantization_levels)
    return result

def read_data_for_vehicles(file_name, quantization_levels):
    '''
    function reads and quantizates measurements for all tree types of vehicle
    param: file_name - name of the file wiht data
    param: quantization_levels - levels for quantization
    '''
    data = load_data(file_name)
    hmv = quantization(load_data_for_group(data, 'HMV', 4), quantization_levels)
    lmv1 = quantization(load_data_for_group(data, 'LMV1', 4), quantization_levels)
    lmv2 = quantization(load_data_for_group(data, 'LMV2', 4), quantization_levels)

    return hmv, lmv1, lmv2

def compute_similarity_matrix(data, threshold, alphabet_size):
    '''
    function compute similarity matrix for provided data
    param: data - data matrix with signlas in columns
    param: threshold - threshold for annihilation_circut
    param: alphabet_size - size of used alphabet
    '''
    ds = Datasmashing(alphabet_size)
    similarity_matrix = np.empty([data.shape[1], data.shape[1]])

    print data.shape
    repeating = 1
    for i in range(data.shape[1]):
        column_i = data[:, i]
        column_i_concat = np.array([])
        for k in range(repeating):
            column_i_concat = np.concatenate((column_i_concat, column_i), axis=0)
        print_stream_in_file(column_i_concat, 'test_' + str(i))
        for j in range(i+1):
            
            column_j = data[:, j]            
            column_j_concat = np.array([])
            for k in range(repeating):                
                column_j_concat = np.concatenate((column_j_concat, column_j), axis=0)

            column_i_concat = [int(l) for l in column_i_concat]
            column_j_concat = [int(l) for l in column_j_concat]
            
            result_matrix, corectness = ds.annihilation_circut(column_i_concat, column_j_concat, threshold)

            # print result_matrix

            similarity_matrix[i][j] = result_matrix[0][1]
            similarity_matrix[j][i] = result_matrix[1][0]
            # print corectness


    return similarity_matrix

def read_text_file(name):
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
        data = []
        for row in reader:
            data.append(int(row[0]) + 1)
    return data


# hmv, lmv1, lmv2 = read_data_for_vehicles('data.mat', [0.3, 1])

# print lmv1[:,1:3]


# from algorithm_test import print_stream_in_file

# print_stream_in_file(lmv1[:,0], '1test1_lmv1.txt')
# print_stream_in_file(lmv1[:,1], '1test2_lmv1.txt')

# print lmv1.shape
from algorithm_test import print_stream_in_file

tx1 = read_text_file('0017.txt')
tx2 = read_text_file('0018.txt')

print_stream_in_file(tx1, '0017' + str('-1'))
print_stream_in_file(tx2, '0018' + str('-1'))


ds = Datasmashing(2)
result_matrix, corectness = ds.annihilation_circut(tx1, tx2, 0.25)
print result_matrix


# similarity_matrix = compute_similarity_matrix(np.concatenate((hmv[:,:3], lmv1[:,:3], lmv2[:,:3]), axis=1), 0.1, 2)
# print similarity_matrix