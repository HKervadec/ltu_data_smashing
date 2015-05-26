import numpy as np
# import scipy.io as scio
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

def compute_similarity_matrix_aditional(data, threshold, quantization_levels):
    '''
    function is similar to compute similarity_matrix but it add some more data to show on the web page
    param: data - data as python array each row one input vector
    param: threshold - threshold for annihilation_circut
    param: alphabet_size - size of used alphabet
    '''

    # quantizate input set
    quantization_levels.append(float('inf'))
    quantizated_data = quantizate_input_set(data, quantization_levels)

    ds = Datasmashing(len(quantization_levels))

    similarity_matrix = np.empty([len(quantizated_data), len(quantizated_data)])

    inverted_strams = []
    independent_stream_copies = []
    stream_sumations = [[None for _ in range(len(quantizated_data))] for _ in range(len(quantizated_data))]

    for i in range(len(quantizated_data)):
        # crate independet stream copies
        inverted_strams.append(ds.stream_inversion(quantizated_data[i]))
        independent_stream_copies.append(ds.independent_stream_copy(quantizated_data[i]))

        for j in range(i+1):
            
            # smash signals
            result_matrix, corectness = ds.annihilation_circut(quantizated_data[i], quantizated_data[j], threshold)

            similarity_matrix[i][j] = result_matrix[0][1]
            similarity_matrix[j][i] = result_matrix[1][0]

            # set stream sumations
            stream_sumations[i][j] = (ds.stream_sumation(quantizated_data[i], ds.stream_inversion(quantizated_data[j])))
            stream_sumations[j][i] = (ds.stream_sumation(quantizated_data[j], ds.stream_inversion(quantizated_data[i])))

    return similarity_matrix, inverted_strams, independent_stream_copies, [item for sublist in stream_sumations for item in sublist]

def quantizate_vector(vector, quantization_levels):
    '''
    function quatizate vector of data depending on quantization levels
    and return quatizated list
    param: vector - values to quantizate as python list
    param: quantization_levels - list of levels used for quantization
    '''
    results = []
    for v in vector:
        results.append(quantizate(v, quantization_levels))
    return results

def quantizate_input_set(input_set, quantization_levels):
    '''
    function make quantization task for all vectors of input set and return it as list of vectors
    param: input_set - list of vectors for quantizate
    param: quantization_levels - list of levels used for quantization
    '''
    
    results = []
    for v in input_set:
        results.append(quantizate_vector(v, quantization_levels))
    return results


# hmv, lmv1, lmv2 = read_data_for_vehicles('data.mat', [0.3, 1])


# from algorithm_test import print_stream_in_file

# for i in range(hmv.shape[1]):
#     print_stream_in_file(hmv[:,i], 'hmv-' + str(i) + '.txt')

# for i in range(lmv1.shape[1]):
#     print_stream_in_file(lmv1[:,i], 'lmv1-' + str(i) + '.txt')

# for i in range(lmv2.shape[1]):
#     print_stream_in_file(lmv2[:,i], 'lmv2-' + str(i) + '.txt')

# print lmv1[:,1:3]


# from algorithm_test import print_stream_in_file

# print_stream_in_file(lmv1[:,0], '1test1_lmv1.txt')
# print_stream_in_file(lmv1[:,1], '1test2_lmv1.txt')

# print lmv1.shape
# from algorithm_test import print_stream_in_file



# ds = Datasmashing(2)
# result_matrix, corectness = ds.annihilation_circut(tx1, tx2, 0.25)
# print result_matrix


# similarity_matrix = compute_similarity_matrix(np.concatenate((hmv[:,:3], lmv1[:,:3], lmv2[:,:3]), axis=1), 0.1, 2)
# print similarity_matrix