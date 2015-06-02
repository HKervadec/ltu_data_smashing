import numpy as np
import scipy.io as scio
from data_smashing import Datasmashing
import csv
from operator import itemgetter

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



def get_training_sets(hmv, lmv1, lmv2):
    '''
    function export training set depends on our observations
    param: hmv - data for hmv cars
    param: lmv1 - data for lmv1 cars
    param: lmv2 - data for lmv2 cars
    '''
    result_hmv = np.concatenate((hmv[:, :7], hmv[:, 8:13], hmv[:, 14:16], hmv[:, 17:24], hmv[:, 25:27], hmv[:, 28:39],
                                hmv[:, 41:44], hmv[:, 45: 49], hmv[:, 50:66], hmv[:, 67:70]), axis=1)
    result_lmv1 = np.concatenate((lmv1[:, :2], lmv1[:, 3:7], lmv1[:, 8: 11],
                                 lmv1[:, 12:22], lmv1[:, 23:24], lmv1[:, 25:34], lmv1[:, 35:36]), axis=1)
    result_lmv2 = np.concatenate((lmv2[:, :7], lmv2[:, 8:12], lmv2[:, 13:26], lmv2[:, 27:30], lmv2[:, 32:43], 
                                lmv2[:, 44:54], lmv2[:, 55:61], lmv2[:, 63:64], lmv2[:, 65:73], lmv2[:, 75:76],
                                lmv2[:, 77:80], lmv2[:, 81:86], lmv2[:, 90:91], lmv2[:, 93:99], lmv2[:, 100:101]), axis=1)

    return result_hmv, result_lmv1, result_lmv2

def compute_similarities(stream, labeled_streams, alphabet_size, threshold):
    '''
    function compute similarity between stream and traingin streams
    param: stream - strema to classify
    param: labeled_streams - training stream with class lables in the first row
    param: alphabet_size - size of the alphabet
    param: threshold - data smashing threshold
    '''
    results = []
    ds = Datasmashing(alphabet_size)
    for i in range(labeled_streams.shape[1]):
        sim_mat = ds.annihilation_circut(stream, labeled_streams[1:, i], threshold)
        if sim_mat[0][1,1] < threshold:
            results.append([labeled_streams[0, i], min(sim_mat[0][0,1], sim_mat[0][1,0])])
        else:
            print 'Selfcheck value for ' + str(i) + 'th training stream was too high: ' + str(sim_mat[1, 1])

    return sorted(results, key=itemgetter(1))

def classification_simple(stream, labeled_streams, alphabet_size, threshold, num_of_best):
    '''
    function does classification for strem according to training_streams
    method get num of num_of_best most similar training streams and return label with the hightest frequency
    param: stream - strema to classify
    param: labeled_streams - training stream with class lables in the first row
    param: alphabet_size - size of the alphabet
    param: threshold - data smashing threshold
    param: num_of_best - number of the most similar stream we are checking
    '''
    similarity = compute_similarities(stream, labeled_streams, alphabet_size, threshold)
    counter = [0, 0, 0]
    for row in similarity[0:num_of_best]:
        counter[int(row[0])] += 1 

    return counter.index(max(counter))

def classification_average(stream, labeled_streams, alphabet_size, threshold):
    '''
    function does classification for strem according to training_streams
    method take average similarity for every class and take the best
    param: stream - strema to classify
    param: labeled_streams - training stream with class lables in the first row
    param: alphabet_size - size of the alphabet
    param: threshold - data smashing threshold
    param: num_of_best - number of the most similar stream we are checking
    '''
    similarity = compute_similarities(stream, labeled_streams, alphabet_size, threshold)
    sums = [0, 0, 0]
    counter = [0, 0, 0]
    for row in similarity:
        counter[int(row[0])] += 1
        sums[int(row[0])] += row[1]
    print counter
    sums = [sums[i] / counter[i] for i in range(len(counter))]

    return sums.index(min(sums))

def classification_simple_list(streams, labeled_streams, alphabet_size, threshold, num_of_best):
    '''
    function make simple classification for whole streams array
    param: streams - list of streams
    param: labeled_streams - training stream with class lables in the first row
    param: alphabet_size - size of the alphabet
    param: threshold - data smashing threshold
    param: num_of_best - number of the most similar stream we are checking
    '''
    results = []
    counter = 0
    for i in range(streams.shape[1]):
        results.append(classification_simple(streams[:, i], labeled_streams, alphabet_size, threshold, num_of_best))
        print i
    return results

def classification_average_list(streams, labeled_streams, alphabet_size, threshold):
    '''
    function make simple classification for whole streams array
    param: streams - list of streams
    param: labeled_streams - training stream with class lables in the first row
    param: alphabet_size - size of the alphabet
    param: threshold - data smashing threshold
    param: num_of_best - number of the most similar stream we are checking
    '''
    results = []
    counter = 0
    for i in range(streams.shape[1]):
        results.append(classification_average(streams[:, i], labeled_streams, alphabet_size, threshold))
        print i
    return results


hmv, lmv1, lmv2 = read_data_for_vehicles('data.mat', [0.3, 1])

train_hmv, train_lmv1, train_lmv2 = get_training_sets(hmv, lmv1, lmv2)

train_hmv = np.concatenate((np.ones((1, train_hmv.shape[1])) * 0, train_hmv), axis=0)
train_lmv1 = np.concatenate((np.ones((1, train_lmv1.shape[1])) * 1, train_lmv1), axis=0)
train_lmv2 = np.concatenate((np.ones((1, train_lmv2.shape[1])) * 2, train_lmv2), axis=0)

labeled_streams = np.concatenate((train_hmv, train_lmv1, train_lmv2), axis=1)

print classification_average_list(hmv[:, 70:100], labeled_streams, 2, 0.3)




# from algorithm_test import print_stream_in_file

# for i in range(train_hmv.shape[1]):
#     print_stream_in_file(train_hmv[:,i], 'hmv-train-' + str(i) + '.txt')

# for i in range(train_lmv1.shape[1]):
#     print_stream_in_file(train_lmv1[:,i], 'lmv1-train-' + str(i) + '.txt')

# for i in range(train_lmv2.shape[1]):
#     print_stream_in_file(train_lmv2[:,i], 'lmv2-train-' + str(i) + '.txt')

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