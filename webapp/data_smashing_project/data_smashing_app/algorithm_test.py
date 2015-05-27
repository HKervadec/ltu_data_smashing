import numpy as np
from data_smashing import Datasmashing

#from mpl_toolkits.mplot3d import Axes3D
from automata import Automata
# matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt

def set_axis(tresholds, alphabet_sizes):
    X, Y = np.meshgrid(tresholds, alphabet_sizes)
    return X, Y

def calculate_line(tresholds, alphabet_size):
    ds = Datasmashing(alphabet_size)
    
    s1 = create_stream(alphabet_size, 10000)
    s2 = create_stream(alphabet_size, 10000)

    results = np.empty(tresholds.shape, dtype=float)
    colors = np.empty(tresholds.shape, dtype=str)
    for i in range(len(tresholds)):
        results[i], color = ds.annihilation_circut(s1, s2, tresholds[i])
        colors[i] = 'g' if color else 'r'
    
    return results, colors

def calculate_data(tresholds, alphabet_sizes):
    X, Y = set_axis(tresholds, alphabet_sizes)

    results = np.empty(X.shape, dtype=float)
    colors = np.empty(X.shape, dtype=str)
    for i in range(len(alphabet_sizes)):
        results[i, :], colors[i, :] = calculate_line(tresholds, alphabet_sizes[i]) 

    return results, colors, X, Y


def create_stream(alphabet_size, stream_length, probabilities):
    '''
    function create streams over alphabet 1 to alphabet_sizes
    of length stream_length using probabilities
    param: alphabet_size
    param: stream_length
    param: probabilities - matrix of size alphabet_size x alphabet_size, which describes transition function for automata
    '''
    a = Automata(probabilities)
    return a.gen_stream(stream_length)

def print_stream_in_file(s, file_name):
    '''
    function print stream in space seperated file
    param: s - stream
    param: file_name - name of file
    '''
    with open(file_name, 'w') as fp:
        for v in s:
            fp.write(str(v) + "\n") # python will convert \n to os.linesep
        fp.close()

def run_datasmashing(tresholds, alphabet_size):
    ds = Datasmashing(alphabet_size)

    # p1 = [[0.5, 0.3, 0.2], [0.4, 0.5, 0.1], [0.6, 0.3, 0.1]] #[[0.8, 0.2], [0.8, 0.2]]
    
    # p2 = [[0.4, 0.5, 0.1], [0.1, 0.3, 0.6], [0.2, 0.3, 0.5]]  
    if alphabet_size == 2:
        p1 =[[0.8, 0.2], [0.8, 0.2]]
        p2 =[[0.85, 0.15], [0.85, 0.15]]
        p3 =[[0.25, 0.75], [0.75, 0.25]]
        p4 =[[0.5, 0.5], [0.5, 0.5]]
    else:
        p1 =[[0.8, 0.1,0.1], [0.8, 0.1,0.1], [0.8, 0.1,0.1]]
        p2 =[[0.9, 0.05,0.05], [0.9, 0.05,0.05], [0.9, 0.05,0.05]]
        p3 =[[0.20, 0.4,0.4], [0.8, 0.1,0.1], [0.8, 0.1,0.1]]
        p4 =[[0.35, 0.35,0.3], [0.35, 0.35,0.3], [0.35, 0.35,0.3]]
    # p2 =[[0.5, 0.5], [0.5, 0.5]]
    # p2 = [[0.8, 0.1, 0.02, 0.02, 0.06],
    #     [0.8, 0.1, 0.02, 0.02, 0.06],
    #     [0.3, 0.25, 0.2, 0.15, 0.1],
    #     [0.5, 0.3, 0.1, 0.05, 0.05],
    #     [0.45, 0.3, 0.15, 0.05, 0.05]]
    # p2 = [[0.05, 0.3, 0.25, 0.05, 0.35],
    #     [0.1, 0.25, 0.2, 0.15, 0.3],        
    #     [0.45, 0.3, 0.15, 0.05, 0.05],
    #     [0.5, 0.3, 0.1, 0.05, 0.05],
    #     [0.4, 0.3, 0.2, 0.07, 0.03]]

    # p1 = [[0.2, 0.2, 0.2, 0.2, 0.2],
    #     [0.2, 0.2, 0.2, 0.2, 0.2],        
    #     [0.2, 0.2, 0.2, 0.2, 0.2],
    #     [0.2, 0.2, 0.2, 0.2, 0.2],
    #     [0.2, 0.2, 0.2, 0.2, 0.2]]


    saa1 = create_stream(alphabet_size, 500, p1)
    saa2 = create_stream(alphabet_size, 500, p1)
    saa3 = create_stream(alphabet_size, 500, p1)
    saa4 = create_stream(alphabet_size, 50, p1)
    sab1 = create_stream(alphabet_size, 500, p2)
    sab2 = create_stream(alphabet_size, 500, p2)
    sab3 = create_stream(alphabet_size, 500, p2)
    sab4 = create_stream(alphabet_size, 5000, p2)
    sb1 = create_stream(alphabet_size, 500, p3)
    sb2 = create_stream(alphabet_size, 500, p3)
    sb3 = create_stream(alphabet_size, 500, p3)
    sb4 = create_stream(alphabet_size, 500, p3)
    sc1 = create_stream(alphabet_size, 500, p4)
    sc2 = create_stream(alphabet_size, 500, p4)
    sc3 = create_stream(alphabet_size, 500, p4)
    sc4 = create_stream(alphabet_size, 500, p4)
    #import collections
    #print collections.Counter(s1)

    print_stream_in_file(saa1, str(alphabet_size) + '_saa1.txt')
    print_stream_in_file(saa2, str(alphabet_size) + '_saa2.txt')
    print_stream_in_file(saa3, str(alphabet_size) + '_saa3.txt')
    print_stream_in_file(saa4, str(alphabet_size) + '_saa4.txt')
    print_stream_in_file(sab1, str(alphabet_size) + '_sab1.txt')
    print_stream_in_file(sab2, str(alphabet_size) + '_sab2.txt')
    print_stream_in_file(sab3, str(alphabet_size) + '_sab3.txt')
    print_stream_in_file(sab4, str(alphabet_size) + '_sab4.txt')
    print_stream_in_file(sb1, str(alphabet_size) + '_sb1.txt')
    print_stream_in_file(sb2, str(alphabet_size) + '_sb2.txt')
    print_stream_in_file(sb3, str(alphabet_size) + '_sb3.txt')
    print_stream_in_file(sb4, str(alphabet_size) + '_sb4.txt')
    print_stream_in_file(sc1, str(alphabet_size) + '_sc1.txt')
    print_stream_in_file(sc2, str(alphabet_size) + '_sc2.txt')
    print_stream_in_file(sc3, str(alphabet_size) + '_sc3.txt')
    print_stream_in_file(sc4, str(alphabet_size) + '_sc4.txt')


    #results = np.empty(tresholds.shape, dtype=float)
    #length_sufficiency = np.empty(tresholds.shape, dtype=str)
    #for i in range(len(tresholds)):
     #   results, length_sufficiency = ds.annihilation_circut(s1, s2, tresholds[i])
      #  print "treshold: ", tresholds[i]
       # print "results:"
        #print results
        #print "sufficiency: ", length_sufficiency

# p = [[[1]],
#     [[0.8, 0.2], [0.6, 0.4]],
#     [[0.5, 0.3, 0.2], [0.4, 0.5, 0.1], [0.6, 0.3, 0.1]],
#     [[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.15, 0.15], [0.45, 0.25, 0.20, 0.1], [0.35, 0.25, 0.2, 0.2]],
#     [[0.4, 0.3, 0.2, 0.07, 0.03],
#         [0.35, 0.3, 0.25, 0.05, 0.05],
#         [0.3, 0.25, 0.2, 0.15, 0.1],
#         [0.5, 0.3, 0.1, 0.05, 0.05],
#         [0.45, 0.3, 0.15, 0.05, 0.05]]]



tresholds = np.arange(0.4,0,-0.02) #np.array([0.5, 0.2, 0.05, 0.01, 0.005, 0.001])
alphabet_size = 2
run_datasmashing(tresholds, alphabet_size)
run_datasmashing(tresholds, 3)
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, results, rstride=1, cstride=1, facecolors=colors,
#         linewidth=0, antialiased=False)
# plt.show()
# print results[0, :]