from django.shortcuts import render
import numpy as np
from data_smashing import Datasmashing

# Create your views here.


def visualizer(request):

    dict = {'signal': []}

    # create distributions
    ds = Datasmashing(5)

    probabilities = [0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.03, 0.02, 0.06, 0.06]
    probabilities = [0.4, 0.3, 0.2, 0.05, 0.05]
    # probabilities = [0.5, 0.3, 0.07, 0.03]

    s1 = ds.generate(probabilities, 10000)
    s2 = ds.generate(probabilities, 10000)

    dict['signal1'] = s1
    dict['signal2'] = s2

    dict['s1_string'] = ''.join(str(x) for x in s1)
    dict['s2_string'] = ''.join(str(x) for x in s2)

    s1_inverted = ds.stream_inversion(s1)
    s2_inverted = ds.stream_inversion(s2)

    dict['s1_inverted_string'] = ''.join(str(x) for x in s1_inverted)
    dict['s2_inverted_string'] = ''.join(str(x) for x in s2_inverted)

    s1_s1_sum = ds.stream_sumation(s1, s1_inverted)
    s1_s2_sum = ds.stream_sumation(s1_inverted, s2)
    s2_s2_sum = ds.stream_sumation(s2, s2_inverted)

    dict['s1_s1_sum'] = ''.join(str(x) for x in s1_s1_sum)
    dict['s1_s2_sum'] = ''.join(str(x) for x in s1_s2_sum)
    dict['s2_s2_sum'] = ''.join(str(x) for x in s2_s2_sum)

    treshold = 0.01

    l = int(np.log(1 / treshold) / np.log(ds.alphabet_size))
    print l

    epsilon11 = ds.deviation(s1_s1_sum, l)
    epsilon12 = ds.deviation(s1_s2_sum, l)
    epsilon22 = ds.deviation(s2_s2_sum, l)
    
    dict['epsilon11'] = epsilon11
    dict['epsilon12'] = epsilon12
    dict['epsilon22'] = epsilon22


    return render(request, 'data_smashing_app/visualizer.html', dict)
