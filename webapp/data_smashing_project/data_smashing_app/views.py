from django.shortcuts import render, render_to_response
import numpy as np
from data_smashing import Datasmashing
from data_processing import compute_similarity_matrix_aditional

# Create your views here.


def visualizer(request):

    dict = {'signal': []}

    submited_data = []
    filenames = []

    dict['data_posted'] = False
    
    # if post files were uploaded
    if request.method == "POST":
        # for every uploaded file
        for afile in request.FILES.getlist('myfiles'):
            # read data as string and change them to list of ints
            input_set = afile.read().split()
            input_set = [float(x) for x in input_set]
            submited_data.append(input_set)
            filenames.append(afile.name)

        # read threshold and quantization levels
        threshold = float(request.POST.get("threshold", ""))
        class_threshold = float(request.POST.get("class_threshold", ""))
        quantization = request.POST.get("quantization", "").split()
        quantization = [float(x) for x in quantization]
        # TODO: data prcessing is called
        similarity_matrix, inverted_strams, independent_stream_copies, stream_sumations = compute_similarity_matrix_aditional(submited_data, threshold, quantization)

        dict['files'] = filenames
        dict['signals'] = zip(filenames, submited_data)
        dict['similarity'] = zip(filenames, similarity_matrix.tolist())
        dict['inverted_strams'] = zip(filenames, inverted_strams)
        dict['independent_stream_copies'] = zip(filenames, independent_stream_copies)
        dict['stream_sumations'] = zip( [x for x in filenames for i in range(len(filenames))], filenames * (len(filenames)), stream_sumations)
        dict['threshold'] = threshold
        dict['class_threshold'] = class_threshold

        dict['data_posted'] = True

    return render(request, 'data_smashing_app/visualizer.html', dict)
