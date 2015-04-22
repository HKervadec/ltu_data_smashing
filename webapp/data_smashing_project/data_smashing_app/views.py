from django.shortcuts import render
from random import randint

# Create your views here.


def visualizer(request):

    dict = {'signal': []}
    for i in range(200):
        dict['signal'].append(randint(1, 500))

    return render(request, 'data_smashing_app/visualizer.html', dict)
