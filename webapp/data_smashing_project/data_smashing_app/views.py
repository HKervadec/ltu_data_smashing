from django.shortcuts import render

# Create your views here.


def visualizer(request):

    return render(request, 'data_smashing_app/visualizer.html', {})
