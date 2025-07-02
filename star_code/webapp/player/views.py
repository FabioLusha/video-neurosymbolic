from django.shortcuts import render, redirect
from .models import Video
from .forms import VideoForm

# Create your views here.

def video_upload(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_list')
    else:
        form = VideoForm()
    return render(request, 'player/video_upload.html', {'form': form})

def video_list(request):
    videos = Video.objects.all()
    return render(request, 'player/video_list.html', {'videos': videos})
