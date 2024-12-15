from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponse
import json
from Faceswap.faceswap import process
from PIL import Image
import io
from django.shortcuts import render


def home_page(request):
    return render(request, 'index.html')

@csrf_exempt
def face_swap(request):
    if request.method == 'POST':
        source_images = request.FILES.getlist('source_images[]')
        
        target_image = request.FILES.get("target_image")
        source_indexes = request.POST.get("source_indexes")
        target_indexes = request.POST.get("target_indexes")

        if source_indexes is None or source_indexes == '':
            source_indexes = "-1"
        if target_indexes is None or target_indexes == '':
            target_indexes = "-1"

        all_source_images = []

        if not source_images:
            return JsonResponse({"error": "No source images were found"}, status=400)
        
        if not target_image:
            return JsonResponse({"error": "No target images were found"}, status=400)
        
        try:
            target_image = Image.open(target_image)
            for source_image in source_images:
                image = Image.open(source_image)
                all_source_images.append(image)
        except Exception as e:
            return JsonResponse({"error":str(e)}, status=400)
    

        model = "./checkpoints/inswapper_128.onnx"
        result_image = process(all_source_images, target_image,source_indexes, target_indexes, model)
        if result_image.status_code==200:
            buffer = io.BytesIO()
            
            result_image['image'].save(buffer,format='PNG')

            image_bytes = buffer.getvalue()
    
            response = HttpResponse(content_type='image/png')
            response.write(image_bytes)
            
            return response
        else:
            return result_image
        
    return JsonResponse({"message": "Only POST requests are allowed"}, status=405)
    