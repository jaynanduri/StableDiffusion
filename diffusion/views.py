import base64
import json
from io import BytesIO
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .LatentDiffusion.model import StableDiffusion

# Create your views here.
diffuser = StableDiffusion()


@csrf_exempt
def post_generated_image(request):
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        content = body['prompt']
        image = diffuser.generate(content)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return JsonResponse({"image": "data:image/png;base64,"+img_str})
    else:
        return JsonResponse({"error": "This api only handles post requests"})
