from django.shortcuts import render

def index(request):
    context = {
        'title':'Knooooooow birds',
        'app_name':'什么鸟？',
        'decription':'这是一个机器学习的鸟类识别应用，上传一张照片，即可识别出鸟类名称',
        'upload_titie':'网站处于测试阶段，部分功能未完善。'
    }

    return render(request, 'index.html', context)


# def post(request):
#     json = {
#         'code':200,
#         'kind':20
#     }
#     return json