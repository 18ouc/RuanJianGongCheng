# 软件工程 鸟类识别API

## 技术栈
- 前端： Layui 经典模块化前端框架、jQuery
- 后端：Django

## 部署

### 环境
- Python3.6+
- Django



如果你没有 Django，可以使用pip安装:
```
pip install Django
```

### 部署方法

```
git clone https://github.com/18ouc/RuanJianGongCheng.git
cd RuanJianGongCheng
python manage.py runserver 8000
```

接着打开浏览器[http://127.0.0.1:8000/](http://127.0.0.1:8000/)即可看到

## 文件说明

```
birds_api
├── __init__.py
├── api.py 			# 图片接口实现
├── asgi.py  		
├── settings.py     # Django 配置
├── urls.py    		
├── views.py        #视图渲染
└── wsgi.py
static/              # CSS js 静态文件
templates/   		# 模板
```

## 已完成工作
- 主页界面编写
- 图片上传功能
- 上传后端接口

## Todo
- 模型训练
- 模型部署
