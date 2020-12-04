<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
	<div style="margin:0 auto">
		上传的图片：
		<br><br><img src="upload/1.jpg" onload='if (this.width>256 || this.height>144)this.width=256,this.height=144' alt="上传图片"/>
	</div>	
	<div style="margin:0 auto">
		鸟的种类为：${birdkind}
	</div>
</body>
</html>