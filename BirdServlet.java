package com.dzz.servlet;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.Part;

import com.dzz.service.BirdService;

import jdk.nashorn.internal.ir.RuntimeNode.Request;

public class BirdServlet extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp)throws ServletException, IOException {
        BirdService birdService=new BirdService();
    	Part part = req.getPart("bird");
        String disposition = part.getHeader("Content-Disposition");
        String suffix = disposition.substring(disposition.lastIndexOf("."),disposition.length()-1);
          //随机的生存一个32的字符串
        String filename = UUID.randomUUID()+suffix;
          //获取上传的文件名
        InputStream is = part.getInputStream();
        //动态获取服务器的路径
        String serverpath = req.getServletContext().getRealPath("upload");
        FileOutputStream fos = new FileOutputStream(serverpath+"/"+filename);
        byte[] bty = new byte[1024];
        int length =0;
        while((length=is.read(bty))!=-1){
            fos.write(bty,0,length);
        }
        fos.close();
        is.close();
        String birdkind=birdService.KnowBird("upload/"+filename);
        req.setAttribute("filename", filename);
        req.setAttribute("birdkind", birdkind);
        req.getRequestDispatcher("knowingbird.jsp").forward(req,resp);
    }
}