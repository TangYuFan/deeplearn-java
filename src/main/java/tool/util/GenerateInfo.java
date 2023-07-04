package tool.util;


import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
*   @desc : 生成 deeplearning 包下面所有程序的信息
*   @auth : tyf
*   @date : 2022-05-11  18:03:40
*/
public class GenerateInfo {


    public static String readInfo(String path){

        File clazz = new File(path);
        String info = "unknow";

        try {
            BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(clazz),"UTF-8"));
            String line = null;
            while ((line = br.readLine()) != null) {
                boolean stop = false;
                // 读取 @desc 或者 @Desc
                if(line.contains("@desc")){
                    info = line.substring(line.indexOf("desc")).replace("desc","").replace(":","").replace("：","");
                    stop = true;
                }
                else if(line.contains("@Desc")){
                    info = line.substring(line.indexOf("Desc")).replace("Desc","").replace(":","").replace("：","");
                    stop = true;
                }
                if(stop){
                    break;
                }
            }
            br.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return info;
    }

    public static void writeMd(String path,String content){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path,false))) {
            writer.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws Exception{


        // 当前项目目录
        String root = new File("").getCanonicalPath();

        String readMePath = root+"\\README.md";

        // 包目录
        String pack = root + "\\src\\main\\java\\tool\\deeplearning\\";

        // 遍历所有文件
        File packPath = new File(pack);

        System.out.println("包路径:"+pack);
        System.out.println("说明文件:"+readMePath);

        // 保存所有类的信息
        JSONObject infos = new JSONObject(true);

        if(packPath.isDirectory()){
            File[] javaFile = packPath.listFiles();
            Arrays.stream(javaFile).forEach(file->{
                // 类文件
                String clazzPath = file.getAbsolutePath();
                // 类名称
                String clazz = clazzPath.replace(pack,"").replace(".java","");
                // 读取的信息
                String info = readInfo(clazzPath);
                // 打印
                infos.put(clazz,info);
            });
        }

        // 生成readme需要写入的字符串
        StringBuffer readme = new StringBuffer();
        readme.append("<div class=\"container\">\n" +
                "<div class=\"row\">\n" +
                "<h3>deeplearn-java</h3>\t\n" +
                "</div>\n" +
                "<div class=\"row\">\n" +
                "<div class=\"span4\">\n" +
                "<table class=\"table\">\n" +
                "<thead>\n" +
                "<tr>\n" +
                "<th>\n" +
                "编号\n" +
                "</th>\n" +
                "<th>\n" +
                "项目\n" +
                "</th>\n" +
                "<th>\n" +
                "说明\n" +
                "</th>\n" +
                "</tr>\n" +
                "</thead>\n" +
                "<tbody>\n");


        // 插入所有项目
        AtomicInteger i= new AtomicInteger(1);
        infos.entrySet().forEach(n->{
            String k = n.getKey();
            String v = (String)n.getValue();
            readme.append("<tr><td>"+i+"</td><td>"+k+"</td><td>"+v+"</td></tr>").append("\n");
            i.getAndIncrement();
        });


        readme.append("</tbody>\n" +
                "</table>\n" +
                "</div>\n" +
                "<div class=\"span4\">\n" +
                "</div>\n" +
                "</div>\n" +
                "</div>");

        // 打开md文件覆盖模式写入
        writeMd(readMePath,readme.toString());

    }




}
