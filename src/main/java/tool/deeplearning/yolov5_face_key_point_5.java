package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

/**
*   @desc : 使用 yolov5 人脸关键点检测 5 点
 *
 *          下面包含了多个模型.主要是参数量不一样,模型输入输出一样
 *          yolov5face-blazeface-640x640.onnx   3.4Mb
 *          yolov5face-l-640x640.onnx   181Mb
 *          yolov5face-m-640x640.onnx   	83Mb
 *          yolov5face-n-0.5-320x320.onnx   2.5Mb
 *          yolov5face-n-0.5-640x640.onnx   4.6Mb
 *          yolov5face-n-640x640.onnx   9.5Mb
 *          yolov5face-s-640x640.onnx   30Mb
 *
*   @auth : tyf
*   @date : 2022-05-23  15:57:59
*/
public class yolov5_face_key_point_5 {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }


    public static class ImageObj{
        // 原始图片(原始尺寸)
        Mat src;
        // 原始图片(模型尺寸的)
        Mat dst;
        // 输入张量
        OnnxTensor tensor;
        // 阈值
        float conf_thres = 0.2f;
        float iou_thres = 0.5f;
        // 边框置信度过滤后得到
        ArrayList<float[]> datas;
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src.clone(),640,640);
            this.tensor = this.transferTensor(this.dst.clone(),3,640,640); // 转张量
            this.run(); // 执行推理
            this.nms(); // 执行nms过滤
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        // YOLOv5的输入是RGB格式的3通道图像，图像的每个像素需要除以255来做归一化，并且数据要按照CHW的顺序进行排布
        public float[] whc2cwh(float[] src) {
            float[] chw = new float[src.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < src.length; i += 3) {
                    chw[j] = src[i];
                    j++;
                }
            }
            return chw;
        }
        // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
        public Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(netWidth, netHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }
        public OnnxTensor transferTensor(Mat dst,int channels,int netWidth,int netHeight){
            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

            //  归一化 0-255 转 0-1
            dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);

            // 创建 onnxruntime 需要的 tensor
            // 传入输入的图片 float 数组并指定数组shape
            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,channels,netWidth,netHeight});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }

        // 中心点坐标转 xin xmax ymin ymax
        public static float[] xywh2xyxy(float[] bbox,float maxWidth,float maxHeight) {
            // 中心点坐标
            float x = bbox[0];
            float y = bbox[1];
            float w = bbox[2];
            float h = bbox[3];
            // 计算
            float x1 = x - w * 0.5f;
            float y1 = y - h * 0.5f;
            float x2 = x + w * 0.5f;
            float y2 = y + h * 0.5f;
            // 限制在图片区域内
            return new float[]{
                    x1 < 0 ? 0 : x1,
                    y1 < 0 ? 0 : y1,
                    x2 > maxWidth ? maxWidth:x2,
                    y2 > maxHeight? maxHeight:y2};
        }

        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));

                // ---------模型1输出-----------
                // output -> [1, 25200, 16] -> FLOAT
                float[][] out = ((float[][][])(res.get(0)).getValue())[0];

                ArrayList<float[]> datas = new ArrayList<>();

                // 25200个预选框
                for(int i=0;i< out.length;i++){
                    // x y w h score  中心点坐标和分数
                    // x y 关键点坐标
                    // x y 关键点坐标
                    // x y 关键点坐标
                    // x y 关键点坐标
                    // x y 关键点坐标
                    // cls_conf 人脸置信度
                    float[] data = out[i];

                    float score1 = data[4]; // 边框置信度
                    float score2 = data[15];// 人脸置信度

                    if( score1 >= 0.2 && score2>= 0.2){
                        // xywh 转 x1y1x2y2
                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];
                        float[] xyxy = xywh2xyxy(new float[]{x,y,w,h},640,640);
                        data[0] = xyxy[0];
                        data[1] = xyxy[1];
                        data[2] = xyxy[2];
                        data[3] = xyxy[3];
                        datas.add(data);
                    }

                }

                // 保存边框
                this.datas = datas;
            }
            catch (Exception e){
                e.printStackTrace();
            }
        }
        // 图片缩放
        public BufferedImage resize(BufferedImage img, int newWidth, int newHeight) {
            Image scaledImage = img.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
            BufferedImage scaledBufferedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = scaledBufferedImage.createGraphics();
            g2d.drawImage(scaledImage, 0, 0, null);
            g2d.dispose();
            return scaledBufferedImage;
        }
        // Mat 转 BufferedImage
        public BufferedImage mat2BufferedImage(Mat mat){
            BufferedImage bufferedImage = null;
            try {
                // 将Mat对象转换为字节数组
                MatOfByte matOfByte = new MatOfByte();
                Imgcodecs.imencode(".jpg", mat, matOfByte);
                // 创建Java的ByteArrayInputStream对象
                byte[] byteArray = matOfByte.toArray();
                ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArray);
                // 使用ImageIO读取ByteArrayInputStream并将其转换为BufferedImage对象
                bufferedImage = ImageIO.read(byteArrayInputStream);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return bufferedImage;
        }

        // 执行nms过滤
        public void nms(){

            ArrayList<float[]> temp = new ArrayList<>();

            // 将 datas 按照置信度从大到小顺序排放
            datas.sort((o1, o2) -> Float.compare(
                    o2[4]*o2[5],
                    o1[4]*o1[5]
            ));

            while (!datas.isEmpty()){

                float[] max = datas.get(0);
                temp.add(max);
                Iterator<float[]> it = datas.iterator();
                while (it.hasNext()) {
                    // 交并比
                    float[] obj = it.next();
                    double iou = calculateIoU(
                            new float[]{max[0],max[1],max[2],max[3]},
                            new float[]{obj[0],obj[1],obj[2],obj[3]}
                    );
                    if (iou > iou_thres) {
                        it.remove();
                    }
                }

            }

            this.datas = temp;

        }

        // 计算两个框的交并比
        private double calculateIoU(float[] box1, float[] box2) {

            //  getXYXY() 返回 xmin-0 ymin-1 xmax-2 ymax-3

            double x1 = Math.max(box1[0], box2[0]);
            double y1 = Math.max(box1[1], box2[1]);
            double x2 = Math.min(box1[2], box2[2]);
            double y2 = Math.min(box1[3], box2[3]);
            double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
            double box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
            double box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
            double unionArea = box1Area + box2Area - intersectionArea;
            return intersectionArea / unionArea;
        }

        // 弹窗显示
        public void show(){

            // 遍历所有边框
            datas.stream().forEach(n->{

                // x y w h score  中心点坐标和分数
                // x y 关键点坐标
                // x y 关键点坐标
                // x y 关键点坐标
                // x y 关键点坐标
                // x y 关键点坐标
                // cls_conf 人脸置信度

                float x1 = n[0];
                float y1 = n[1];
                float x2 = n[2];
                float y2 = n[3];

                float point1_x = n[5];// 关键点1
                float point1_y = n[6];// 关键点1

                float point2_x = n[7];// 关键点2
                float point2_y = n[8];// 关键点2

                float point3_x = n[9];// 关键点3
                float point3_y = n[10];// 关键点3

                float point4_x = n[11];// 关键点4
                float point4_y = n[12];// 关键点4

                float point5_x = n[13];// 关键点5
                float point5_y = n[14];// 关键点5

                // 画边框
                Imgproc.rectangle(
                        dst,
                        new Point(Float.valueOf(x1).intValue(), Float.valueOf(y1).intValue()),
                        new Point(Float.valueOf(x2).intValue(), Float.valueOf(y2).intValue()),
                        color1,
                        2);
                // 左眼
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(point1_x).intValue(), Float.valueOf(point1_y).intValue()),
                        1, // 半径
                        color2,
                        2);
                // 右眼
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(point2_x).intValue(), Float.valueOf(point2_y).intValue()),
                        1, // 半径
                        color2,
                        2);
                // 鼻子
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(point3_x).intValue(), Float.valueOf(point3_y).intValue()),
                        1, // 半径
                        color2,
                        2);
                // 左嘴角
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(point4_x).intValue(), Float.valueOf(point4_y).intValue()),
                        1, // 半径
                        color2,
                        2);
                // 右嘴角
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(point5_x).intValue(), Float.valueOf(point5_y).intValue()),
                        1, // 半径
                        color2,
                        2);
            });

            JFrame frame = new JFrame("Image");
            frame.setSize(dst.width(), dst.height());

            // 图片转为原始大小
            BufferedImage img = mat2BufferedImage(dst);
            img = resize(img,src.width(),src.height());

            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // input -> [1, 3, 640, 640] -> FLOAT
        // ---------模型1输出-----------
        // output -> [1, 25200, 16] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov5_face_key_point_5\\yolov5face-s-640x640.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov7_face_key_point_5\\pic.png");

        // 显示
        image.show();

    }



}
