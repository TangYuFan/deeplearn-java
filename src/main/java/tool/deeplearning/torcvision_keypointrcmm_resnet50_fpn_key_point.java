package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;

/**
*   @desc : pytorch 下面的 keypoint 算法
*   @auth : tyf
*   @date : 2022-05-05  17:59:44
*/
public class torcvision_keypointrcmm_resnet50_fpn_key_point {

    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型输入输出
        printModel();

    }

    // 目标框
    public static class Detection{
        float[] xyxy;// 边框信息
        long label;// 类别信息
        float score;// 得分
        List<Point> points;// 点集合
        public Detection(float[] xyxy, long label, float score) {
            this.xyxy = xyxy;
            this.label = label;
            this.score = score;
            this.points = new ArrayList<>();
        }
        public void addPoint(Point p){
            this.points.add(p);
        }
    }

    // 目标点
    public static class Point{
        float x;
        float y;
        float visible;
        float score;
        public Point(float x, float y,float score,float visible) {
            this.x = x;
            this.y = y;
            this.score = score;
            this.visible = visible;
        }
    }

    // 打印模型输入输出
    public static void printModel() throws Exception{
        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型输入-----------");
        session.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型输出-----------");
        session.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });

    }


    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }

    // YOLOv5的输入是RGB格式的3通道图像，图像的每个像素需要除以255来做归一化，并且数据要按照CHW的顺序进行排布
    public static float[] whc2cwh(float[] src) {
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

    // 交并比
    private static double calculateIoU(float[] box1, float[] box2) {
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

    public static OnnxTensor transferTensor(OrtEnvironment env, Mat dst, int inputCount, int inputChannel, int inputWidth, int inputHeight){
        // BGR -> RGB
        // python 中图像通常以rgb加载,java通常以 bgr加载
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
//        dst.convertTo(dst, CvType.CV_32FC1);// 矩阵转单精度浮点型
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);// 矩阵转单精度浮点型,并且每个元素除255进行归一化
        // 初始化一个输入数组 channels * netWidth * netHeight
        float[] whc = new float[ inputChannel * inputWidth * inputHeight ];
        dst.get(0, 0, whc);
        // 得到最终的图片转 float 数组 whc 转 chw
        // prtorch 中图片以chw格式加载
        float[] chw = whc2cwh(whc);
        // 创建 onnxruntime 需要的 tensor
        OnnxTensor tensor = null;
        try {
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{inputCount,inputChannel,inputWidth,inputHeight});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }

    // Mat 转 BufferedImage
    public static BufferedImage mat2BufferedImage(Mat mat){
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


    public static void filter1(List<Detection> detections,float scoreThreshold){
        detections.removeIf(detection -> detection.score<=scoreThreshold);
    }

    public static void filter2(List<Detection> detections,float nmsThreshold){

        // 先按得分降序
        detections.sort((o1, o2) -> Float.compare(o2.score,o1.score));

        // 需要删除的
        List<Detection> res = new ArrayList<>();

        while (!detections.isEmpty()){
            Detection max = detections.get(0);
            res.add(max);
            Iterator<Detection> it = detections.iterator();
            while (it.hasNext()){
                Detection bi = it.next();
                // 计算交并比
                if(calculateIoU(max.xyxy,bi.xyxy)>=nmsThreshold){
                    it.remove();
                }
            }
        }

        // 保存剩下的
        detections.clear();
        detections.addAll(res);

    }

    public static void filter3(List<Detection> detections,float pointThreshold){

        detections.stream().forEach(n->{
            n.points.removeIf(point->point.score<=pointThreshold);
        });

    }


    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,不添加留白
    public static Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(inputWidth, inputHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }

    // x 坐标还原
    public static int transferXWithOutPadding(int x,int imgWidth,int netWidth){
        float scala  = Float.valueOf(imgWidth)/Float.valueOf(netWidth);
        float res = x * scala;
        return res >= imgWidth ? imgWidth : Float.valueOf(res).intValue();
    }

    // y 坐标还原
    public static int transferYWithOutPadding(int y,int imgHeight,int netHeight){
        float scala  = Float.valueOf(imgHeight)/Float.valueOf(netHeight);
        float res = y * scala;
        return res >= imgHeight ? imgHeight : Float.valueOf(res).intValue();
    }


    // y 坐标还原

    // 标注原图
    public static void showBoxAndPoint(Mat src, List<Detection> detections,int imgWidth,int imgHeight,int netWidth,int netHeight){

        Scalar color1 = new Scalar(0,0,255);// 边框颜色
        Scalar color2 = new Scalar(0,255,0);// 可见点颜色
        Scalar color3 = new Scalar(255,0,0);// 不可见点颜色

        // 画框
        detections.stream().forEach(n->{

            int x1 = Float.valueOf(n.xyxy[0]).intValue();
            int y1 = Float.valueOf(n.xyxy[1]).intValue();
            int x2 = Float.valueOf(n.xyxy[2]).intValue();
            int y2 = Float.valueOf(n.xyxy[3]).intValue();

            // 坐标转换
            x1 = transferXWithOutPadding(x1,imgWidth,netWidth);
            x2 = transferXWithOutPadding(x2,imgWidth,netWidth);
            y1 = transferYWithOutPadding(y1,imgHeight,netHeight);
            y2 = transferYWithOutPadding(y2,imgHeight,netHeight);

            // 颜色
            Imgproc.rectangle(
                    src,
                    new org.opencv.core.Point(x1,y1),
                    new org.opencv.core.Point(x2,y2),
                    color1,
                    2);

            // 遍历每个点
            n.points.stream().forEach(p->{

                int x = Float.valueOf(p.x).intValue();
                int y = Float.valueOf(p.y).intValue();
                x = transferXWithOutPadding(x,imgWidth,netWidth);
                y = transferYWithOutPadding(y,imgHeight,netHeight);

                float s = p.score;
                float visible = p.visible;
                // 画点,实际上是以xy为圆心画一个圆 可见和不可见使用两种颜色
                if(visible==0){
                    Imgproc.circle(
                            src,
                            new org.opencv.core.Point(x,y),
                            3, // 半径
                            color2,
                            -1);
                }
                else if(visible==1){
                    Imgproc.circle(
                            src,
                            new org.opencv.core.Point(x,y),
                            3, // 半径
                            color3,
                            -1);
                }

            });

        });

        // 弹窗显示
        BufferedImage imageDst = mat2BufferedImage(src);
        JFrame frame = new JFrame("Image");
        frame.setSize(imageDst.getWidth(), imageDst.getHeight());
        JLabel label = new JLabel(new ImageIcon(imageDst));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


    }


    public static void main(String[] args) throws Exception{


        // 模型初始化
        String weight = new File("").getCanonicalPath() + "\\model\\deeplearning\\keypointrcnn_resnet50_fpn\\keypointrcnn_resnet50_fpn.onnx";
        init(weight);

        // 模型输入宽高,onnx网站可以看到
        int inputCount = 1;
        int inputChannel = 3;
        int inputWidth = 1024;
        int inputHeight = 1024;


        // 图片读取成Mat
        String pic = new File("").getCanonicalPath() + "\\model\\deeplearning\\keypointrcnn_resnet50_fpn\\pic.png";
        Mat src = readImg(pic);

        // resize
        Mat dst = resizeWithoutPadding(src,inputWidth,inputHeight);

        // 转为输入
        OnnxTensor tensor =  transferTensor(env,dst,inputCount,inputChannel,inputWidth,inputHeight);

        // 推理
        OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));

        // 获取四个输出
        OnnxTensor out0 = (OnnxTensor)result.get(0);
        OnnxTensor out1 = (OnnxTensor)result.get(1);
        OnnxTensor out2 = (OnnxTensor)result.get(2);
        OnnxTensor out3 = (OnnxTensor)result.get(3);
        OnnxTensor out4 = (OnnxTensor)result.get(4);

        float[][] boxs = (float[][])out0.getValue();
        long[] boxLabels = (long[])out1.getValue();
        float[] boxScores = (float[])out2.getValue();
        float[][][] points = (float[][][])out3.getValue();
        float[][] pointsScores = (float[][])out4.getValue();


        // 遍历每个边框保存为目标集合
        int count = boxs.length;
        List<Detection> detections = new ArrayList<>();

        // 遍历每个目标
        for(int i=0;i<count;i++){
            float[] xyxy = boxs[i];// 边框坐标
            long label = boxLabels[i];// 边框类别
            float box_score = boxScores[i];// 边框得分
            float[][] point_set = points[i];// 关键点集合
            float[] point_score_set = pointsScores[i];// 关键点得分集合
            Detection detection = new Detection(xyxy,label,box_score);
            // 遍历每个关键点
            for(int j=0;j<point_set.length;j++){
                float[] pxy = point_set[j];//每个关键点
                float pscore = point_score_set[j];//每个关键点的得分
                // 目标添加点 pxy是三维的分别是 x 和 y 和 可见性
                detection.addPoint(new Point(pxy[0],pxy[1],pscore,pxy[2]));
            }
            detections.add(detection);
        }

        // 根据得分过滤 目标框
        filter1(detections,0.5f);

        // 根据nms过滤 目标框
        filter2(detections,0.3f);

        // 根据得分过滤 关键点
        filter3(detections,1f);

        // 标注边框和点
        showBoxAndPoint(src,detections,src.width(),src.height(),inputWidth,inputHeight);



    }



}
