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
import java.util.*;
import java.util.List;

/**
*   @desc : 车牌检测 + 车牌字符/颜色识别
*   @auth : tyf
*   @date : 2022-04-26  18:06:14
*/
public class yolov5_car_plate {

    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 记录一个图片的信息
    public static class ImageObj{
        // 图片模型尺寸用于推理
        Mat src;
        // 图片原始尺寸用于绘图
        Mat background;
        // 过滤后的边框信息
        List<float[]> data;
        // 投影变换后的车牌矩阵
        List<Mat> platesMat = new ArrayList<>();
        // 投影变换后的车牌
        List<String> platesStr = new ArrayList<>();
        // 车牌的颜色
        List<Color> platesColor = new ArrayList<>();
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        // 投影变换后车牌的宽高,也就是第二个模型的输入尺寸
        int plateWidth = 168;
        int plateHeight = 48;
        // 原始图片尺寸,也就是第一个模型的输入尺寸
        int picWidth = 640;
        int picHeight = 640;
        // 车牌类别
        char[] plateChar = new char[]{
                '#','京','沪','津','渝','冀','晋','蒙','辽','吉',
                '黑','苏','浙','皖','闽','赣','鲁','豫','鄂','湘',
                '粤','桂','琼','川','贵','云','藏','陕','甘','青',
                '宁','新','学','警','港','澳','挂','使','领','民',
                '航','危','0','1','2','3','4','5','6','7',
                '8','9','A','B','C','D','E','F','G','H',
                'J','K','L','M','N','P','Q','R','S','T',
                'U','V','W','X','Y','Z','险','品',
        };
        // 车牌颜色类别 color=['黑色','蓝色','绿色','白色','黄色']
        Color [] plateScalar = new Color []{
                Color.BLACK,
                Color.BLUE,
                Color.GREEN,
                Color.WHITE,
                Color.YELLOW
        };
        // 宽高缩放比
        float wScale;
        float hScale;
        public ImageObj(String img) {
            // 原始图像
            this.background = readImg(img);
            // 缩放过后的图像
            this.src = resizeWithoutPadding(this.background,this.picWidth,this.picHeight);
            // 保存缩放比
            this.wScale = Float.valueOf(src.width())/ Float.valueOf(background.width());
            this.hScale = Float.valueOf(src.height())/Float.valueOf(background.height());
        }
        public void setDataAndFilter(float[][] output){

            // xywh  objscore   class1 class2  x1y1 x2y2 x3y3 x4y4

            float confThreshold = 0.75f;
            float nmsThreshold = 0.45f; // 车牌识别省略nms

            List<float[]> temp = new ArrayList<>();

            // 置信度过滤
            for(int i=0;i<output.length;i++){
                float[] obj = output[i];
                float x = obj[0];
                float y = obj[1];
                float w = obj[2];
                float h = obj[3];
                float score = obj[4];
                float x1 = obj[5];
                float y1 = obj[6];
                float x2 = obj[7];
                float y2 = obj[8];
                float x3 = obj[9];
                float y3 = obj[10];
                float x4 = obj[11];
                float y4 = obj[12];
                float class1 = obj[13];
                float class2 = obj[14];
                if(score>=confThreshold){
                    // 边框坐标
                    float[] xyxy = xywh2xyxy(new float[]{x,y,w,h},this.picWidth,this.picHeight);
                    // 类别1或者2
                    float clazz = class1>class2?1:2;
                    // 类别概率
                    float clazzScore =  class1>class2?class1:class2;
                    // 关键点坐标
                    temp.add(new float[]{
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3], x1, y1, x2, y2, x3, y3, x4, y4,clazz,clazzScore
                    });
                }
            }

            // 交并比过滤
            // 先按照概率排序
            temp.sort((o1, o2) -> Float.compare(o2[13],o1[13]));

            // 保存最终的过滤结果
            List<float[]> out = new ArrayList<>();
            while (!temp.isEmpty()){
                float[] max = temp.get(0);
                out.add(max);
                Iterator<float[]> it = temp.iterator();
                while (it.hasNext()) {
                    float[] obj = it.next();
                    // 交并比
                    double iou = calculateIoU(
                            new float[]{max[0],max[1],max[2],max[3]},
                            new float[]{obj[0],obj[1],obj[2],obj[3]}
                    );
                    if (iou > nmsThreshold) {
                        it.remove();
                    }
                }
            }
            // 保存最终的边框
            this.data = out;
        }

        // 对所有车牌关键点进行透视变换,拉成一个矩形
        public void transform(){

            // 首先对每个车牌目标进行关键点透视变换

            this.data.stream().forEach(n->{

                float key_point_x1 = n[4];
                float key_point_y1 = n[5];
                float key_point_x2 = n[6];
                float key_point_y2 = n[7];
                float key_point_x3 = n[8];
                float key_point_y3 = n[9];
                float key_point_x4 = n[10];
                float key_point_y4 = n[11];

                Point[] srcPoints = new Point[4];
                Point p1 = new Point(Float.valueOf(key_point_x1).intValue(), Float.valueOf(key_point_y1).intValue());
                Point p2 = new Point(Float.valueOf(key_point_x2).intValue(), Float.valueOf(key_point_y2).intValue());
                Point p3 = new Point(Float.valueOf(key_point_x3).intValue(), Float.valueOf(key_point_y3).intValue());
                Point p4 = new Point(Float.valueOf(key_point_x4).intValue(), Float.valueOf(key_point_y4).intValue());
                srcPoints[0] = p1;
                srcPoints[1] = p2;
                srcPoints[2] = p3;
                srcPoints[3] = p4;

                // 定义透视变换后的目标矩形的四个角点,指定车牌的宽和高
                Point[] dstPoints = new Point[4];
                dstPoints[0] = new Point(0, 0);
                dstPoints[1] = new Point(plateWidth, 0);
                dstPoints[2] = new Point(plateWidth, plateHeight);
                dstPoints[3] = new Point(0, plateHeight);

                // 计算透视变换矩阵
                MatOfPoint2f in1 = new MatOfPoint2f(srcPoints);
                MatOfPoint2f in2 = new MatOfPoint2f(dstPoints);
                Mat M = Imgproc.getPerspectiveTransform(in1, in2);

                // 进行透视变换
                Mat warped = new Mat();
                Imgproc.warpPerspective(src, warped, M, new Size(plateWidth, plateHeight));

                // 保存透视变换得到的车牌
                platesMat.add(warped);

            });


        }

        public void drawBox(){

            // 在原始图片尺寸上绘制,需要坐标转换

            // 遍历每个车牌框
            for(int i=0; i<this.data.size() ; i++ ){

                float[] n = data.get(i);

                // 位置信息
                float x1 = n[0] / wScale;
                float y1 = n[1] / hScale;
                float x2 = n[2] / wScale;
                float y2 = n[3] / hScale;
                float key_point_x1 = n[4] / wScale;
                float key_point_y1 = n[5] / hScale;
                float key_point_x2 = n[6] / wScale;
                float key_point_y2 = n[7] / hScale;
                float key_point_x3 = n[8] / wScale;
                float key_point_y3 = n[9] / hScale;
                float key_point_x4 = n[10] / wScale;
                float key_point_y4 = n[11] / hScale;
                float clazz = n[12];
                float clazzScore = n[13];

                // 画边框
                Imgproc.rectangle(
                        background,
                        new Point(Float.valueOf(x1).intValue(), Float.valueOf(y1).intValue()),
                        new Point(Float.valueOf(x2).intValue(), Float.valueOf(y2).intValue()),
                        color1,
                        2);
                // 画关键点四个
                Imgproc.circle(
                        background,
                        new Point(Float.valueOf(key_point_x1).intValue(), Float.valueOf(key_point_y1).intValue()),
                        3, // 半径
                        color2,
                        2);
                Imgproc.circle(
                        background,
                        new Point(Float.valueOf(key_point_x2).intValue(), Float.valueOf(key_point_y2).intValue()),
                        3, // 半径
                        color2,
                        2);
                Imgproc.circle(
                        background,
                        new Point(Float.valueOf(key_point_x3).intValue(), Float.valueOf(key_point_y3).intValue()),
                        3, // 半径
                        color2,
                        2);
                Imgproc.circle(
                        background,
                        new Point(Float.valueOf(key_point_x4).intValue(), Float.valueOf(key_point_y4).intValue()),
                        3, // 半径
                        color2,
                        2);

                // 获取车牌
                String number = platesStr.get(i);
            }
        }
    }

    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型[1]输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[1]输出-----------");
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


    // 环境初始化
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型[2]输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[2]输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });


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

    // 将图片矩阵转化为 onnxruntime 需要的 tensor
    // 根据yolo的输入张量的预处理,需要进行归一化、BGR -> RGB 等超做 具体可以看 detect.py 脚本
    public static OnnxTensor transferTensor(Mat dst,int channels,int netWidth,int netHeight){

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
            tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }

    public static OnnxTensor transferTensor2(Mat dst,int channels,int netWidth,int netHeight){

        // BGR -> RGB
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

        // 归一化 , 这一步前处理主要和原来的训练方式有关
        // 首先将像素值除以255，将像素值缩放到[0,1]的范围内。
        // 将像素值减去均值0.588，使得像素值中心化。
        // 将像素值除以标准差0.193，使得像素值归一化。
        double[] meanValue = {0.588, 0.588, 0.588};
        double[] stdValue = {0.193, 0.193, 0.193};

        // Convert image to float and normalize using mean and standard deviation values
        dst.convertTo(dst, CvType.CV_32FC3, 1.0 / 255.0);
        Core.subtract(dst, new Scalar(meanValue), dst);
        Core.divide(dst, new Scalar(stdValue), dst);

        // 初始化一个输入数组 channels * netWidth * netHeight
        float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
        dst.get(0, 0, whc);

        // 得到最终的图片转 float 数组
        float[] chw = whc2cwh(whc);

        // 创建 onnxruntime 需要的 tensor
        // 传入输入的图片 float 数组并指定数组shape
        OnnxTensor tensor = null;
        try {
            tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }

    // 计算两个框的交并比
    private static double calculateIoU(float[] box1, float[] box2) {

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

    public static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
    public static Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(netWidth, netHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }


    // 车牌检测,以及4个关键点
    public static void doDetect(ImageObj imageObj) throws Exception{

        // 输入矩阵
        Mat in = imageObj.src.clone();
        // 转为tensor
        OnnxTensor tensor = transferTensor(in,3,imageObj.picWidth,imageObj.picHeight);
        // 推理
        OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));
        // 解析 output -> [1, 25200, 15] -> FLOAT
        float[][] data = ((float[][][])(res.get(0)).getValue())[0];
        // 根据置信度、交并比过滤
        imageObj.setDataAndFilter(data);


    }


    // 识别车牌
    public static void doRecect(ImageObj imageObj){

        // 先将关键点透视变换为矩形方便识别,目标尺寸就是第二个模型的输入 168*48
        imageObj.transform();

        // 第二个模型是crnn 输入投影变换后的车牌图片即可
        imageObj.platesMat.stream().forEach(plate->{

            try {
                // 转为模型输入,主要归一化处理时减均值再除标准差,和普通的除255归一化不一样
                OnnxTensor tensor = transferTensor2(plate.clone(),3,plate.width(),plate.height());
                // 推理
                OrtSession.Result res = session2.run(Collections.singletonMap("images", tensor));
                // 21*78 21表示最多生成21个字符,78表示每个字符的概率
                float[][] data1 = ((float[][][])(res.get(0)).getValue())[0];
                // 遍历每个字符
                char last = '-';
                List<Character> chars = new ArrayList<>();
                for(int i=0;i<data1.length;i++){
                    // 每个字符概率最大值下标
                    int maxIndex = getMaxIndex(data1[i]);
                    // 每个字符概率最大值的char
                    char maxName = imageObj.plateChar[maxIndex];
                    if( maxIndex!=0 && maxName!=last ){
                        chars.add(maxName);
                    }
                    last = maxName;
                }

                StringBuffer car = new StringBuffer();
                chars.stream().forEach(n->{
                    car.append(n);
                });
                imageObj.platesStr.add(car.toString());


                // 5 代表五个颜色
                float[] data2 = ((float[][])(res.get(1)).getValue())[0];
                int maxIndex = getMaxIndex(data2);
                Color color = imageObj.plateScalar[maxIndex];// 从类别下表中查找
                imageObj.platesColor.add(color);
            }
            catch (Exception e){
                e.printStackTrace();
            }

        });

    }


    // 弹窗显示所有信息
    public static void showJpanel(ImageObj img){

        JFrame frame = new JFrame("Car");

        // 一行两列
        JPanel parent = new JPanel();

        // 显示图片
        JPanel p1 = new JPanel();
        p1.add(new JLabel(new ImageIcon(mat2BufferedImage(img.background))));

        // 显示车牌子图片
        JPanel p2 = new JPanel(new FlowLayout(FlowLayout.LEFT, 20, 20));
        JPanel sub = new JPanel(new GridLayout(img.platesMat.size()+1, 1, 0, 5));
//        sub.setLayout(new BoxLayout(sub, BoxLayout.Y_AXIS));
        JPanel title = new JPanel(new GridLayout(1,3,10,10));
        JLabel label1 = new JLabel("投影变换");
        label1.setHorizontalAlignment(JLabel.CENTER);
        title.add(label1);
        JLabel label2 = new JLabel("车牌号");
        label2.setHorizontalAlignment(JLabel.CENTER);
        title.add(label2);
        JLabel label3 = new JLabel("颜色");
        label3.setHorizontalAlignment(JLabel.CENTER);
        title.add(label3);
        sub.add(title);
        for(int i=0;i<img.platesMat.size();i++){
            // 每个车牌占一行
            JPanel line = new JPanel(new GridLayout(1,3,10,10));
            // 车牌图片
            JLabel jLabel1 = new JLabel(new ImageIcon(mat2BufferedImage(img.platesMat.get(i))));
            // 车牌号
            JLabel jLabel2 = new JLabel(img.platesStr.get(i));
            // 车牌颜色
            JLabel jLabel3 = new JLabel("█");
            jLabel3.setForeground(img.platesColor.get(i));
            // 居中
            jLabel1.setHorizontalAlignment(JLabel.CENTER);
            jLabel2.setHorizontalAlignment(JLabel.CENTER);
            jLabel3.setHorizontalAlignment(JLabel.CENTER);
            line.add(jLabel1);
            line.add(jLabel2);
            line.add(jLabel3);
            sub.add(line);
        }
        p2.add(sub);

        parent.add(p1);
        parent.add(p2);

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(parent);
        frame.pack();
        frame.setVisible(true);

    }


    public static void main(String[] args) throws Exception{

        // 原文链接
        // http://t.csdn.cn/aAFzc

        // 模型初始化 车牌检测、车牌识别
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_car_plate\\plate_detect.onnx");
        init2(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_car_plate\\plate_rec_color.onnx");

        // 原始图片
        ImageObj img = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_car_plate\\car.png");

        // 车牌区域检测
        doDetect(img);

        // 车牌识别
        doRecect(img);

        // 原图绘制边框
        img.drawBox();

        // 弹窗显示
        showJpanel(img);

    }



}
