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
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;

/**
*   @desc : 身份证全卡面文字识别
 *
 *          模型：
 *          dbnet.onnx 检测证件中的文字区域
 *          crnn_lite_lstm.onnx 识别文字
 *
*   @auth : tyf
*   @date : 2022-05-23  15:18:19
*/
public class id_card_licence {

    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    public static OrtEnvironment env2;
    public static OrtSession session2;

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
        System.out.println("---------模型输出-----------");
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
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
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


    public static class ImageObj{
        // 原始图片
        Mat src;
        public ImageObj(String pic){
            this.src = this.readImg(pic);
            // 识别文本区域
            this.doDec();
            // 识别文字
            this.doRec();
        }
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(netWidth, netHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }
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
        public static float[] whc2chw(float[] src) {
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

        // 识别文本区域
        public void doDec(){

            // 宽高需要是32的倍数
            int w = 960;
            int h = 640;

            Mat input = resizeWithoutPadding(src.clone(),w,h);
            Mat back = input.clone();

            // BGR -> RGB
            Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);

            // 归一化
            input.convertTo(input, CvType.CV_32FC1, 1. / 255);

            // 减去均值再除以均方差
            double[] meanValue = {0.485, 0.456, 0.406};
            double[] stdValue = {0.229, 0.224, 0.225};
            Core.subtract(input, new Scalar(meanValue), input);
            Core.divide(input, new Scalar(stdValue), input);

            float[] whc = new float[ 3 * w * h ];
            input.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2chw(whc);

            try {
                // ---------模型1输入-----------
                // input0 -> [1, 3, -1, -1] -> FLOAT
                // ---------模型输出-----------
                // out1 -> [1, 1, 320, 320] -> FLOAT
                OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,h,w});
                OrtSession.Result res = session1.run(Collections.singletonMap("input0", tensor));

                // 1 * h * w  => 1 * 640 * 960
                float[][] data = ((float[][][][])(res.get(0)).getValue())[0][0];

                // 显示一下模型输出
                float thresh = 0.8f;
//                showDetectionRes(data,w,h,thresh);

                // 查找边框
                Mat mat = new Mat(w,h, CvType.CV_8U);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        mat.put(y, x, data[y][x] > thresh ? 255 : 0 );
                    }
                }

                // 查找所有轮廓并保存到 contours 中
                ArrayList<MatOfPoint> contours = new ArrayList<>();
                Mat hierarchy = new Mat();
                Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                // 计算每个轮廓的最小外接矩形
                ArrayList<Point[]> contours_points = new ArrayList<>();
                for (int index = 0; index < contours.size(); index++) {
                    MatOfPoint contour = contours.get(index);
                    // 最小外交矩形四个点保存到 points 中
                    RotatedRect boundingBox = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
                    // 这个是求最大外接矩形
                    Point[] points = new Point[4];
                    boundingBox.points(points);
                    // 保存过滤后的最小外接矩形边框坐标
                    contours_points.add(points);
                }

                // 边框四个点排序
                ArrayList<Point[]> contours_points_sort = new ArrayList<>();
                // 便利每个边框
                contours_points.stream().forEach(n->{
                    // 找到 x+y 最小的  左上
                    Point p1 = Arrays.stream(n).min(Comparator.comparingDouble(o -> o.x + o.y)).get();
                    // 找到 x+y 最大的  右下
                    Point p2 = Arrays.stream(n).max(Comparator.comparingDouble(o -> o.x + o.y)).get();
                    // 找到 x 最大的  右上
                    Point p3 = Arrays.stream(n).filter(point -> point!=p1&&point!=p2).max(Comparator.comparingDouble(o -> o.x)).get();
                    // 找到 x 最小  左下
                    Point p4 = Arrays.stream(n).filter(point -> point!=p1&&point!=p2).min(Comparator.comparingDouble(o -> o.x)).get();
                    contours_points_sort.add(new Point[]{
                            // 左上、左下、右上、右下
                            p1,p4,p3,p2
                    });
                });

                // 对边框进行扩展,并限制在图片整体区域内,不扩展的话可能边框只能框住文字的一部分,扩展的点不能超出原始图片
                ArrayList<Point[]> contours_points_sort_unclip = this.contoursPointsUnclip(contours_points_sort,w,h);

                // 弹窗显示
                showContours(back,contours_points_sort_unclip);

            }



            catch (Exception e){
                e.printStackTrace();
            }
        }

        // 显示模型1的输出二值化查找轮廓的结果
        public void showContours(Mat back,ArrayList<Point[]> points){

            // 复制一个原始尺寸的图片用于标注
            Mat show = back.clone();

            Scalar color1 = new Scalar(0, 0, 255);
            Scalar color2 = new Scalar(0, 255, 0);

            // 先将所有框按照p1坐标排个序
            points.stream().sorted(Comparator.comparingDouble(o -> o[0].x + o[0].y));

            // 遍历所有框
            AtomicInteger index = new AtomicInteger(1);
            points.stream().forEach(n->{
                // 左上、左下、右上、右下
                Point p1 = n[0];
                Point p2 = n[1];
                Point p3 = n[2];
                Point p4 = n[3];
                // 画线,注意这里不是直接使用p2p4矩形画框,因为这个矩形四个点是倾斜的,p2p4画框是和图片平行的
                Imgproc.line(show, new Point(p1.x ,p1.y ), new Point(p2.x ,p2.y ), color1, 2);
                Imgproc.line(show, new Point(p3.x ,p3.y ), new Point(p4.x ,p4.y ), color1, 2);
                Imgproc.line(show, new Point(p1.x ,p1.y ), new Point(p3.x ,p3.y ), color1, 2);
                Imgproc.line(show, new Point(p2.x ,p2.y ), new Point(p4.x ,p4.y ), color1, 2);
                // 标注一下序号
                Imgproc.putText(show, String.valueOf(index.get()) , new Point(p1.x ,p1.y ), Imgproc.FONT_HERSHEY_SIMPLEX, 1, color2, 2);
                // 画四个点
//                Imgproc.circle(show, new Point(p1.x * w_scala,p1.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p2.x * w_scala,p2.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p3.x * w_scala,p3.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p4.x * w_scala,p4.y * h_scala), 2, color2, 1);

                index.getAndIncrement();
            });

            // 弹窗显示
            JFrame frame = new JFrame();
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(mat2BufferedImage(show))));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }

        // 对边界框进行扩展,防止文字被拦腰截断
        public static ArrayList<Point[]> contoursPointsUnclip(ArrayList<Point[]> contours_points,int max_w,int max_h){

            ArrayList<Point[]> tmp = new ArrayList<>();

            // 原始图片面积
            double src_area = max_w * max_h;

            contours_points.forEach(n -> {

                // 左上、左下、右上、右下
                Point p1 = n[0];
                Point p2 = n[1];
                Point p3 = n[2];
                Point p4 = n[3];

                // 计算最小的一条边的边长
                double h = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                double w = Math.sqrt(Math.pow(p4.x - p2.x, 2) + Math.pow(p4.y - p2.y, 2));

                // 区域面积
                double area = h * w;

                // 缩放比例
                double ratio = 1.7;


//                // 宽高比是否接近1
//                if( w/h >= 0.8 && w/h <= 1.3){
//                    // 区域面积占比原图面积很大说明多个文字.多个文字宽高很大需要缩小缩放比例
//                    if( area/src_area >= 0.001){
//                        ratio =  0.55;
//                    }
//                    // 说明文字很少
//                    else{
//                        ratio = 0.5;
//                    }
//                }
//                // 不接近1
//                else{
//                    // 区域面积占比原图面积很大则缩小缩放距离
//                    if( area/src_area >= 0.001){
//                        ratio =  0.3;
//                    }else{
//                        ratio = 0.6;
//                    }
//                }

                double dis = Math.min(w,h) * ratio;

                // 返回扩展的坐标
                tmp.add(new Point[]{
                        new Point(p1.x-dis,p1.y-dis), // 左上、
                        new Point(p2.x-dis,p2.y+dis), // 左下、
                        new Point(p3.x+dis,p3.y-dis), // 右上、
                        new Point(p4.x+dis,p4.y+dis), // 右下
                });

            });

            return tmp;
        }

        // 识别文字
        public void doRec(){

            // ---------模型2输入-----------
            // input -> [-1, 3, 32, -1] -> FLOAT
            // ---------模型2输出-----------
            // out -> [64, 1, 5531] -> FLOAT


        }

        // 显示模型1的输出
        public void showDetectionRes(float[][] data,int w,int h,float thresh){

            BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);

            System.out.println("xxx:"+data.length+",mmm:"+data[0].length);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    // 设置灰度值
                    float d = data[y][x];
                    if( d > thresh){
                        // 设置白色
                        img.setRGB(x, y, Color.WHITE.getRGB());
                    }
                    else{
                        // 设置黑色
                        img.setRGB(x, y, Color.BLACK.getRGB());
                    }
                }
            }

            // 弹窗显示
            JFrame frame = new JFrame();
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(img)));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);

        }

        public void show(){

        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // input0 -> [1, 3, -1, -1] -> FLOAT
        // ---------模型输出-----------
        // out1 -> [1, 1, 320, 320] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_licence\\dbnet.onnx");

        // ---------模型2输入-----------
        // input -> [-1, 3, 32, -1] -> FLOAT
        // ---------模型2输出-----------
        // out -> [64, 1, 5531] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_licence\\crnn_lite_lstm.onnx");


        // 注意模型输入的图片需要先对身份证进行检测画框后截图(且仿射变换类似于通过四个角关键点),因为身份证外背景如果包含文字也会识别出来
        // 这里省略了这一步所以要求输入的身份证照片必须截取后进行输入
        // 也可以使用证件照检测模型 faster_rcnn_card_det : https://github.com/hpc203/faster-rcnn-card-opencv

        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_licence\\id_card.png";
        ImageObj imageObj = new ImageObj(img);


        // 显示
        imageObj.show();


    }

}
