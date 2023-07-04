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
 *  @Desc: E2Pose人体关键点检测
 *              github:https://github.com/hpc203/E2Pose-detect-onnxrun-cpp-py
 *              20个不同尺寸的onnx:链接：https://pan.baidu.com/s/1JEks5KvpJzOwgFsi_qVLAw 提取码：mvz7
 *  @Date: 2022-05-08 21:09:32
 *  @auth: TYF
 */
public class e2_pose_detection {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());

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
        session.getMetadata().getCustomMetadata().entrySet().forEach(n->{
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
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        Scalar color3 = new Scalar(255, 0, 0);
        Scalar color4 = new Scalar(255, 255, 0);
        Scalar color5 = new Scalar(0, 255, 255);
        Scalar color6 = new Scalar(255, 0, 255);
        Scalar color7 = new Scalar(255, 128, 0);
        Scalar color8 = new Scalar(128, 0, 255);
        Scalar color9 = new Scalar(0, 255, 128);
        Scalar color10 = new Scalar(128, 255, 0);
        Scalar color11 = new Scalar(255, 0, 128);
        Scalar color12 = new Scalar(0, 128, 255);
        Scalar color13 = new Scalar(128, 128, 0);
        Scalar color14 = new Scalar(0, 128, 128);
        Scalar color15 = new Scalar(128, 0, 128);
        Scalar color16 = new Scalar(192, 192, 192);
        Scalar color17 = new Scalar(255, 255, 255);
        // 每个人17个点,也就是17倍数
        ArrayList<float[]> points = new ArrayList<>();
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src.clone(),512,512);
            this.tensor = this.transferTensor(this.dst.clone(),3,512,512); // 转张量
            this.run(); // 执行推理
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

            //  浮点
            dst.convertTo(dst, CvType.CV_32FC1);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);

            // 创建 onnxruntime 需要的 tensor
            // 传入输入的图片 float 数组并指定数组shape
            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,channels,netWidth,netHeight});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }


        // 执行推理
        public void run(){
            try {

                OrtSession.Result res = session.run(Collections.singletonMap("inputimg", tensor));


                // kvxy/concat -> [1, 341, 17, 3] -> FLOAT
                float[][][] kvxy = ((float[][][][])(res.get(0)).getValue())[0];
                // pv/concat -> [1, 341, 1, 1] -> FLOAT
                float[][][] pv = ((float[][][][])(res.get(1)).getValue())[0];

                int num_proposal = kvxy.length;
                int num_pts = kvxy[0].length;


                float confThreshold = 0.5f;


                // 341 个关键点
                for (int i = 0; i < num_proposal; i++) {
                    // pv 是 341, 1, 1 也就是341个关键点的置信度
                    if (pv[i][0][0] >= confThreshold) {
                        // 每个关键点有17中可能（17中关键点）脸5+手腕2+胳膊2+肩膀2+跨2+膝盖2+脚踝2
                        for (int j = 0; j < num_pts; j++) {
                            float score = kvxy[i][j][0] * 2;
                            if (score >= confThreshold) {
                                float x = kvxy[i][j][1] * dst.cols();
                                float y = kvxy[i][j][2] * dst.rows();
                                // 关键点
                                points.add(new float[]{x,y});
                            }
                        }
                    }
                }


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



        // 弹窗显示
        public void show(){

            if(points.size()%17==0){

                int count = points.size()/17;

                // 每个人17个点
                for(int i=0;i<count;i++){

                    float[] p1 = points.get(i*17+0);
                    float[] p2 = points.get(i*17+1);
                    float[] p3 = points.get(i*17+2);
                    float[] p4 = points.get(i*17+3);
                    float[] p5 = points.get(i*17+4);
                    float[] p6 = points.get(i*17+5);
                    float[] p7 = points.get(i*17+6);
                    float[] p8 = points.get(i*17+7);
                    float[] p9 = points.get(i*17+8);
                    float[] p10 = points.get(i*17+9);
                    float[] p11 = points.get(i*17+10);
                    float[] p12 = points.get(i*17+11);
                    float[] p13 = points.get(i*17+12);
                    float[] p14 = points.get(i*17+13);
                    float[] p15 = points.get(i*17+14);
                    float[] p16 = points.get(i*17+15);
                    float[] p17 = points.get(i*17+16);

                    int p1_x = Float.valueOf(p1[0]).intValue();
                    int p1_y = Float.valueOf(p1[1]).intValue();

                    int p2_x = Float.valueOf(p2[0]).intValue();
                    int p2_y = Float.valueOf(p2[1]).intValue();

                    int p3_x = Float.valueOf(p3[0]).intValue();
                    int p3_y = Float.valueOf(p3[1]).intValue();

                    int p4_x = Float.valueOf(p4[0]).intValue();
                    int p4_y = Float.valueOf(p4[1]).intValue();

                    int p5_x = Float.valueOf(p5[0]).intValue();
                    int p5_y = Float.valueOf(p5[1]).intValue();

                    int p6_x = Float.valueOf(p6[0]).intValue();
                    int p6_y = Float.valueOf(p6[1]).intValue();

                    int p7_x = Float.valueOf(p7[0]).intValue();
                    int p7_y = Float.valueOf(p7[1]).intValue();

                    int p8_x = Float.valueOf(p8[0]).intValue();
                    int p8_y = Float.valueOf(p8[1]).intValue();

                    int p9_x = Float.valueOf(p9[0]).intValue();
                    int p9_y = Float.valueOf(p9[1]).intValue();

                    int p10_x = Float.valueOf(p10[0]).intValue();
                    int p10_y = Float.valueOf(p10[1]).intValue();

                    int p11_x = Float.valueOf(p11[0]).intValue();
                    int p11_y = Float.valueOf(p11[1]).intValue();

                    int p12_x = Float.valueOf(p12[0]).intValue();
                    int p12_y = Float.valueOf(p12[1]).intValue();

                    int p13_x = Float.valueOf(p13[0]).intValue();
                    int p13_y = Float.valueOf(p13[1]).intValue();

                    int p14_x = Float.valueOf(p14[0]).intValue();
                    int p14_y = Float.valueOf(p14[1]).intValue();

                    int p15_x = Float.valueOf(p15[0]).intValue();
                    int p15_y = Float.valueOf(p15[1]).intValue();

                    int p16_x = Float.valueOf(p16[0]).intValue();
                    int p16_y = Float.valueOf(p16[1]).intValue();

                    int p17_x = Float.valueOf(p17[0]).intValue();
                    int p17_y = Float.valueOf(p17[1]).intValue();


                    Point pp1 = new Point(p1_x,p1_y);
                    Point pp2 = new Point(p2_x,p2_y);
                    Point pp3 = new Point(p3_x,p3_y);
                    Point pp4 = new Point(p4_x,p4_y);
                    Point pp5 = new Point(p5_x,p5_y);
                    Point pp6 = new Point(p6_x,p6_y);
                    Point pp7 = new Point(p7_x,p7_y);
                    Point pp8 = new Point(p8_x,p8_y);
                    Point pp9 = new Point(p9_x,p9_y);
                    Point pp10 = new Point(p10_x,p10_y);
                    Point pp11 = new Point(p11_x,p11_y);
                    Point pp12 = new Point(p12_x,p12_y);
                    Point pp13 = new Point(p13_x,p13_y);
                    Point pp14 = new Point(p14_x,p14_y);
                    Point pp15 = new Point(p15_x,p15_y);
                    Point pp16 = new Point(p16_x,p16_y);
                    Point pp17 = new Point(p17_x,p17_y);

                    // 鼻子
                    Imgproc.circle(dst, pp1, 2, color1, 2);
                    // 右眼
                    Imgproc.circle(dst, pp2, 2, color2, 2);
                    // 左眼
                    Imgproc.circle(dst, pp3, 2, color3, 2);
                    // 右耳
                    Imgproc.circle(dst, pp4, 2, color4, 2);
                    // 左边耳
                    Imgproc.circle(dst, pp5, 2, color5, 2);
                    // 右肩
                    Imgproc.circle(dst, pp6, 2, color6, 2);
                    // 左肩
                    Imgproc.circle(dst, pp7, 2, color7, 2);
                    // 右胳膊
                    Imgproc.circle(dst, pp8, 2, color8, 2);
                    // 左胳膊
                    Imgproc.circle(dst, pp9, 2, color9, 2);
                    // 右手
                    Imgproc.circle(dst, pp10, 2, color10, 2);
                    // 左手
                    Imgproc.circle(dst, pp11, 2, color11, 2);
                    // 右跨
                    Imgproc.circle(dst, pp12, 2, color12, 2);
                    // 左跨
                    Imgproc.circle(dst, pp13, 2, color13, 2);
                    // 右膝盖
                    Imgproc.circle(dst, pp14, 2, color14, 2);
                    // 左膝盖
                    Imgproc.circle(dst, pp15, 2, color15, 2);
                    // 右脚
                    Imgproc.circle(dst, pp16, 2, color16, 2);
                    // 左脚
                    Imgproc.circle(dst, pp17, 2, color17, 2);

                    // 画线
                    Imgproc.line(dst, pp12, pp14, color1, 2);
                    Imgproc.line(dst, pp14, pp16, color1, 2);
                    Imgproc.line(dst, pp13, pp15, color1, 2);
                    Imgproc.line(dst, pp15, pp17, color1, 2);

                    Imgproc.line(dst, pp6, pp8, color1, 2);
                    Imgproc.line(dst, pp8, pp10, color1, 2);
                    Imgproc.line(dst, pp7, pp9, color1, 2);
                    Imgproc.line(dst, pp9, pp11, color1, 2);

                    Imgproc.line(dst, pp6, pp7, color1, 2);
                    Imgproc.line(dst, pp12, pp13, color1, 2);
                    Imgproc.line(dst, pp6, pp12, color1, 2);
                    Imgproc.line(dst, pp7, pp13, color1, 2);

                    Imgproc.line(dst, pp1, pp6, color1, 2);
                    Imgproc.line(dst, pp1, pp7, color1, 2);

                }




            }
            else {
                System.out.println("人体不满足17点:"+points.size());
            }



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



        // ---------模型输入-----------
        // inputimg -> [1, 3, 512, 512] -> FLOAT
        // ---------模型输出-----------
        // kvxy/concat -> [1, 341, 17, 3] -> FLOAT
        // pv/concat -> [1, 341, 1, 1] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\e2_pose_detection\\e2epose_resnet50_1x3x512x512.onnx");

        // 加载图片
        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\e2_pose_detection\\pic.png";
        ImageObj image = new ImageObj(pic);

        // 显示
        image.show();

    }


}
