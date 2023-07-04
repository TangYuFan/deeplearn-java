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
*   @desc : yolov3 输入 112*112的人脸图片进行 106关键点检测
*   @auth : tyf
*   @date : 2022-05-08  09:34:34
*/
public class yolov3_face_key_point_106 {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init1(String weight) throws Exception{
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
        // 所有点
        ArrayList<float[]> points = new ArrayList<>();
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src.clone(),112,112);
            this.tensor = this.transferTensor(this.dst.clone(),3,112,112); // 转张量
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
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,channels,netWidth,netHeight});
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
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));
                // ---------模型输出-----------
                // output1 -> [1, 40, 28, 28] -> FLOAT
                // output ->  [1, 212] -> FLOAT
                float[][][] out1 = ((float[][][][])(res.get(0)).getValue())[0];
                float[] out2 = ((float[][])(res.get(1)).getValue())[0];


                // 212 也就是一共106个点
                for(int i=0;i<211;i=i+2){
                    float x = out2[i];
                    float y = out2[i+1];
                    this.points.add(new float[]{x,y});//保存一个点
                }

                System.out.println("sys:"+res);

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

            // 遍历所有点
            points.stream().forEach(n->{

                float x = n[0]*112;
                float y = n[1]*112;

                // 点
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(x).intValue(), Float.valueOf(y).intValue()),
                        1, // 半径
                        color2,
                        1);
            });

            JFrame frame = new JFrame("Image");
            frame.setSize(dst.width(), dst.height());

            // 图片转为原始大小
            BufferedImage img = mat2BufferedImage(dst);
//            img = resize(img,src.width(),src.height());

            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
    }

    public static void main(String[] args) throws Exception{


        // https://github.com/hpc203/yolov7-detect-face-onnxrun-cpp-py


        /*
        ---------模型输入-----------
        input -> [1, 3, 112, 112] -> FLOAT
        ---------模型输出-----------
        output1 -> [1, 40, 28, 28] -> FLOAT
        output -> [1, 212] -> FLOAT
         */
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov3_face_key_point_106\\v3.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov3_face_key_point_106\\face.png");

        // 显示
        image.show();

    }
}
