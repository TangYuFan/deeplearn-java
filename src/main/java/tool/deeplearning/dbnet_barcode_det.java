package tool.deeplearning;


import ai.onnxruntime.*;
import javafx.animation.Animation;
import javafx.animation.RotateTransition;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.geometry.Point3D;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Sphere;
import javafx.scene.transform.Rotate;
import javafx.util.Duration;
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
import java.util.concurrent.atomic.AtomicBoolean;

/**
*   @desc : dbnet 检测条形码
 *
 *
*   @auth : tyf
*   @date : 2022-05-23  15:50:19
*/
public class dbnet_barcode_det {

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
        public ImageObj(String image) throws Exception{
            this.src = this.resizeWithoutPadding(this.readImg(image),736,736);
            this.dst = src.clone();
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public OnnxTensor transferTensor(Mat dst,int channels,int netWidth,int netHeight){
            // 只需要做 BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            // 转为浮点
            dst.convertTo(dst, CvType.CV_32FC1);

            // 归一化
            dst.convertTo(dst, CvType.CV_32FC3, 1.0 / 255.0);

            double[] meanValue = {0.485, 0.456, 0.406};
            double[] stdValue = {0.229, 0.224, 0.225};

            Core.subtract(dst, new Scalar(meanValue), dst);
            Core.divide(dst, new Scalar(stdValue), dst);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ channels * netWidth * netHeight ];
            dst.get(0, 0, whc);
            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);
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

        public Mat resizeWithPadding(Mat src, int netWidth, int netHeight) {
            Mat dst = new Mat();
            int oldW = src.width();
            int oldH = src.height();
            double r = Math.min((double) netWidth / oldW, (double) netHeight / oldH);
            int newUnpadW = (int) Math.round(oldW * r);
            int newUnpadH = (int) Math.round(oldH * r);
            int dw = (Long.valueOf(netWidth).intValue() - newUnpadW) / 2;
            int dh = (Long.valueOf(netHeight).intValue() - newUnpadH) / 2;
            int top = (int) Math.round(dh - 0.1);
            int bottom = (int) Math.round(dh + 0.1);
            int left = (int) Math.round(dw - 0.1);
            int right = (int) Math.round(dw + 0.1);
            Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
            Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);
            return dst;
        }


        // 执行推理
        public void run() throws Exception{


            // 原始图片的备份
            Mat back = dst.clone();

            // ---------模型1输入-----------
            // input -> [1, 3, 736, 736] -> FLOAT
            OnnxTensor ten = transferTensor(dst,3,dst.width(),dst.height());

            // ---------模型1输出-----------
            // output -> [736, 736] -> FLOAT
            OrtSession.Result res = session1.run(Collections.singletonMap("input", ten));
            float[][] output = ((float[][])(res.get(0)).getValue());

            // 根据阈值进行二值化
            float binary_threshold = 0.3f;
            // 设置颜色为白色（255, 255, 255）
            double[] white = {255, 255, 255};

            // 遍历宽高,去掉原始图片的非二维码区域
            for(int i=0;i<736;i++){
                for(int j=0;j<736;j++){
                    float value = output[i][j];
                    // 设置为白色
                    if(value<binary_threshold){
                        back.put(i, j, white);
                    }
                }
            }

            // 后续就是根据opencv  findcounter 查找矩形区域
            // TODO

            // 保存结果
            this.dst.release();
            this.dst = back;


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

            // 弹窗显示
            JFrame frame = new JFrame("Hand");
            frame.setSize(dst.width(), dst.height());

            JPanel all = new JPanel();
            all.add(new JLabel(new ImageIcon(mat2BufferedImage(src))));
            all.add(new JLabel(new ImageIcon(mat2BufferedImage(dst))));

            frame.getContentPane().add(all);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.pack();

        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // input -> [1, 3, 736, 736] -> FLOAT
        // ---------模型1输出-----------
        // output -> [736, 736] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\dbnet_barcode_det\\model_0.88_depoly.onnx");

        // 图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\dbnet_barcode_det\\pic.jpg";
        ImageObj imageObj = new ImageObj(pic);

        // 显示
        imageObj.show();


    }
}
