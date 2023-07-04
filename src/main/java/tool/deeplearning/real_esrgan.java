package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.*;
import java.util.Arrays;
import java.util.Collections;


/**
*   @desc : Real-ESRGAN - 图像超分辨率修复
 *
 *          Real-ESRGAN 是由腾讯 ARC 实验室发布的一个盲图像超分辨率模型，
 *          它的目标是开发出实用的图像/视频修复算法，Real-ESRGAN 是在 ESRGAN 的基础上使用纯合成数据来进行训练的，
 *          基本上就是通过模拟高分辨率图像变低分辩率过程中的各种退化，然后再通过低清图倒推出它的高清图，
 *          简单说你也可以把它理解为一个图像/视频修复、放大工具。
 *
 *          Real-ESRGAN 目前提供了五种模型，分别是
 *          realesrgan-x4plus（默认）、
 *          reaesrnet-x4plus、
 *          realesrgan-x4plus-anime（针对动漫插画图像优化，有更小的体积）、
 *          realesr-animevideov3（针对动漫视频）和
 *          realesrgan-x4plus-anime-6B，你可以根据你要处理的图片或视频选择合适的模型进行使用。
 *
 *          参考文档：
 *          https://blog.csdn.net/qq_32577169/article/details/127333135
 *
*   @auth : tyf
*   @date : 2022-05-23  15:09:45
*/
public class real_esrgan {


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
//        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }


    public static class ImageObj{
        // 原始图片(原始尺寸)
        Mat src;
        // 原始图片(模型尺寸的)
        Mat dst;
        // 转换结果
        BufferedImage out;
        public ImageObj(String image) throws Exception{
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.readImg(image),512,512);
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }

        public float[] whc2cwh(float[] src) {
            float[] chw = new float[src.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < src.length; i += 3) {
                    chw[j] = Float.valueOf(src[i]).floatValue();
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

        // 求最大值




        // 执行推理
        public void run() throws Exception{

            // 只需要做 BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            // 转为浮点
            dst.convertTo(dst, CvType.CV_32FC3);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ 3 * 512 * 512 ];
            dst.get(0, 0, whc);
            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);
            try {


                // 这里，模型输入要求 float16 默认的库没有实现,这里主要将 OnnxTensor 源码
                // 复制出来然后建立相同的包来修改源码实现传入目标类型 OnnxTensorType

                // ---------模型1输入-----------
                // input -> [-1, 3, 512, 512] -> FLOAT
                OnnxTensor tensor = OnnxTensor.createTensor(
                        env1,
                        FloatBuffer.wrap(chw),
                        new long[]{1,3,512,512},
                        // 重载的一个接口来支持 float16
                        TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                );

                // ---------模型1输出-----------
                // output -> [-1, 3, 2048, 2048] -> FLOAT
                OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));
                float[][][] output = ((float[][][][])(res.get(0)).getValue())[0];

                // 获取最大值
                float max = Float.MIN_VALUE;
                for (float[][] subArray : output) {
                    for (float[] subSubArray : subArray) {
                        for (float element : subSubArray) {
                            if (element > max) {
                                max = element;
                            }
                        }
                    }
                }

                // 转为RGB图像
                BufferedImage image = new BufferedImage(2048, 2048, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < 2048; y++) {
                    for (int x = 0; x < 2048; x++) {

                        float r = output[0][y][x]   * 150f;
                        float g = output[1][y][x]  * 150f;
                        float b = output[2][y][x]  * 150f;

                    }
                }
            this.out = image;
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
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

            // 弹窗显示
            JFrame frame = new JFrame("img");

            JPanel all = new JPanel();

            // 输入图片 512*512
            all.add(new JLabel(new ImageIcon(mat2BufferedImage(dst))));
            // 为方便显示也转为 512*512
            all.add(new JLabel(new ImageIcon(resize(out,512,512))));

            frame.getContentPane().add(all);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.pack();

        }
    }

    public static void main(String[] args) throws Exception{

        // ---------模型1输入-----------
        // input -> [-1, 3, 512, 512] -> FLOAT
        // ---------模型1输出-----------
        // output -> [-1, 3, 2048, 2048] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\real_esrgan\\RealESRGAN_ANIME_6B_512x512.onnx");

        // 输入图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\real_esrgan\\img.png";
        ImageObj imageObj = new ImageObj(pic);

        // 显示
        imageObj.show();


    }
}
