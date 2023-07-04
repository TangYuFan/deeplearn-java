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
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : 百度PaddleSeg发布的实时人像抠图模型PP-MattingV2
 *          github以及不同尺寸模型下载: https://github.com/hpc203/PP-MattingV2-onnxrun-cpp-py
*   @auth : tyf
*   @date : 2022-05-09  15:59:43
*/
public class paddlepaddle_mattingv2 {

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
        // 空图片来显示人像,也可以使用一个背景图片进行背景替换
        BufferedImage src_img;
        BufferedImage back_img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,512,512);
            this.src_img = this.matToBufferedImage(this.dst.clone());
            this.tensor = this.transferTensor(this.dst.clone(),3,512,512); // 转张量
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
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
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }
        public BufferedImage matToBufferedImage(Mat matrix) {
            int type = BufferedImage.TYPE_BYTE_GRAY;
            if (matrix.channels() > 1) {
                type = BufferedImage.TYPE_3BYTE_BGR;
            }
            int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
            byte[] buffer = new byte[bufferSize];
            matrix.get(0, 0, buffer);
            BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
            final byte[] targetPixels = ((java.awt.image.DataBufferByte) image.getRaster().getDataBuffer()).getData();
            System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
            return image;
        }
        // 执行推理
        public void run(){
            try {

                OrtSession.Result res = session.run(Collections.singletonMap("img", tensor));

                // sigmoid_5.tmp_0 -> [1, 1, 512, 512] -> FLOAT

                float[][] data = ((float[][][][])(res.get(0)).getValue())[0][0];

                for(int i=0;i<512;i++){
                    for(int j=0;j<512;j++){
                        // 该点是人像取值是0.9xxx 如果不是人像取值是0.000
                        float mask = data[i][j];
                        // 像素点是人,则设置原来的颜色
                        if(mask>0.99){
                            // 从原始图片中获取该点的颜色
                            back_img.setRGB(j, i,src_img.getRGB(j,i));
                        }
                        // 像素点不是人,则设置为白色
                        else{
                            back_img.setRGB(j, i, new Color(255,255,255).getRGB()
                            );
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


            // 弹窗显示 原始图片和新的图片
            // 一行两列
            JPanel content = new JPanel(new GridLayout(1,2,5,5));
            // display the image in a window
            ImageIcon icon = new ImageIcon(src_img);
            JLabel le1 = new JLabel(icon);
            ImageIcon ico2 = new ImageIcon(back_img);
            JLabel le2 = new JLabel(ico2);

            content.add(le1);
            content.add(le2);

            JFrame frame = new JFrame();
            frame.add(content);
            frame.pack();
            frame.setVisible(true);

        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // img -> [1, 3, 512, 512] -> FLOAT
        // ---------模型输出-----------
        // sigmoid_5.tmp_0 -> [1, 1, 512, 512] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\paddlepaddle_mattingv2\\ppmattingv2_stdc1_human_512x512.onnx");

        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\mod_net\\ren.png");

        image.show();

    }

}
