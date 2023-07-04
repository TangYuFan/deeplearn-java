package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : PP-HumanSeg肖像分割
 *
 *
*   @auth : tyf
*   @date : 2022-05-18  17:56:17
*/
public class pp_human_seg {

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
        BufferedImage src_img;
        // 原始图片(模型尺寸的)
        Mat dst;
        BufferedImage dst_img;
        // 输入张量
        OnnxTensor tensor;
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,192,192);
            this.src_img = matToBufferedImage(dst.clone());
            this.tensor = this.transferTensor(this.dst.clone(),3,192,192); // 转张量
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
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

            // 只需要做 BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            //  归一化 0-255 转 0-1
            dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

            // 减去均值再除以均方差
            double[] meanValue = {0.5, 0.5, 0.5};
            double[] stdValue = {0.5, 0.5, 0.5};
            Core.subtract(dst, new Scalar(meanValue), dst);
            Core.divide(dst, new Scalar(stdValue), dst);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ channels * netWidth * netHeight ];
            dst.get(0, 0, whc);
            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);
            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
            }
            catch (Exception e){
                e.printStackTrace();
            }
            return tensor;
        }
        // 执行推理
        public void run(){
            try {

                // ---------模型输出-----------
                // tf.identity -> [1, 192, 192, 2] -> FLOAT
                OrtSession.Result res = session.run(Collections.singletonMap("x", tensor));

                float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];


                // 创建两个图片,后面的2代表(前景/背景)
                BufferedImage img = new BufferedImage(192, 192, BufferedImage.TYPE_INT_RGB);

                for(int y=0;y<192;y++){
                    for(int x=0;x<192;x++){
                        float d1 = data[y][x][0];
                        float d2 = data[y][x][1];
                        // 人像
                        if(d2>0.8){

                            // 获取原始图像的颜色
                            img.setRGB(x, y,src_img.getRGB(x,y));
                        }else{
                            img.setRGB(x, y, Color.WHITE.getRGB());
                        }
                    }
                }

                // 保存去掉背景的图片
                this.dst_img = img;

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
            ImageIcon icon = new ImageIcon(this.src_img);
            JLabel le1 = new JLabel(icon);
            ImageIcon ico2 = new ImageIcon(this.dst_img);
            JLabel le2 = new JLabel(ico2);

            content.add(le1);
            content.add(le2);

            JFrame frame = new JFrame();
            frame.add(content);
            frame.pack();
            frame.setVisible(true);

        }
    }


    public static void main(String[] args) throws Exception {


        // ---------模型输入-----------
        // x -> [1, 3, 192, 192] -> FLOAT
        // ---------模型输出-----------
        // tf.identity -> [1, 192, 192, 2] -> FLOAT
        init(new File("").getCanonicalPath() +
                "\\model\\deeplearning\\pp_human_seg\\model_float32.onnx");


        // 图片
        String pic = new File("").getCanonicalPath() +
                "\\model\\deeplearning\\pp_human_seg\\pic3.png";
        ImageObj imageObj = new ImageObj(pic);


        // 显示
        imageObj.show();

    }


}
