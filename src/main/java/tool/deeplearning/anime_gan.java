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
*   @desc : 使用ONNXRuntime部署人脸动漫化——AnimeGAN
*   @auth : tyf
*   @date : 2022-04-28  16:26:57
*/
public class anime_gan {


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
        // 输出图片(模型尺寸)
        public BufferedImage out_img;
        // 输入图片(原始尺寸)
        public BufferedImage in_img;
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.in_img = this.mat2BufferedImage(this.src);
            this.dst = this.resizeWithoutPadding(this.src,512,512);
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
            // 只需要做 BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            // 转为浮点
            dst.convertTo(dst, CvType.CV_32FC1);
            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);
            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);
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
        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("input_image", tensor));
                // 解析 output -> [3, 512, 512] -> FLOAT 注意输出的图片是 RGB 的
                float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];
                // create a new BufferedImage
                BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < 512; y++) {
                    for (int x = 0; x < 512; x++) {
                        float red = data[0][y][x];
                        float green = data[1][y][x];
                        float blue = data[2][y][x];

                        // 在RGB颜色空间中，亮度的感知不是线性的，而是对低亮度值更敏感。因此，当你将非线性数据转换为RGB图像时，亮度可能会被过度放大，导致黑色区域显得偏白。
                        // 为了解决这个问题可以尝试使用gamma校正来平衡图像的亮度。gamma校正是一种非线性转换，用于将非线性数据转换为线性数据，从而平衡亮度感知。
                        // 具体来说，在RGB颜色空间中，gamma校正是将每个通道的值提高到幂次方，其中指数称为gamma值。通常，gamma值在2.2左右，这是一个标准值
                        float gamma = 2.2f;
                        red = (float) Math.pow(red, 1/gamma);
                        green = (float) Math.pow(green, 1/gamma);
                        blue = (float) Math.pow(blue, 1/gamma);
                        int r = (int) (red * 255);
                        int g = (int) (green  * 255);
                        int b = (int) (blue  * 255);
                        Color color = new Color(Math.abs(r), Math.abs(g), Math.abs(b));
                        img.setRGB(x, y, color.getRGB());
                    }
                }
                // 缩放到原始尺寸并保存
                this.out_img = resize(img,src.width(),src.height());


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
        // 将 BufferedImage 缩小一半
        public static BufferedImage shrinkByHalf(BufferedImage image) {
            int w = image.getWidth();
            int h = image.getHeight();
            BufferedImage shrunkImage = new BufferedImage(w / 4, h / 4, image.getType());
            Graphics2D g = shrunkImage.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            AffineTransform at = AffineTransform.getScaleInstance(0.25, 0.25);
            AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
            scaleOp.filter(image, shrunkImage);
            g.dispose();
            return shrunkImage;
        }
        // 弹窗显示
        public void show(){

            // 一行两列
            JPanel content = new JPanel(new GridLayout(1,2,5,5));

            // display the image in a window
            ImageIcon icon = new ImageIcon(this.in_img);
            JLabel le1 = new JLabel(icon);

            ImageIcon ico2 = new ImageIcon(this.out_img);
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

        // https://github.com/hpc203/AnimeGAN-onnxruntime

        // ---------模型输入-----------
        // input_image -> [1, 3, 512, 512] -> FLOAT
        //---------模型输出-----------
        // output_image -> [1, 3, 512, 512] -> FLOAT
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\anime_gan\\face_paint_512_v2_0.onnx");


        // 加载图片
//        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\anime_gan\\liushishi.jpg");
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\face.JPG");

        // 显示
        image.show();

    }
}
