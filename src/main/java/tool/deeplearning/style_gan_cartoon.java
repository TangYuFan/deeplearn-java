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
*   @desc : 使用 style-gan 将人脸头像转为卡通风格
 *          输入图片是人像大头照，如果输入图片里包含太多背景，需要先做人脸检测+人脸矫正。
 *          裁剪出人像区域后，作为本套程序的输入图像，否则效果会大打折扣。
*   @auth : tyf
*   @date : 2022-05-06  09:46:56
*/
public class style_gan_cartoon {


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
        // 输出图片(模型尺寸)
        public BufferedImage out_img;
        // 输入图片(原始尺寸)
        public BufferedImage in_img;
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.in_img = this.mat2BufferedImage(this.src);
            this.dst = this.resizeWithoutPadding(this.src,256,256);
            this.tensor = this.transferTensor(this.dst.clone(),3,256,256); // 转张量
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
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

                // 解析 output_image -> [3, 256, 256] -> FLOAT 注意输出的图片是三通道的

                // 三通道的图像
                float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];

                // 在源程序python中第一步使用实例分割获取了头部的mask,这样在这里就可以将mask之外的区域也就是背景固定为白色或者灰色,这里去掉这一步
                // face = cv2.resize(face, (256, 256))：将输入的头像图像调整为256x256大小。这一步是为了满足模型的输入要求，因为模型期望输入为256x256大小的图像。
                // mask = np.ones((256, 256, 1), dtype=np.float32)：创建一个与头像图像相同大小的全为1的掩码。这个掩码将用于后续对头像图像进行加权处理。
                // face = (face * mask + (1 - mask) * 255) / 127.5 - 1：将头像图像与掩码相乘并加权处理。这一步将使头像图像的非脸部区域变为灰色。具体而言，将头像图像的每个像素乘以掩码的对应像素值，然后加上 (1 - mask) * 255，最后除以 127.5 并减去 1。这个操作将使非脸部区域的像素值在-1到1之间，而脸部区域保持不变。

                // 创建一个空白图像
                BufferedImage img = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < 256; y++) {
                    for (int x = 0; x < 256; x++) {

                        float red = data[0][y][x];
                        float green = data[1][y][x];
                        float blue = data[2][y][x];

                        // gamma值决定了颜色的亮度变化程度。较低的gamma值会使图像变暗，而较高的gamma值会使图像变亮。
                        // 如果希望图像变亮，可以增加gamma值，如1.5或更高。如果希望图像变暗，可以减小gamma值，如0.8或更低。
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

        // https://github.com/hpc203/photo2cartoon-onnxrun-cpp-py


        /*
        ---------模型输入-----------
        input -> [1, 3, 256, 256] -> FLOAT
        ---------模型输出-----------
        output -> [1, 3, 256, 256] -> FLOAT
         */
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\minivision_female_photo2cartoon.onnx");

        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\face.JPG");

        // 显示
        image.show();


    }







}
