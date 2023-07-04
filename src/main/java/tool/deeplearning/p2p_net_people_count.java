package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.core.Point;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : P2PNet 检测人群以及人群计数
 *          腾讯优图实验室在ICCV 2021发布了一篇论文
 *          《Rethinking Counting and Localization in Crowds:A Purely Point-Based Framework》
*   @auth : tyf
*   @date : 2022-05-08  10:20:15
*/
public class p2p_net_people_count {


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
        // 计算的新的宽高
        int new_w;
        int new_h;
        // 原始图片(模型尺寸的)
        Mat dst;
        // 输入张量
        OnnxTensor tensor;
        // 人点坐标
        ArrayList<float[]> points = new ArrayList();
        Scalar color1 = new Scalar(0, 0, 255);
        public ImageObj(String image) {

            // 原始图片
            this.src = this.readImg(image);

            // 计算新的wh,模型输入图片没有大小要求,但是wh必须被128整除
            int w = this.src.cols();
            int h = this.src.rows();
            this.new_w = w / 128 * 128;
            this.new_h = h / 128 * 128;

            System.out.println("new_w:"+new_w+",new_h:"+new_h);

            this.dst = this.resizeWithoutPadding(this.src,this.new_w,this.new_h);
            this.tensor = this.transferTensor(this.dst.clone(),3,this.new_w,this.new_h); // 转张量
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

            // 归一化
            dst.convertTo(dst, CvType.CV_32FC3, 1.0 / 255.0);

            double[] meanValue = {0.485, 0.456, 0.406};
            double[] stdValue = {0.229, 0.224, 0.225};

            Core.subtract(dst, new Scalar(meanValue), dst);
            Core.divide(dst, new Scalar(stdValue), dst);

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

                // ---------模型输出-----------
                // pred_logits -> [-1, -1] -> FLOAT
                // pred_points -> [-1, -1, 2] -> FLOAT

                // 212504   每个人是否存在的置信度估计,
                float[] pred_logits = ((float[][])(res.get(0)).getValue())[0];

                // 212504 * 2   每个人所在点的坐标
                float[][] pred_points = ((float[][][])(res.get(1)).getValue())[0];


                int count = pred_logits.length;

                for(int i=0;i<count;i++){
                    float score = pred_logits[i];
                    if (score > 0.5) {
                        // 计算点
                        float a = pred_points[i][0];
                        float b = pred_points[i][1];

                        // TODO 这里xy坐标转换到原始图片中未完成
                        float x = a * new_w;
                        float y = b * new_h;

                        System.out.println("a:"+a+",b:"+b+",x:"+x+",y:"+y);

                        // 添加点
                         points.add(new float[]{x, y});
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

            // 遍历所有点
            points.stream().forEach(n->{

                float x = n[0] * new_w;
                float y = n[1] * new_h;

                // 点
                Imgproc.circle(
                        dst,
                        new Point(Float.valueOf(x).intValue(), Float.valueOf(y).intValue()),
                        1, // 半径
                        color1,
                        1);
            });

            JFrame frame = new JFrame("Image");
            frame.setSize(dst.width(), dst.height());

            // 图片转为原始大小
            BufferedImage img = mat2BufferedImage(dst);
            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // input -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型输出-----------
        // pred_logits -> [-1, -1] -> FLOAT
        // pred_points -> [-1, -1, 2] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\p2p_net_people_count\\SHTechA.onnx");

        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\p2p_net_people_count\\demo.jpg");

        image.show();

    }


}
