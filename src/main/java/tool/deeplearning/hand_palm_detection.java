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

/**
*   @desc : 手掌目标检测, 标注首长起区域 + 掌心点 + 掌心上下两个关键点
*   @auth : tyf
*   @date : 2022-05-12  15:54:11
*/
public class hand_palm_detection {


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
        // 保存所有检测出来的手掌
        ArrayList<float[]> palms = new ArrayList<>();
        Scalar red = new Scalar(0, 0, 255);
        Scalar green = new Scalar(0, 255, 0);
        public ImageObj(String image) throws Exception{
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,192,192);
            this.tensor = this.transferTensor(this.dst.clone()); // 转张量
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

        // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,不添加留白
        public static Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(inputWidth, inputHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }

        public OnnxTensor transferTensor(Mat dst) throws Exception{
            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            dst.convertTo(dst, CvType.CV_32FC1);

            // 数组
            float[] whc = new float[ 3 * 192 * 192 ];
            dst.get(0, 0, whc);

            // 将图片维度从 HWC 转换为 CHW
            float[] chw = new float[whc.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < whc.length; i += 3) {
                    chw[j] = whc[i];
                    // 除 255 [0, 1]
                    chw[j] = chw[j] / 255f;
                    j++;
                }
            }
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,3,192,192});
            return tensor;
        }

        // 角度归一化方法
        private static float normalizeRadians(float angle) {
            angle %= 2 * (float) Math.PI;
            if (angle < 0) {
                angle += 2 * (float) Math.PI;
            }
            return angle;
        }


        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

                // pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y -> [-1, 8] -> FLOAT
                float[][] pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y = ((float[][])(res.get(0)).getValue());

                // 检测到手掌个数
                int count = pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y.length;

                for(int i=0;i<count;i++){

                    float[] data = pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y[i];

                    float pd_score = data[0];  // 手掌得分
                    float box_x = data[1]; // 手掌边界框的中心点x
                    float box_y = data[2]; // 手掌边界框的中心点y
                    float box_size = data[3]; // 手掌边界框的大小
                    float kp0_x = data[4]; // 手掌关键点0的 x 坐标
                    float kp0_y = data[5]; // 手掌关键点0的 y 坐标
                    float kp2_x = data[6]; // 手掌关键点2的 x 坐标
                    float kp2_y = data[7]; // 手掌关键点2的 y 坐标



                    if(pd_score>0.6){
                        palms.add(data);
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

            int w = src.width();
            int h = src.height();

            // 遍历每个手掌
            palms.stream().forEach(data->{

                // 所有输出都是 0~1 ,坐标和边框信息需要缩放到原始图片
                float pd_score = data[0];  // 手掌得分
                float box_x = data[1]; // 手掌边界框的中心点x
                float box_y = data[2]; // 手掌边界框的中心点y
                float box_size = data[3]; // 手掌边界框的大小
                float kp0_x = data[4]; // 手掌关键点0的 x 坐标
                float kp0_y = data[5]; // 手掌关键点0的 y 坐标
                float kp2_x = data[6]; // 手掌关键点2的 x 坐标
                float kp2_y = data[7]; // 手掌关键点2的 y 坐标


                // 计算边框

                float half = box_size / 2;
                float x1 = box_x - half;
                float y1 = box_y - half;
                float x2 = box_x + half;
                float y2 = box_y + half;

                x1 = x1 * w;
                x2 = x2 * w;
                y1 = y1 * h;
                y2 = y2 * h;

                // 边框
                Imgproc.rectangle(src, new Point(x1, y1), new Point(x2, y2), red, 2);

                // 两个关键点主要使用计算手掌旋转角度
                box_x = box_x * w;
                box_y = box_y * h;
                kp0_x = kp0_x * w;
                kp0_y = kp0_y * h;
                kp2_x = kp2_x * w;
                kp2_y = kp2_y * h;
                Imgproc.circle(src, new Point(box_x ,box_y), 2, red,2);
                Imgproc.circle(src, new Point(kp0_x ,kp0_y), 2, green,2);
                Imgproc.circle(src, new Point(kp2_x, kp2_y), 2, green,2);

            });

            // 弹窗
            BufferedImage image = mat2BufferedImage(src);
            JFrame frame = new JFrame("Image");
            frame.setSize(image.getWidth(), image.getHeight());
            JLabel label = new JLabel(new ImageIcon(image));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);



        }
    }



    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // input -> [1, 3, 192, 192] -> FLOAT
        // ---------模型输出-----------
        // pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y -> [-1, 8] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\hand_palm_detection\\palm_detection_full_inf_post_192x192.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\hand_3d_landmark\\hand.png");

        // 显示
        image.show();

    }
}
