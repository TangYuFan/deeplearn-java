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
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : google 的 movenet 人体17关键点检测
 *
 *
*   @auth : tyf
*   @date : 2022-05-09  14:10:28
*/
public class google_move_net_people_key_point {


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
        // 图片原始宽高
        int w;
        int h;
        // 输入张量
        OnnxTensor tensor;
        // 保存点
        ArrayList<float[]> points = new ArrayList<>();
        Scalar color = new Scalar(0, 255, 0);
        Scalar color2 = new Scalar(0, 0, 255);
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.w = this.src.width();
            this.h = this.src.height();
            this.dst = this.resizeWithoutPadding(this.src,192,192);
            this.tensor = this.transferTensor(this.dst.clone(),3,192,192); // 转张量
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }

        public int[] whc2cwh(int[] src) {
            int[] chw = new int[src.length];
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

            // 模型输入是int形 CV_32SC1 整形也就没有归一化了
            dst.convertTo(dst, CvType.CV_32SC3);

            // 模型输入的是 [1, 192, 192, 3] 注意通道在最后也就是不需要 whc2cwh 了
            int[] whc = new int[ channels * netWidth * netHeight ];

            dst.get(0, 0, whc);

            OnnxTensor tensor = null;
            try {
                // 模型输入的是 [1, 192, 192, 3] 注意通道在最后
                tensor = OnnxTensor.createTensor(env, IntBuffer.wrap(whc), new long[]{1,netWidth,netHeight,channels});
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

                // [1, 1, 17, 3]  17个点xy置信度

                // 17, 3
                float[][] data = ((float[][][][])(res.get(0)).getValue())[0][0];

                // 遍历17个点
               for(int i=0;i<data.length;i++){
                    float y = data[i][0];
                    float x = data[i][1];
                    float score = data[i][2];

                    // 模型输出的是 0~1 直接按照宽高比例缩放会原始宽高即可
                   points.add(new float[]{
                           x,
                           y,
                           score
                   });
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

            // 17个点
            float[] p1 = points.get(0);
            float[] p2 = points.get(1);
            float[] p3 = points.get(2);
            float[] p4 = points.get(3);
            float[] p5 = points.get(4);
            float[] p6 = points.get(5);
            float[] p7 = points.get(6);
            float[] p8 = points.get(7);
            float[] p9 = points.get(8);
            float[] p10 = points.get(9);
            float[] p11 = points.get(10);
            float[] p12 = points.get(11);
            float[] p13 = points.get(12);
            float[] p14 = points.get(13);
            float[] p15 = points.get(14);
            float[] p16 = points.get(15);
            float[] p17 = points.get(16);

            int x1 = Float.valueOf(p1[0] * w).intValue();
            int y1 = Float.valueOf(p1[1] * h).intValue();

            int x2 = Float.valueOf(p2[0] * w).intValue();
            int y2 = Float.valueOf(p2[1] * h).intValue();

            int x3 = Float.valueOf(p3[0] * w).intValue();
            int y3 = Float.valueOf(p3[1] * h).intValue();

            int x4 = Float.valueOf(p4[0] * w).intValue();
            int y4 = Float.valueOf(p4[1] * h).intValue();

            int x5 = Float.valueOf(p5[0] * w).intValue();
            int y5 = Float.valueOf(p5[1] * h).intValue();

            int x6 = Float.valueOf(p6[0] * w).intValue();
            int y6 = Float.valueOf(p6[1] * h).intValue();

            int x7 = Float.valueOf(p7[0] * w).intValue();
            int y7 = Float.valueOf(p7[1] * h).intValue();

            int x8 = Float.valueOf(p8[0] * w).intValue();
            int y8 = Float.valueOf(p8[1] * h).intValue();

            int x9 = Float.valueOf(p9[0] * w).intValue();
            int y9 = Float.valueOf(p9[1] * h).intValue();

            int x10 = Float.valueOf(p10[0] * w).intValue();
            int y10 = Float.valueOf(p10[1] * h).intValue();

            int x11 = Float.valueOf(p11[0] * w).intValue();
            int y11 = Float.valueOf(p11[1] * h).intValue();

            int x12 = Float.valueOf(p12[0] * w).intValue();
            int y12 = Float.valueOf(p12[1] * h).intValue();

            int x13 = Float.valueOf(p13[0] * w).intValue();
            int y13 = Float.valueOf(p13[1] * h).intValue();

            int x14 = Float.valueOf(p14[0] * w).intValue();
            int y14 = Float.valueOf(p14[1] * h).intValue();

            int x15 = Float.valueOf(p15[0] * w).intValue();
            int y15 = Float.valueOf(p15[1] * h).intValue();

            int x16 = Float.valueOf(p16[0] * w).intValue();
            int y16 = Float.valueOf(p16[1] * h).intValue();

            int x17 = Float.valueOf(p17[0] * w).intValue();
            int y17 = Float.valueOf(p17[1] * h).intValue();


            // 标注原图,以镜像方式看左右
            Imgproc.circle(src, new Point(x1,y1), 2, color, 2); // 鼻子
            Imgproc.circle(src, new Point(x2,y2), 2, color, 2); // 右眼
            Imgproc.circle(src, new Point(x3,y3), 2, color, 2); // 左眼
            Imgproc.circle(src, new Point(x4,y4), 2, color, 2); // 右耳
            Imgproc.circle(src, new Point(x5,y5), 2, color, 2); // 左耳
            Imgproc.circle(src, new Point(x6,y6), 2, color, 2); // 右肩
            Imgproc.circle(src, new Point(x7,y7), 2, color, 2); // 左肩
            Imgproc.circle(src, new Point(x8,y8), 2, color, 2); // 右肘
            Imgproc.circle(src, new Point(x9,y9), 2, color, 2); // 左肘
            Imgproc.circle(src, new Point(x10,y10), 2, color, 2);// 右手腕
            Imgproc.circle(src, new Point(x11,y11), 2, color, 2);// 左手腕
            Imgproc.circle(src, new Point(x12,y12), 2, color, 2);// 右跨
            Imgproc.circle(src, new Point(x13,y13), 2, color, 2);// 左跨
            Imgproc.circle(src, new Point(x14,y14), 2, color, 2);// 右膝盖
            Imgproc.circle(src, new Point(x15,y15), 2, color, 2);// 左膝盖
            Imgproc.circle(src, new Point(x16,y16), 2, color, 2);// 右脚
            Imgproc.circle(src, new Point(x17,y17), 2, color, 2);// 左脚

            // 画线
            Imgproc.line(src, new Point(x12,y12), new Point(x14,y14), color, 2);
            Imgproc.line(src, new Point(x14,y14), new Point(x16,y16), color, 2);
            Imgproc.line(src, new Point(x13,y13), new Point(x15,y15), color, 2);
            Imgproc.line(src, new Point(x15,y15), new Point(x17,y17), color, 2);
            Imgproc.line(src, new Point(x6,y6), new Point(x8,y8), color, 2);
            Imgproc.line(src, new Point(x8,y8), new Point(x10,y10), color, 2);
            Imgproc.line(src, new Point(x7,y7), new Point(x9,y9), color, 2);
            Imgproc.line(src, new Point(x9,y9), new Point(x11,y11), color, 2);
            Imgproc.line(src, new Point(x6,y6), new Point(x7,y7), color, 2);
            Imgproc.line(src, new Point(x12,y12), new Point(x13,y13), color, 2);

            Imgproc.line(src, new Point(x6,y6), new Point(x12,y12), color, 2);
            Imgproc.line(src, new Point(x7,y7), new Point(x13,y13), color, 2);

            JFrame frame = new JFrame("Image");
            frame.setSize(src.width(), src.height());
            BufferedImage img = mat2BufferedImage(src);
            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
    }

    public static void main(String[] args) throws Exception{


        // 注意 singlepose 都是只检测一个人,除非是mutl开头的权重才是检测多人
        // movenet_singlepose_lightning_4 轻量级 9m
        // movenet_singlepose_thunder_4 重量级 24m

        // ---------模型输入-----------
        //  input -> [1, 192, 192, 3] -> INT32
        // ---------模型输出-----------
        // output_0 -> [1, 1, 17, 3] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\google_move_net_people_key_point\\movenet_singlepose_lightning_4.onnx");

        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\google_move_net_people_key_point\\people.png");

        image.show();

    }

}
