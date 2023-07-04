package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : yolov7人头检测（人头密度检测）：
 *
*   @auth : tyf
*   @date : 2022-05-11  17:21:31
*/
public class yolov7_head_detection {

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

        // 保存目标边框
        ArrayList<long[]> box = new ArrayList<>();

        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);

        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src.clone(),640,480);
            this.tensor = this.transferTensor(this.dst.clone(),3,640,480); // 转张量
            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }

        public float[] whc2chw(float[] src) {
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
            float[] whc = new float[ channels * netWidth * netHeight ];
            dst.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2chw(whc);

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


        // 执行推理
        public void run(){
            try {

                // ---------模型输入-----------
                // input -> [1, 3, 480, 640] -> FLOAT
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

                // ---------模型输出-----------
                // score -> [-1, 1] -> FLOAT 检测到的目标的分数
                float[][] score = ((float[][])(res.get(0)).getValue());
                // batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64] 检测到的目标
                long[][] batchno_classid_y1x1y2x2 = ((long[][])(res.get(1)).getValue());

                // 处理目标框(模型做了合并nms处理,所以只需要按照阈值过滤即可)
                int num = score.length;// 检测到的目标个数
                System.out.println("检测到目标个数:"+num);
                for(int i=0;i<num;i++){
                    float s = score[i][0];//分数
                    box.add(batchno_classid_y1x1y2x2[i]);
                }

            }
            catch (Exception e){
                e.printStackTrace();
            }
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

            // 画边框
            box.stream().forEach(n->{

                // batchno_classid_y1x1y2x2
                float batchno = n[0];
                float classid = n[1];
                float y1 = n[2];
                float x1 = n[3];
                float y2 = n[4];
                float x2 = n[5];

                // 画边框
                Imgproc.rectangle(
                        dst,
                        new Point(Float.valueOf(x1).intValue(), Float.valueOf(y1).intValue()),
                        new Point(Float.valueOf(x2).intValue(), Float.valueOf(y2).intValue()),
                        color1,
                        2);

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
        // input -> [1, 3, 480, 640] -> FLOAT
        // ---------模型输出-----------
        // score -> [-1, 1] -> FLOAT
        // batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64
        init(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov7_head_detection\\yolov7_tiny_head_0.768_post_480x640.onnx");
//                "\\model\\deeplearning\\yolov7_head_detection\\yolov7_head_0.752_post_480x640.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov7_head_detection\\people.png");

        // 显示
        image.show();

    }


}
