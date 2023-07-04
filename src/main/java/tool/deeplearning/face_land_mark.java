package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
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
import java.util.List;

/**
*   @desc : 人脸关键点检测，也就是人脸对齐
*   @auth : tyf
*   @date : 2022-04-28  16:31:32
*/
public class face_land_mark {

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

    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
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
    public static Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(netWidth, netHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }

    public static OnnxTensor transferTensor(Mat dst, int channels, int netWidth, int netHeight){
        // BGR -> RGB
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

        //  归一化 0-255 转 0-1
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

        // 减去均值，除以标准差
        Core.subtract(dst, new Scalar(0.485, 0.456, 0.406), dst);
        Core.divide(dst, new Scalar(0.229, 0.224, 0.225), dst);

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


    // Mat 转 BufferedImage
    public static BufferedImage mat2BufferedImage(Mat mat){
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


    public static float stc_x(float scala,float x){
        return scala*x;
    }

    public static float stc_y(float scala,float y){
        return scala*y;
    }


    public static void main(String[] args) throws Exception{

        // https://github.com/cunjian/pytorch_face_landmark


        /*
        ---------模型输入-----------
        input -> [-1, 3, 56, 56] -> FLOAT
        ---------模型输出-----------
        output -> [-1, 136] -> FLOAT
         */
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\face_land_mark\\landmark_detection_56_se_external.onnx");

        // 原始图片
        Mat src = readImg(new File("").getCanonicalPath()+"\\model\\deeplearning\\face_land_mark\\face.jpg");

        // 缩放 56*56
        Mat dst = resizeWithoutPadding(src.clone(),56,56);

        // 预处理
        OnnxTensor tensor = transferTensor(dst.clone(),3,56,56);

        // 推理
        OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

        // [136] 也就是68点的xy坐标
        float[] data = ((float[][])(res.get(0)).getValue())[0];


        Scalar color = new Scalar(0, 255, 0);

        // 关键点坐标 56*56 内
        List<float[]> points = new ArrayList<>();

        // 输出的 136 是 xyxyxyxyxy 排列的,并且每个元素都小于1,需要乘图片宽高
        for(int i=0;i<data.length-1;i=i+2){
            // 每个元素都小于1,需要乘输入图片宽高,也就是56
            // 另外坐标需要缩放到原始图片中也就是除56,再乘以原始宽高,所以56约掉了,相当于直接乘原始图片宽高即可
            float x = data[i] * src.width();
            float y = data[i+1] * src.height();
            points.add(new float[]{x,y});
        }


        // 在原始图片画点
        points.stream().forEach(n->{
            float x = n[0];
            float y = n[1];
            Imgproc.circle(
                    src,
                    new Point(Float.valueOf(x).intValue(), Float.valueOf(y).intValue()),
                    2, // 半径
                    color,
                    2);
        });


        // 弹窗显示
        JFrame frame = new JFrame("Image");
        frame.setSize(src.width(), src.height());
        JLabel label = new JLabel(new ImageIcon(mat2BufferedImage(src)));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }
}
