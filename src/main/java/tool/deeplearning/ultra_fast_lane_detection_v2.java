package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
*   @desc : Ultra-Fast-Lane-Detection-v2 车道线检测
 *              Ultra-Fast-Lane-Detection-v2是TPAMI2022期刊里的论文，它是速度精度双SOTA的最新车道线检测算法。
*   @auth : tyf
*   @date : 2022-05-06  12:09:31
*/
public class ultra_fast_lane_detection_v2 {

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
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,1600,320);
            this.tensor = this.transferTensor(this.dst.clone(),3,1600,320); // 转张量
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

                //     loc_row -> [1, 200, 72, 4] -> FLOAT 垂直方向车道线信息
                float[][][][] loc_row = ((float[][][][])(res.get(0)).getValue());
                //     exist_row -> [1, 2, 72, 4] -> FLOAT 垂直方向车道线存在情况
                float[][][][] exist_row = ((float[][][][])(res.get(1)).getValue());
                //     loc_col -> [1, 100, 81, 4] -> FLOAT 水平方向车道线信息
                float[][][][] loc_col = ((float[][][][])(res.get(2)).getValue());
                //     exist_col -> [1, 2, 81, 4] -> FLOAT 水平方向车道线存在情况
                float[][][][] exist_col = ((float[][][][])(res.get(3)).getValue());

                this.parse(loc_row,exist_row,loc_col,exist_col,640,640);


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

        // 解析输出
        public void parse(float[][][][] loc_row,float[][][][] exist_row,float[][][][] loc_col,float[][][][] exist_col,int img_w,int img_h){

            int batch_size = loc_row.length;

            int num_grid_row = loc_row[0].length;
            int num_cls_row = loc_row[0][0].length;
            int num_lane_row = loc_row[0][0][0].length;

            int num_grid_col = loc_col[0].length;
            int num_cls_col = loc_col[0][0].length;
            int num_lane_col = loc_col[0][0][0].length;



        }


        // 弹窗显示
        public void show(){


        }
    }

    public static void main(String[] args) throws Exception {


        // https://github.com/hpc203/u2net-onnxruntime


        /*
            ---------模型输入-----------
            input -> [1, 3, 320, 1600] -> FLOAT
            ---------模型输出-----------
            loc_row -> [1, 200, 72, 4] -> FLOAT
            loc_col -> [1, 100, 81, 4] -> FLOAT
            exist_row -> [1, 2, 72, 4] -> FLOAT
            exist_col -> [1, 2, 81, 4] -> FLOAT
         */
        init(new File("").getCanonicalPath() + "\\model\\deeplearning\\ultra_fast_lane_detection_v2\\ufldv2_culane_res34_320x1600.onnx");


        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\ultra_fast_lane_detection_v2\\car.jpeg");

        image.show();


    }




}
