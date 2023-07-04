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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
*   @desc : 视频人像抠图
 *
 *          GitHub：https://github.com/PeterL1n/RobustVideoMatting/blob/master/README_zh_Hans.md
 *          不同于现有神经网络将每一帧作为单独图片处理，RVM 使用循环神经网络，在处理视频流时有时间记忆。
 *          RVM 可在任意视频上做实时高清抠像。在 Nvidia GTX 1080Ti 上实现 4K 76FPS 和 HD 104FPS。
 *          此研究项目来自字节跳动。相比其他基于像素点的分割,这个模型输入输出可以保持视频原始尺寸
 *
 *          C++/Python 推理参考：
 *          https://github.com/hpc203/robustvideomatting-onnxruntime
 *
 *          onnx 模型下载：
 *          rvm_mobilenetv3_fp32.onnx 102.5m  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
 *          rvm_mobilenetv3_fp16.onnx 51.3m  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx
 *          rvm_resnet50_fp32.onnx 14.3m https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
 *          rvm_resnet50_fp16.onnx 7.2m https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp16.onnx
 *
 *
*   @auth : tyf
*   @date : 2022-05-09  17:26:43
*/
public class robust_video_matting {


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

        // 保存处理得到的前景
        Mat dst;

        // 输入张量
        OnnxTensor tensor_src;
        OnnxTensor tensor_r1i;
        OnnxTensor tensor_r2i;
        OnnxTensor tensor_r3i;
        OnnxTensor tensor_r4i;
        OnnxTensor tensor_downsample_ratio;

        public ImageObj(String image) {

            // 读取图片
            this.src = this.readImg(image);

            // 输入图片宽高
            int w = this.src.width();
            int h = this.src.height();

            // 输入张量1 src -> [-1, 3, -1, -1] -> FLOAT 传入原始图像
            this.tensor_src = this.transferTensor_src(this.src.clone(),3,w,h);
            // 输入张量2 downsample_ratio -> [1] -> FLOAT 缩放比例 0.25
            this.tensor_downsample_ratio = this.transferTensor_downsample_ratio();
            // 输入张量3  r1i -> [-1, -1, -1, -1] -> FLOAT
            this.tensor_r1i = this.transferTensor_r1i();
            // 输入张量4  r2i -> [-1, -1, -1, -1] -> FLOAT
            this.tensor_r2i = this.transferTensor_r2i();
            // 输入张量5  r3i -> [-1, -1, -1, -1] -> FLOAT
            this.tensor_r3i = this.transferTensor_r3i();
            // 输入张量6  r4i -> [-1, -1, -1, -1] -> FLOAT
            this.tensor_r4i = this.transferTensor_r4i();

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
        public OnnxTensor transferTensor_src(Mat dst, int channels, int netWidth, int netHeight){

            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

            //  归一化 0-255 转 0-1
            dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);

            // float16 数据类型是半精度浮点数，使用 16 位来表示浮点数，范围和精度相对较小。
            // float32 数据类型是单精度浮点数，使用 32 位来表示浮点数，范围和精度较大。
            // 在 Java 中，可以使用 short 来表示 float16 数据类型，使用 float 来表示 float32 数据类型。
            // 需要注意的是，float16 类型的数据需要使用 HalfFloatUtils 类进行转换和处理。

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

        public OnnxTensor transferTensor_downsample_ratio(){
            // 创建 downsample_ratio 张量
            float downsample_ratio = 0.25f; // 根据您的需求设置 downsample_ratio 的值
            // 创建包含 downsample_ratio 值的张量
            OnnxTensor downsampleRatioTensor = null;
            try {
                float[] downsampleRatioData = { downsample_ratio };
                downsampleRatioTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(downsampleRatioData), new long[]{1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return downsampleRatioTensor;
        }

        public OnnxTensor transferTensor_r1i(){
            // 根据您的需求设置 r1i 的值
            float[] r1iData = { 0 };
            OnnxTensor r1iTensor = null;
            try {
                r1iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r1iData), new long[]{1, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r1iTensor;
        }

        public OnnxTensor transferTensor_r2i(){
            // 根据您的需求设置 r1i 的值
            float[] r2iData = { 0 };
            OnnxTensor r2iTensor = null;
            try {
                r2iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r2iData), new long[]{1, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r2iTensor;
        }
        public OnnxTensor transferTensor_r3i(){
            // 根据您的需求设置 r1i 的值
            float[] r3iData = { 0 };
            OnnxTensor r3iTensor = null;
            try {
                r3iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r3iData), new long[]{1, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r3iTensor;
        }
        public OnnxTensor transferTensor_r4i(){
            // 根据您的需求设置 r1i 的值
            float[] r4iData = { 0 };
            OnnxTensor r4iTensor = null;
            try {
                r4iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r4iData), new long[]{1, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r4iTensor;
        }
        // 执行推理
        public void run(){
            try {

                // ---------模型输入-----------
                // src -> [-1, 3, -1, -1] -> FLOAT
                // r1i -> [-1, -1, -1, -1] -> FLOAT
                // r2i -> [-1, -1, -1, -1] -> FLOAT
                // r3i -> [-1, -1, -1, -1] -> FLOAT
                // r4i -> [-1, -1, -1, -1] -> FLOAT
                // downsample_ratio -> [1] -> FLOAT
                Map<String,OnnxTensor> in = new HashMap<>();
                in.put("src",this.tensor_src);
                in.put("downsample_ratio",this.tensor_downsample_ratio);
                in.put("r1i",this.tensor_r1i);
                in.put("r2i",this.tensor_r2i);
                in.put("r3i",this.tensor_r3i);
                in.put("r4i",this.tensor_r4i);

                // ---------模型输出-----------
                // fgr -> [-1, 3, -1, -1] -> FLOAT  前景
                // pha -> [-1, 1, -1, -1] -> FLOAT  透明度
                // r1o -> [-1, 16, -1, -1] -> FLOAT 第一层残差块的输出
                // r2o -> [-1, 32, -1, -1] -> FLOAT 第二层残差块的输出
                // r3o -> [-1, 64, -1, -1] -> FLOAT 第三层残差块的输出
                // r4o -> [-1, 128, -1, -1] -> FLOAT 第四层残差块的输出
                OrtSession.Result res = session.run(in);

                // 前景 3 * w * h
                float[][][] fgr = ((float[][][][])(res.get(0)).getValue())[0];
                // 透明度 1 * w * h
                float[][][] pha = ((float[][][][])(res.get(1)).getValue())[0];

                // fgr 就是扣出的图像了
                // 假设 fgr 的形状为 [channels, height, width]
                int channels = fgr.length; // RGB
                int height = fgr[0].length; // height
                int width = fgr[0][0].length; // width

                // 创建一个与 fgr 形状相同的 Mat 对象
                Mat image = new Mat(height, width, CvType.CV_32FC3);

                // 将 fgr 数组的值复制到 Mat 对象，并调整通道顺序为 BGR
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {

                        // 取这个点的透明度
                        float pha_value = pha[0][h][w];

                        // RGB颜色
                        float r = fgr[0][h][w]*255;
                        float g = fgr[1][h][w]*255;
                        float b = fgr[2][h][w]*255;

                        // 前景透明度大于0则设置颜色
                        if(pha_value>0){
                            image.put(h, w, b, g, r);
                        }
                        // 背景设置为白色
                        else{
                            image.put(h, w, 255,255,255);
                        }

                    }
                }

                this.dst = image;

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

            // 原始图片
            Mat m1 = this.src;
            // 扣出的前景图片
            Mat m2 = this.dst;

            // 弹窗显示 原始图片和新的图片
            JPanel content = new JPanel(new GridLayout(1,2,5,5));
            // display the image in a window
            ImageIcon icon = new ImageIcon(mat2BufferedImage(m1));
            JLabel le1 = new JLabel(icon);
            ImageIcon ico2 = new ImageIcon(mat2BufferedImage(m2));
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
        //  src -> [-1, 3, -1, -1] -> FLOAT
        //  r1i -> [-1, -1, -1, -1] -> FLOAT
        //  r2i -> [-1, -1, -1, -1] -> FLOAT
        //  r3i -> [-1, -1, -1, -1] -> FLOAT
        //  r4i -> [-1, -1, -1, -1] -> FLOAT
        //  downsample_ratio -> [1] -> FLOAT
        //  ---------模型输出-----------
        //  fgr -> [-1, 3, -1, -1] -> FLOAT
        //  pha -> [-1, 1, -1, -1] -> FLOAT
        //  r1o -> [-1, 16, -1, -1] -> FLOAT
        //  r2o -> [-1, 32, -1, -1] -> FLOAT
        //  r3o -> [-1, 64, -1, -1] -> FLOAT
        //  r4o -> [-1, 128, -1, -1] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\robust_video_matting\\rvm_resnet50_fp32.onnx");

        // 模型可以处理视频的上下文信息,使用图片的
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\mod_net\\ren.png");

        image.show();

    }
}
