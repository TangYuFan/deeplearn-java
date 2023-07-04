package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
*   @desc : 视频人像抠图
 *
 *
*   @auth : tyf
*   @date : 2022-05-09  17:26:43
*/
public class robust_video_matting_video {


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
        Mat src_1;
        Mat src_2;
        Mat src_3;
        Mat src_4;
        Mat src_5;
        Mat src_6;
        Mat src_7;
        Mat src_8;
        Mat src_9;
        Mat src_10;

        // 保存处理得到的前景
        Mat dst_1;
        Mat dst_2;
        Mat dst_3;
        Mat dst_4;
        Mat dst_5;
        Mat dst_6;
        Mat dst_7;
        Mat dst_8;
        Mat dst_9;
        Mat dst_10;

        // 帧原始宽高
        int w;
        int h;

        // 输入张量
        OnnxTensor tensor_src;
        OnnxTensor tensor_r1i;
        OnnxTensor tensor_r2i;
        OnnxTensor tensor_r3i;
        OnnxTensor tensor_r4i;
        OnnxTensor tensor_downsample_ratio;

        // 10个图片一起输入
        public ImageObj(Mat m1,Mat m2,Mat m3,Mat m4,Mat m5,Mat m6,Mat m7,Mat m8,Mat m9,Mat m10) {

            this.src_1 = m1;
            this.src_2 = m2;
            this.src_3 = m3;
            this.src_4 = m4;
            this.src_5 = m5;
            this.src_6 = m6;
            this.src_7 = m7;
            this.src_8 = m8;
            this.src_9 = m9;
            this.src_10 = m10;

            this.tensor_src = this.transferTensor_src(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10);// 张量1 10个图像
            this.tensor_downsample_ratio = this.transferTensor_downsample_ratio();// 张量2 缩放比例
            this.tensor_r1i = this.transferTensor_r1i();
            this.tensor_r2i = this.transferTensor_r2i();
            this.tensor_r3i = this.transferTensor_r3i();
            this.tensor_r4i = this.transferTensor_r4i();

            this.run(); // 执行推理
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public static float[] whc2chw(float[] src) {
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

        // 将10个图片转为 tensor
        public OnnxTensor transferTensor_src(Mat m1,Mat m2,Mat m3,Mat m4,Mat m5,Mat m6,Mat m7,Mat m8,Mat m9,Mat m10){

            int h = m1.rows(); // 假设所有图像具有相同的高度
            int w = m1.cols(); // 假设所有图像具有相同的宽度

            // 初始化一个输入数组 channels * netWidth * netHeight * batchSize
            float[] whc = new float[3 * h * w * 10];
            int index = 0;

            // 处理每个图像
            for (Mat m : Arrays.asList(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10)) {
                // BGR -> RGB
                Imgproc.cvtColor(m, m, Imgproc.COLOR_BGR2RGB);

                // 归一化 0-255 转 0-1
                m.convertTo(m, CvType.CV_32FC1, 1. / 255);

                // 将图像数据存储到 whc 数组中
                float[] imageData = new float[3 * h * w];
                m.get(0, 0, imageData);
                for (float value : imageData) {
                    whc[index] = value;
                    index++;
                }
            }

            // 将 whc 数组转换为 CHW 格式的数组
            float[] chw = whc2chw(whc);

            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{10, 3, h, w});
            } catch (Exception e) {
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
            float[] r1iData = { 0,0,0,0,0,0,0,0,0,0 };
            OnnxTensor r1iTensor = null;
            try {
                r1iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r1iData), new long[]{10, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r1iTensor;
        }

        public OnnxTensor transferTensor_r2i(){
            // 根据您的需求设置 r1i 的值
            float[] r2iData = { 0,0,0,0,0,0,0,0,0,0 };
            OnnxTensor r2iTensor = null;
            try {
                r2iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r2iData), new long[]{10, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r2iTensor;
        }
        public OnnxTensor transferTensor_r3i(){
            // 根据您的需求设置 r1i 的值
            float[] r3iData = { 0,0,0,0,0,0,0,0,0,0 };
            OnnxTensor r3iTensor = null;
            try {
                r3iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r3iData), new long[]{10, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r3iTensor;
        }
        public OnnxTensor transferTensor_r4i(){
            // 根据您的需求设置 r1i 的值
            float[] r4iData = { 0,0,0,0,0,0,0,0,0,0 };
            OnnxTensor r4iTensor = null;
            try {
                r4iTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(r4iData), new long[]{10, 1,1,1});
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(0);
            }
            return r4iTensor;
        }
        // 执行推理
        public void run(){
            try {

                Map<String,OnnxTensor> in = new HashMap<>();
                in.put("src",this.tensor_src);
                in.put("downsample_ratio",this.tensor_downsample_ratio);
                in.put("r1i",this.tensor_r1i);
                in.put("r2i",this.tensor_r2i);
                in.put("r3i",this.tensor_r3i);
                in.put("r4i",this.tensor_r4i);

                OrtSession.Result res = session.run(in);

                // 输入10个图片输出就是10个图片

                // 前景 10 * 3 * w * h
                float[][][][] fgr = ((float[][][][])(res.get(0)).getValue());
                // 透明度 10 * 1 * w * h
                float[][][][] pha = ((float[][][][])(res.get(1)).getValue());

                this.dst_1 = parseMat(fgr[0],pha[0]);
                this.dst_2 = parseMat(fgr[1],pha[1]);
                this.dst_3 = parseMat(fgr[2],pha[2]);
                this.dst_4 = parseMat(fgr[3],pha[3]);
                this.dst_5 = parseMat(fgr[4],pha[4]);
                this.dst_6 = parseMat(fgr[5],pha[5]);
                this.dst_7 = parseMat(fgr[6],pha[6]);
                this.dst_8 = parseMat(fgr[7],pha[7]);
                this.dst_9 = parseMat(fgr[8],pha[8]);
                this.dst_10 = parseMat(fgr[9],pha[9]);

            }
            catch (Exception e){
                e.printStackTrace();
            }
        }

        // 处理单个mat的输出,需要处理10个
        public Mat parseMat(float[][][] fgr,float[][][] pha){
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
            return image;
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

    public static void showMat(Mat src){
        // 弹窗显示
        BufferedImage imageDst = mat2BufferedImage(src);
        JFrame frame = new JFrame("Image");
        frame.setSize(imageDst.getWidth(), imageDst.getHeight());
        JLabel label = new JLabel(new ImageIcon(imageDst));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
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

        // 输入vide,因为模型是可以处理上下文的,每10帧进行一次推理
        String video = new File("").getCanonicalPath() + "\\model\\deeplearning\\mod_net2\\people.mp4";
        VideoCapture cap = new VideoCapture(video);

        // 获取视频的帧率、宽度和高度等信息
        double fps = cap.get(Videoio.CAP_PROP_FPS);
        int width = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        int numFrames = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);


        System.out.println("fps :"+fps);
        System.out.println("width :"+width);
        System.out.println("height :"+height);
        System.out.println("numFrames :"+numFrames);

        // 总帧数%10
        int frames = numFrames / 10 * 10;
        int epoch = frames / 10 -1 ;
        System.out.println("待处理总帧数:"+frames+",每次读取10帧总读取次数:"+epoch);

        // 每次读取10帧
        Mat m1 = new Mat(height, width, CvType.CV_8UC3);
        Mat m2 = new Mat(height, width, CvType.CV_8UC3);
        Mat m3 = new Mat(height, width, CvType.CV_8UC3);
        Mat m4 = new Mat(height, width, CvType.CV_8UC3);
        Mat m5 = new Mat(height, width, CvType.CV_8UC3);
        Mat m6 = new Mat(height, width, CvType.CV_8UC3);
        Mat m7 = new Mat(height, width, CvType.CV_8UC3);
        Mat m8 = new Mat(height, width, CvType.CV_8UC3);
        Mat m9 = new Mat(height, width, CvType.CV_8UC3);
        Mat m10 = new Mat(height, width, CvType.CV_8UC3);


        // 顺序保存所有处理过后的mat 写到新视频中
        String filename = "output.avi";
        VideoWriter videoWriter = new VideoWriter(filename, VideoWriter.fourcc('M', 'J', 'P', 'G'), fps, new Size(width, height));

        // 每次读取10帧进行处理
        for(int i=0;i<=epoch;i++){

            cap.read(m1);
            cap.read(m2);
            cap.read(m3);
            cap.read(m4);
            cap.read(m5);
            cap.read(m6);
            cap.read(m7);
            cap.read(m8);
            cap.read(m9);
            cap.read(m10);

            // 处理推理
            ImageObj imageObj = new ImageObj(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10);

            imageObj.dst_1.convertTo(imageObj.dst_1, CvType.CV_8UC3);
            imageObj.dst_2.convertTo(imageObj.dst_2, CvType.CV_8UC3);
            imageObj.dst_3.convertTo(imageObj.dst_3, CvType.CV_8UC3);
            imageObj.dst_4.convertTo(imageObj.dst_4, CvType.CV_8UC3);
            imageObj.dst_5.convertTo(imageObj.dst_5, CvType.CV_8UC3);
            imageObj.dst_6.convertTo(imageObj.dst_6, CvType.CV_8UC3);
            imageObj.dst_7.convertTo(imageObj.dst_7, CvType.CV_8UC3);
            imageObj.dst_8.convertTo(imageObj.dst_8, CvType.CV_8UC3);
            imageObj.dst_9.convertTo(imageObj.dst_9, CvType.CV_8UC3);
            imageObj.dst_10.convertTo(imageObj.dst_10, CvType.CV_8UC3);

            // 得到处理后的10个帧
            videoWriter.write(imageObj.dst_1);
            videoWriter.write(imageObj.dst_2);
            videoWriter.write(imageObj.dst_3);
            videoWriter.write(imageObj.dst_4);
            videoWriter.write(imageObj.dst_5);
            videoWriter.write(imageObj.dst_6);
            videoWriter.write(imageObj.dst_7);
            videoWriter.write(imageObj.dst_8);
            videoWriter.write(imageObj.dst_9);
            videoWriter.write(imageObj.dst_10);

            System.out.println("总次数:"+epoch+",当前完成到:"+(i+1));

        }

        // 释放资源
        videoWriter.release();
        cap.release();

        System.out.println("完成视频生成:"+filename);

    }
}
