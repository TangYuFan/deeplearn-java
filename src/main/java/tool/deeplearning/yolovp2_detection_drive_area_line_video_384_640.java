package tool.deeplearning;


import ai.onnxruntime.*;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;

/**
 *   @desc : YOLOPV2目标检测+可驾驶区域分割+车道线分割
 *          onnx_post文件夹里的onnx文件，是把最后3个yolo层在经过decode之后，经过torch.cat(z, 1)合并成一个张量，并且还包含nms的。
 *          因此在加载onnx_post文件夹里的onnx文件做推理之后的后处理非常简单，只需要过滤置信度低的检测框。
 *
 *   @auth : tyf
 *   @date : 2022-05-06  16:47:01
 */
public class yolovp2_detection_drive_area_line_video_384_640 {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 类别信息
    public static JSONObject clazzs = JSON.parseObject("{\"1\":\"person\",\"2\":\"bicycle\",\"3\":\"car\",\"4\":\"motorcycle\",\"5\":\"airplane\",\"6\":\"bus\",\"7\":\"train\",\"8\":\"truck\",\"9\":\"boat\",\"10\":\"traffic light\",\"11\":\"fire hydrant\",\"12\":\"stop sign\",\"13\":\"parking meter\",\"14\":\"bench\",\"15\":\"bird\",\"16\":\"cat\",\"17\":\"dog\",\"18\":\"horse\",\"19\":\"sheep\",\"20\":\"cow\",\"21\":\"elephant\",\"22\":\"bear\",\"23\":\"zebra\",\"24\":\"giraffe\",\"25\":\"backpack\",\"26\":\"umbrella\",\"27\":\"handbag\",\"28\":\"tie\",\"29\":\"suitcase\",\"30\":\"frisbee\",\"31\":\"skis\",\"32\":\"snowboard\",\"33\":\"sports ball\",\"34\":\"kite\",\"35\":\"baseball bat\",\"36\":\"baseball glove\",\"37\":\"skateboard\",\"38\":\"surfboard\",\"39\":\"tennis racket\",\"40\":\"bottle\",\"41\":\"wine glass\",\"42\":\"cup\",\"43\":\"fork\",\"44\":\"knife\",\"45\":\"spoon\",\"46\":\"bowl\",\"47\":\"banana\",\"48\":\"apple\",\"49\":\"sandwich\",\"50\":\"orange\",\"51\":\"broccoli\",\"52\":\"carrot\",\"53\":\"hot dog\",\"54\":\"pizza\",\"55\":\"donut\",\"56\":\"cake\",\"57\":\"chair\",\"58\":\"couch\",\"59\":\"potted plant\",\"60\":\"bed\",\"61\":\"dining table\",\"62\":\"toilet\",\"63\":\"tv\",\"64\":\"laptop\",\"65\":\"mouse\",\"66\":\"remote\",\"67\":\"keyboard\",\"68\":\"cell phone\",\"69\":\"microwave\",\"70\":\"oven\",\"71\":\"toaster\",\"72\":\"sink\",\"73\":\"refrigerator\",\"74\":\"book\",\"75\":\"clock\",\"76\":\"vase\",\"77\":\"scissors\",\"78\":\"teddy bear\",\"79\":\"hair drier\",\"80\":\"toothbrush\"}");


    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        // 设置gpu deviceId=0 注释这两行则使用cpu
        options.addCUDA(0);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

        session = env.createSession(weight, options);

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
        // 原始图片(模型尺寸的)
        Mat src;
        Mat dst;
        // 输入张量
        OnnxTensor tensor;
        // 保存目标边框
        ArrayList<long[]> box = new ArrayList<>();
        // 保存车道线
        float[][] line;
        // 保存可行使区域
        float[][] area;
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        public ImageObj(Mat mat) {
            this.src = this.resizeWithoutPadding(mat,640,384);
            this.dst = src.clone();
            this.tensor = this.transferTensor(this.dst.clone(),3,640,384); // 转张量
            this.run(); // 执行推理
            this.show(); // 执行标注
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
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);
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


                // score -> [-1, 1] -> FLOAT 检测到的目标的分数
                float[][] score = ((float[][])(res.get(2)).getValue());
                // batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64] 检测到的目标
                long[][] batchno_classid_y1x1y2x2 = ((long[][])(res.get(3)).getValue());

                // seg -> [1, 2, 384, 640] -> FLOAT 可行驶区域的mask
                float[][][][] seg = ((float[][][][])(res.get(0)).getValue());
                // ll -> [1, 1, 384, 640] -> FLOAT 车道线的mask
                float[][][][] ll = ((float[][][][])(res.get(1)).getValue());

                // 处理目标框(模型做了合并nms处理,所以只需要按照阈值过滤即可)
                int num = score.length;// 检测到的目标个数
                for(int i=0;i<num;i++){
                    float s = score[i][0];//分数
                    if(s>0.5){
                        box.add(batchno_classid_y1x1y2x2[i]);
                    }
                }

                // 处理车道线 mask
                this.line = ll[0][0];

                // 处理可行驶区域mask  seg -> [1, 2, 384, 640] 里面是个2,测试发现两个mask都一样
//                this.area = seg[0][0];
                this.area = seg[0][1];


            }
            catch (Exception e){
//                e.printStackTrace();
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


            // 创建3个线程去执行
            CountDownLatch latch = new CountDownLatch(2);

            new Thread(()->{
                // 画边框
                box.stream().forEach(n->{

                    // batchno_classid_y1x1y2x2
                    float batchno = n[0];

                    float classid = n[1];// 类别序号, 在 coco.name 中有匹配
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

                    String c = clazzs.getString(Float.valueOf(classid).intValue()+"");


                    // 类别
                    Imgproc.putText(
                            dst,
                            c ,// 概率取两位小数
                            new Point(x1, y1-3),
                            Imgproc.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color1,
                            2);

                });
                latch.countDown();
            }).start();

            new Thread(()->{
                if(area==null||line==null){
                    latch.countDown();
                    return;
                }
                for(int row =0;row<384;row++){
                    for(int clo =0;clo<640;clo++){
                        // 画可行使区域
                        float area_d = area[row][clo];
                        if(area_d>0.5){
                            // 修改这个点的颜色
                            double[] c = dst.get(row,clo);
                            c[0] = 255;
                            dst.put(row,clo,c);
                        }
                        // 画车道线
                        float line_d = line[row][clo];
                        if(line_d>0.5){
                            // 修改这个点的颜色
                            double[] c = dst.get(row,clo);
                            c[1] = 255;
                            dst.put(row,clo,c);
                        }
                    }
                }
                latch.countDown();
            }).start();

            try {
                latch.await();
            }
            catch (Exception e){
                e.printStackTrace();
            }

        }
    }

    public static ImageObj runRec(Mat src){
        ImageObj image = new ImageObj(src);
        return image;
    }


    public static void main(String[] args) throws Exception{


        /*
        ---------模型输入-----------
        input -> [1, 3, 384, 640] -> FLOAT
        ---------模型输出-----------
        seg -> [1, 2, 384, 640] -> FLOAT
        ll -> [1, 1, 384, 640] -> FLOAT
        score -> [-1, 1] -> FLOAT
        batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64
         */
        init(new File("").getCanonicalPath() + "\\model\\deeplearning\\yolovp2_detection_drive_area_line\\yolopv2_post_384x640.onnx");



        // 视频、rtsp流等
        String video = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolovp2_detection_drive_area_line\\car.mp4";

        // 创建VideoCapture对象并打开视频文件
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


        // 创建一个Mat对象用于存储每一帧
        Mat frame = new Mat(height, width, CvType.CV_8UC3);
        // 读取视频帧,处理完毕之后放入缓冲
        while (cap.read(frame)) {
            // Mat 进行处理
            try {

                // 推理、标注后的图像
                ImageObj res = runRec(frame.clone());

                // 推理后的图像
                Mat m1 = res.dst;

                // 推理前的图像
                Mat m2 = res.src;

                // 水平拼接到一个mat中
                Mat all = new Mat();
                ArrayList<Mat> mats = new ArrayList<>();
                mats.add(m2);
                mats.add(m1);
                Core.hconcat(mats, all);

                // 显示处理后的图像
                org.opencv.highgui.HighGui.imshow("Video", all);

                org.opencv.highgui.HighGui.waitKey(30);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }


        Runtime.getRuntime().addShutdownHook(new Thread(()->{
            // 释放资源
            System.out.println("程序退出!");
            cap.release();
        }));



    }

}
