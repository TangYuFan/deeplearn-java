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
*   @desc : YOLOPV2目标检测+可驾驶区域分割+车道线分割
 *          onnx_post文件夹里的onnx文件，是把最后3个yolo层在经过decode之后，经过torch.cat(z, 1)合并成一个张量，并且还包含nms的。
 *          因此在加载onnx_post文件夹里的onnx文件做推理之后的后处理非常简单，只需要过滤置信度低的检测框。
 *
*   @auth : tyf
*   @date : 2022-05-06  16:47:01
*/
public class yolovp2_detection_drive_area_line_736_1280 {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();


        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        // 设置gpu deviceId=0 注释这两行则使用cpu
//        options.addCUDA(0);
//        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

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
        // 原始图片(原始尺寸)
        Mat src;
        // 原始图片(模型尺寸的)
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
        public ImageObj(String image) {
            this.src = this.resizeWithoutPadding(this.readImg(image),1280,736);
            this.dst = src.clone();
            this.tensor = this.transferTensor(this.dst.clone(),3,1280,736); // 转张量
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

                // score -> [-1, 1] -> FLOAT 检测到的目标的分数
                // batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64] 检测到的目标
                // seg -> [1, 2, 736, 1280] -> FLOAT 可行驶区域的mask
                // ll -> [1, 1, 736, 1280] -> FLOAT 车道线的mask

                // score -> [-1, 1] -> FLOAT 检测到的目标的分数
                float[][] score = ((float[][])(res.get(2)).getValue());
                // batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64] 检测到的目标
                long[][] batchno_classid_y1x1y2x2 = ((long[][])(res.get(3)).getValue());

                // seg -> [1, 2, 736, 1280] -> FLOAT 可行驶区域的mask
                float[][][][] seg = ((float[][][][])(res.get(0)).getValue());
                // ll -> [1, 1, 736, 1280] -> FLOAT 车道线的mask
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

                // 处理可行驶区域mask  seg -> [1, 2, 736, 1280] 里面是个2,测试发现两个mask都一样
//                this.area = seg[0][0];
                this.area = seg[0][1];


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


            // 画可行使区域和车道线
            for(int row =0;row<736;row++){
                for(int clo =0;clo<1280;clo++){
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


            JFrame frame = new JFrame("Image");


            // 将两个图片合并到一起显示
            Mat all = new Mat();
            ArrayList<Mat> mats = new ArrayList<>();
            mats.add(src);
            mats.add(dst);
            Core.hconcat(mats, all);

            JLabel label = new JLabel(new ImageIcon(mat2BufferedImage(all)));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        }
    }


    public static void main(String[] args) throws Exception{


        /*
        ---------模型输入-----------
        input -> [1, 3, 736, 1280] -> FLOAT
        ---------模型输出-----------
        seg -> [1, 2, 736, 1280] -> FLOAT
        ll -> [1, 1, 736, 1280] -> FLOAT
        score -> [-1, 1] -> FLOAT
        batchno_classid_y1x1y2x2 -> [-1, 6] -> INT64
         */
        init(new File("").getCanonicalPath() + "\\model\\deeplearning\\yolovp2_detection_drive_area_line\\yolopv2_post_736x1280.onnx");

        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolovp2_detection_drive_area_line\\car.png");

        image.show();


        // CPU:
        // 预处理耗时:30毫秒
        // 推理耗时:588毫秒
        // 后处理耗时:18毫秒


    }

}
