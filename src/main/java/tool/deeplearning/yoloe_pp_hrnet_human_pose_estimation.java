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
import java.util.*;

/**
*   @desc : PP-YOLOE行人检测+HRNet人体骨骼关键点检测,姿态估计
 *
 *          源自百度的飞桨目标检测开发套件 PaddleDetection 中 PP-Human，
 *          它是一个SOTA的产业级开源实时行人分析工具，但是它是使用PaddlePaddle框架做推理引擎的
 *          模型：
 *          dark_hrnet_w32_256x192.onnx     人谷歌关键点检测
 *          mot_ppyoloe_l_36e_pipeline.onnx     行人检测，高精度
 *          mot_ppyoloe_s_36e_pipeline.onnx     行人检测，轻量级
 *
*   @auth : tyf
*   @date : 2022-05-23  16:30:42
*/
public class yoloe_pp_hrnet_human_pose_estimation {

    public static OrtEnvironment env1;
    public static OrtSession session1;

    public static OrtEnvironment env2;
    public static OrtSession session2;


    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }

    // 环境初始化
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }

    public static class ImageObj{
        // 原始图片(原始尺寸)
        Mat src;
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        // 保存所有边框信息
        ArrayList<float[]> datas = new ArrayList<>();
        // 保存所有边框下所有关键点
        ArrayList<ArrayList<float[]>> points = new ArrayList<>();
        public ImageObj(String image) {
            this.src = this.readImg(image);
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

        // 执行推理 目标检测
        public void run1(){

            Mat dst = resizeWithoutPadding(this.src,640,640);
            // 只需要做 BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            // 转为浮点
            dst.convertTo(dst, CvType.CV_32FC1);

            float[] whc = new float[ 3 * 640 *640 ];
            dst.get(0, 0, whc);
            // 得到最终的图片转 float 数组
            float[] chw = whc2chw(whc);

            try {
                // ---------模型2输入-----------
                // image -> [-1, 3, 640, 640] -> FLOAT  图像
                // scale_factor -> [-1, 2] -> FLOAT  缩放因子,宽高,因为这里在输入模型前已经转为640了直接设置为1即可
                OnnxTensor tensor1 = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,3,640,640});
                OnnxTensor tensor2 = OnnxTensor.createTensor(env2, FloatBuffer.wrap(new float[]{1,1}), new long[]{1,2});

                Map<String,OnnxTensor> input = new HashMap<>();
                input.put("image",tensor1);
                input.put("scale_factor",tensor2);

                // ---------模型2输出-----------
                // multiclass_nms3_0.tmp_0 -> [8400, 6] -> FLOAT
                // multiclass_nms3_0.tmp_2 -> [1] -> INT32
                OrtSession.Result res = session2.run(input);

                float[][] tmp_0 = ((float[][])(res.get(0)).getValue());
                int[] tmp_2 = ((int[])(res.get(1)).getValue());

                float w_scala = Float.valueOf(src.width()) / 640f;
                float h_scala = Float.valueOf(src.height()) / 640;

                // 遍历每个框
                int count = tmp_0.length;
                for (int i = 0; i <count ; i++) {
                    // class score x1 y1 x2 y2
                    float[] data = tmp_0[i];
                    if(data[1]>0.7){
                        // 坐标缩放到原始坐标
                        data[2] = data[2] * w_scala;
                        data[3] = data[3] * h_scala;
                        data[4] = data[4] * w_scala;
                        data[5] = data[5] * h_scala;
                        datas.add(data);
                    }
                }

            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }

        }
        // 执行推理 关键点
        public void run2(){

            // 遍历每个边框截取子图
            datas.stream().forEach(n->{

                float x1 = n[2];
                float y1 = n[3];
                float x2 = n[4];
                float y2 = n[5];

                // 在原始图像中进行截图脸部
                Mat box = new Mat(
                        src,
                        new Rect(
                                Float.valueOf(x1).intValue(),
                                Float.valueOf(y1).intValue(),
                                Float.valueOf(x2 - x1).intValue(),
                                Float.valueOf(y2 - y1).intValue()
                        )
                ).clone();

                // 保存当前字图片的宽高
                int w = box.width();
                int h = box.height();

                Mat people = resizeWithoutPadding(box,192,256);


                // 只需要做 BGR -> RGB
                Imgproc.cvtColor(people, people, Imgproc.COLOR_BGR2RGB);
                // 转为浮点
                people.convertTo(people, CvType.CV_32FC1);

                float[] whc = new float[ 3 * 192 * 256 ];
                people.get(0, 0, whc);
                // 得到最终的图片转 float 数组
                float[] chw = whc2chw(whc);

                try {
                    // ---------模型1输入-----------
                    // image -> [-1, 3, 256, 192] -> FLOAT
                    OnnxTensor tensor1 = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,256,192});
                    OrtSession.Result res = session1.run(Collections.singletonMap("image", tensor1));

                    // ---------模型1输出-----------
                    // conv2d_585.tmp_1 -> [-1, 17, 64, 48] -> FLOAT
                    // argmax_0.tmp_0 -> [-1, 17] -> INT64
                    float[][][] tmp_1 = ((float[][][][])(res.get(0)).getValue())[0];
                    long[] tmp_0 = ((long[][])(res.get(1)).getValue())[0];

                    // 其中17是关键点个数, 64*48是模型输入的 256*192缩小后的关键点热图,所以两个输出如下
                    // tmp_1: 17个关键点,每个关键点生成的 64*48的热图
                    // tmp_0: 17个关键点,每个关键点的热于中最大值的索引

                    // 每个人子图定义一个关键点集合
                    ArrayList<float[]> point = new ArrayList<>();

                    int count = 17;
                    for (int i = 0; i < count; i++) {
                        // 64*48
                        float[][] sub_map = tmp_1[i];
                        // 1
                        long index_max = tmp_0[i];

                        int height = sub_map.length;
                        int width = sub_map[0].length;
                        int y = (int) (index_max / width); // 关键点在热图中的行索引
                        int x = (int) (index_max % width); // 关键点在热图中的列索引
                        float confidence = sub_map[y][x]; // 关键点的置信度

                        // 转换到 src 256*192 坐标中 64*48是模型输入的 256*192缩小后 也就是4倍
                        int xx = x * 4;
                        int yy = y * 4;

                        // 转换到截取的人图中,缩放到截图的图片上
                        xx = Float.valueOf( Float.valueOf(xx) / Float.valueOf(192) * Float.valueOf(w)).intValue();
                        yy = Float.valueOf( Float.valueOf(yy) / Float.valueOf(256) * Float.valueOf(h)).intValue();

                        // 再加上子图左上角坐标的偏移量,平移到src原始图片中
                        xx = xx + Float.valueOf(x1).intValue();
                        yy = yy + Float.valueOf(y1).intValue();

                        point.add(new float[]{xx,yy});
                    }

                    // 保存这个人的所有关键点
                    points.add(point);

                }
                catch (Exception e){
                    e.printStackTrace();
                }
            });

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


            // 遍历所有边框
            datas.stream().forEach(n->{
                // class score x1 y1 x2 y2
                float x1 = n[2];
                float y1 = n[3];
                float x2 = n[4];
                float y2 = n[5];
                // 画边框
                Imgproc.rectangle(
                        src,
                        new Point(x1,y1),
                        new Point(x2,y2),
                        color1,
                        2);
            });

            // 遍历所有关键点
            points.stream().forEach(n->{

                // 每个人有17个关键点
                n.stream().forEach(m->{
                    float xx = m[0];
                    float yy = m[1];
                    // src画一个点
                    Imgproc.circle(
                            src,
                            new Point(xx,yy),
                            1, // 半径
                            color2,
                            2);
                });

            });

            JFrame frame = new JFrame("Image");
            frame.setSize(src.width(), src.height());
            // 图片转为原始大小
            BufferedImage img = mat2BufferedImage(src);
            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // image -> [-1, 3, 256, 192] -> FLOAT
        // ---------模型1输出-----------
        // conv2d_585.tmp_1 -> [-1, 17, 64, 48] -> FLOAT
        // argmax_0.tmp_0 -> [-1, 17] -> INT64
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yoloe_pp_hrnet_human_pose_estimation\\dark_hrnet_w32_256x192.onnx");

        // ---------模型2输入-----------
        // image -> [-1, 3, 640, 640] -> FLOAT
        // scale_factor -> [-1, 2] -> FLOAT
        // ---------模型2输出-----------
        // multiclass_nms3_0.tmp_0 -> [8400, 6] -> FLOAT
        // multiclass_nms3_0.tmp_2 -> [1] -> INT32
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yoloe_pp_hrnet_human_pose_estimation\\mot_ppyoloe_l_36e_pipeline.onnx");


        // 行人图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yoloe_pp_hrnet_human_pose_estimation\\bus.jpg";
        ImageObj imageObj = new ImageObj(pic);

        // 目标检测
        imageObj.run1();

        // 关键点检测
        imageObj.run2();

        // 显示
        imageObj.show();

    }



}
