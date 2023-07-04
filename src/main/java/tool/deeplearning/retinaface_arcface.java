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
*   @desc : 人脸检测（retinaface） + 人脸识别（arcface）
*   @auth : tyf
*   @date : 2022-05-04  15:51:16
*/
public class retinaface_arcface {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;


    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型[1]输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[1]输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }


    // 环境初始化
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型[2]输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[2]输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });


    }


    // 人脸检测、识别处理 限制一张图只一个人脸
    public static class FaceImageWorker {

        // 人脸1  640*640
        Mat face;

        public FaceImageWorker(String pic){
            Mat f = this.readImg(pic);
            // 转为模型输入尺寸
            this.face = this.resizeWithoutPadding(f,640,640);
        }

        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }

        // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
        public Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(netWidth, netHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }

        public OnnxTensor tensor1(Mat dst, int channels, int netWidth, int netHeight){

            // BGR -> RGB , 注意原python代码中读取的就是BGR,opencv读取的图片默认也是BGR
//            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

            // 将图像转换为 float32 类型：img = np.float32(img_raw)
            dst.convertTo(dst, CvType.CV_32FC3);

            // 对图像进行像素均值归一化：img -= (104, 117, 123) 这个是BGR顺序
            double[] meanValue = { 104, 117, 123 };
            Core.subtract(dst, new Scalar(meanValue), dst);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ 3*640*640 ];
            dst.get(0, 0, whc);

            // 顺序转换为 C×H×W 的维度顺序：img = img.transpose(2, 0, 1)
//            float[] chw = whc2cwh(whc);
            float[] chw = whc2chw(whc,3,640,640);

            // 创建 onnxruntime 需要的 tensor
            // 传入输入的图片 float 数组并指定数组shape
            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }


        public OnnxTensor tensor2(Mat dst, int channels, int netWidth, int netHeight){

            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

            //  归一化 0-255 转 0-1
            dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

            // 初始化一个输入数组 channels * netWidth * netHeight
            float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
            dst.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2cwh(whc);

            // 创建 onnxruntime 需要的 tensor
            // 传入输入的图片 float 数组并指定数组shape
            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,channels,netHeight,netWidth});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
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

        public float[] whc2chw(float[] src, int channels, int height, int width) {
            float[] chw = new float[src.length];
            for (int ch = 0; ch < channels; ++ch) {
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        chw[ch * height * width + i * width + j] = src[(i * width + j) * channels + ch];
                    }
                }
            }
            return chw;
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

        private float sigmoid(float x) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }


        // 解析retinaface的边框、概率、关键点 传入步长用于计算缩放比例
        public void parse_stride(float[][][] cls,float[][][] bbox,float[][][] landmark,int future_w,int future_h,int future_stride){


            int scala = 640/future_w;// 特征图是20原图是640,缩放比例就是32,用于特征图中边框和关键点缩放到原始图片

            // 20 * 20 是特征图尺寸 特征图的尺寸是20x20，因此共有20x20=400个位置（也称为“锚点”）。
            // 每个锚点都对应着两个固定大小和长宽比的锚框（即边界框，一大一小），这些锚框以特征图上的当前锚点为中心。
            // 8 位的边框信息也就是 x1y1w1h1 x2y2w1h1 以特征图中的当前点生成的锚框(两个一大一小)的坐标,坐标是特征图20*20的坐标,需要进一步缩放到原图640*640
            // 4 位的概率信息分别是 背景、脸部、左眼和右眼 以特征图中的当前点生成的锚框分别属于4个类别的概率,我们只关注人脸
            // 20 位的关家点坐标 x1y1 x2y2 ... x10y10 以特征图中的当前点生成的锚框内人脸关键点坐标(两个框每个框5个点所以一共10个点),坐标时特征图20*20的坐标,需要进一步缩放到原图640*640


            List<float[]> bboxes = new ArrayList<>(); // 边框列表
            List<Float> scores = new ArrayList<>(); // 概率列表
            List<float[]> landmarks = new ArrayList<>(); // 关键点列表


            // 首先遍历特征图上所有特征点
            for (int h = 0; h < future_h; h++) {
                for (int w = 0; w < future_w; w++) {

                    // 每个特征点会生成一大一小两个框
                    // 每个框有属于人脸/背景的概率,两个框就需要4位,所以cls第一维是4
                    // 每个框有边框信息也就是xywh,两个框就需要8位,所以bbox第一维是8
                    // 每个框内有五个关键点信息x1y1x2y2x3y3x4y4x5y5,两个框就需要20位,所以landmark第一维是20

                    float score1 = cls[0][h][w]; // 边框1背景概率
                    float score2 = cls[1][h][w]; // 边框1人脸概率
                    float score3 = cls[2][h][w]; // 边框2背景概率
                    float score4 = cls[3][h][w]; // 边框2人脸概率

                    System.out.println("score1:"+score1+",score2:"+score2+",score3:"+score3+",score4:"+score4);

                    float b1_x = bbox[0][h][w]; // 边框1 x
                    float b1_y = bbox[1][h][w]; // 边框1 y
                    float b1_w = bbox[2][h][w]; // 边框1 w
                    float b1_h = bbox[3][h][w]; // 边框1 h
                    float b2_x = bbox[4][h][w]; // 边框2 x
                    float b2_y = bbox[5][h][w]; // 边框2 y
                    float b2_w = bbox[6][h][w]; // 边框2 w
                    float b2_h = bbox[7][h][w]; // 边框2 h

                    float l1_1_x = landmark[0][h][w]; // 边框1 关键点1 x
                    float l1_1_y = landmark[1][h][w]; // 边框1 关键点1 y
                    float l1_2_x = landmark[2][h][w]; // 边框1 关键点2 x
                    float l1_2_y = landmark[3][h][w]; // 边框1 关键点2 y
                    float l1_3_x = landmark[4][h][w]; // 边框1 关键点3 x
                    float l1_3_y = landmark[5][h][w]; // 边框1 关键点3 y
                    float l1_4_x = landmark[6][h][w]; // 边框1 关键点4 x
                    float l1_4_y = landmark[7][h][w]; // 边框1 关键点4 y
                    float l1_5_x = landmark[8][h][w]; // 边框1 关键点5 x
                    float l1_5_y = landmark[9][h][w]; // 边框1 关键点5 y

                    float l2_1_x = landmark[0][h][w]; // 边框2 关键点1 x
                    float l2_1_y = landmark[1][h][w]; // 边框2 关键点1 y
                    float l2_2_x = landmark[2][h][w]; // 边框2 关键点2 x
                    float l2_2_y = landmark[3][h][w]; // 边框2 关键点2 y
                    float l2_3_x = landmark[4][h][w]; // 边框2 关键点3 x
                    float l2_3_y = landmark[5][h][w]; // 边框2 关键点3 y
                    float l2_4_x = landmark[6][h][w]; // 边框2 关键点4 x
                    float l2_4_y = landmark[7][h][w]; // 边框2 关键点4 y
                    float l2_5_x = landmark[8][h][w]; // 边框2 关键点5 x
                    float l2_5_y = landmark[9][h][w]; // 边框2 关键点5 y


                    // 保存当前锚点两个边框
                    bboxes.add(new float[]{
                            b1_x,
                            b1_y,
                            b1_x + b1_w,
                            b1_y + b1_h
                    });

                    bboxes.add(new float[]{
                            b2_x,
                            b2_y,
                            b2_x + b2_w,
                            b2_y + b2_h
                    });

                    // 保存当前锚点两个边框的5个关键点


                }
            }


            // 画边框
            bboxes.stream().forEach(n->{

                float x1 = n[0] * future_h *scala;
                float y1 = n[1] * future_h *scala;
                float x2 = n[2] * future_h *scala;
                float y2 = n[3] * future_h *scala;

//                System.out.println("x1:"+x1+",y1:"+y1+",x2:"+x2+",y2:"+y2+",宽度:"+future_w+",缩放比例:"+scala);

                Imgproc.rectangle(
                        face,
                        new Point(Float.valueOf(x1).intValue(), Float.valueOf(y1).intValue()),
                        new Point(Float.valueOf(x2).intValue(), Float.valueOf(y2).intValue()),
                        new Scalar((int)(Math.random() * 256), (int)(Math.random() * 256), (int)(Math.random() * 256)),
                        2);
            });

        }



        // 人脸检测
        public void detect() throws Exception{

            // 预处理并转为tenser,注意源码中使用的 BGR格式的图片并且归一化处理
            OnnxTensor tens1 = this.tensor1(face.clone(),3,640,640);
            // 推理
            OrtSession.Result res = session1.run(Collections.singletonMap("data", tens1));

            // 模型有下列输出 cls（类别概率）、bbox（边框坐标偏移）、landmark（关键点坐标便宜） 一共32、16、8三种特征图步长大小
            // "stride32"、"stride16"、"stride8" 分别表示特征图的步长大小，用于检测不同大小的人脸。步长越小，特征图的尺寸就越大，能够检测到更小的人脸。
            // 模型输出详解: https://blog.csdn.net/jcyao_/article/details/106003213

            // 步长32
            float[][][] data0 = ((float[][][][])(res.get(0)).getValue())[0];// face_rpn_cls_prob_reshape_stride32 [1, 4, 20, 20]
            float[][][] data1 = ((float[][][][])(res.get(1)).getValue())[0];// face_rpn_bbox_pred_stride32 [1, 8, 20, 20]
            float[][][] data2 = ((float[][][][])(res.get(2)).getValue())[0];// face_rpn_landmark_pred_stride32 [1, 20, 20, 20]
            // 步长16
            float[][][] data3 = ((float[][][][])(res.get(3)).getValue())[0];// face_rpn_cls_prob_reshape_stride16 [1, 4, 40, 40]
            float[][][] data4 = ((float[][][][])(res.get(4)).getValue())[0];// face_rpn_bbox_pred_stride16 [1, 8, 40, 40]
            float[][][] data5 = ((float[][][][])(res.get(5)).getValue())[0];// face_rpn_landmark_pred_stride16 [1, 20, 40, 40]
            // 步长8
            float[][][] data6 = ((float[][][][])(res.get(6)).getValue())[0];// face_rpn_cls_prob_reshape_stride8 [1, 4, 80, 80]
            float[][][] data7 = ((float[][][][])(res.get(7)).getValue())[0];// face_rpn_bbox_pred_stride8 [1, 8, 80, 80]
            float[][][] data8 = ((float[][][][])(res.get(8)).getValue())[0];// face_rpn_landmark_pred_stride8 [1, 20, 80, 80]

            // 特征图 20*20 步长 32
            this.parse_stride(data0,data1,data2,20,20,32);
            // 特征图 40*40 步长 16
            this.parse_stride(data3,data4,data5,40,40,16);
            // 特征图 80*80 步长 8
            this.parse_stride(data6,data7,data8,80,80,8);


        }

        // 人脸编码
        public void encode() throws Exception{

        }

        public void doWork() throws Exception{
            // 人脸检测
            this.detect();
            // 人脸编码
            this.encode();
        }

    }



    public static void main(String[] args) throws Exception{

        /*
        参考文档：
        http://t.csdn.cn/fgg2N
        http://t.csdn.cn/SG8jW
         */

        // 加载 retinaface（人脸检测对齐目录下有多个，输入输出一样只是骨干网络不一样）

        // \retinaface_arcface\RetinaFace\mnet.25\mnet.25_deconv.onnx
        // \retinaface_arcface\RetinaFace\mnet.25\mnet.25_resize.onnx
        // \retinaface_arcface\RetinaFace\R50\retinaface-R50\R50_onnx.onnx
        // \retinaface_arcface\RetinaFace\retinaface_mnet025_v1\mnet025_fix_gamma_v1.onnx
        // \retinaface_arcface\RetinaFace\retinaface_mnet025_v2\mnet025_fix_gamma_v2.onnx

        // 加载 arcface（人脸编码，目录下有多个，输入输出一样只是骨干网络不一样）
        // \retinaface_arcface\ArcFace\mobilefacenet-res2-6-10-2-dim512\modelnew2_onnx.onnx
        // \retinaface_arcface\ArcFace\model-r34-amf-slim\model2_onnx.onnx


        /*
            ---------模型[1]输入-----------
            data -> [1, 3, 640, 640] -> FLOAT
            ---------模型[1]输出-----------
            face_rpn_cls_prob_reshape_stride32 -> [1, 4, 20, 20] -> FLOAT
            face_rpn_bbox_pred_stride32 -> [1, 8, 20, 20] -> FLOAT
            face_rpn_landmark_pred_stride32 -> [1, 20, 20, 20] -> FLOAT
            face_rpn_cls_prob_reshape_stride16 -> [1, 4, 40, 40] -> FLOAT
            face_rpn_bbox_pred_stride16 -> [1, 8, 40, 40] -> FLOAT
            face_rpn_landmark_pred_stride16 -> [1, 20, 40, 40] -> FLOAT
            face_rpn_cls_prob_reshape_stride8 -> [1, 4, 80, 80] -> FLOAT
            face_rpn_bbox_pred_stride8 -> [1, 8, 80, 80] -> FLOAT
            face_rpn_landmark_pred_stride8 -> [1, 20, 80, 80] -> FLOAT
         */
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\retinaface_arcface\\RetinaFace\\R50\\retinaface-R50\\R50_onnx.onnx");

        /*
            ---------模型[2]输入-----------
            data -> [1, 3, 112, 112] -> FLOAT
            ---------模型[2]输出-----------
            fc1 -> [1, 512] -> FLOAT
         */
        init2(new File("").getCanonicalPath()+"\\model\\deeplearning\\retinaface_arcface\\ArcFace\\mobilefacenet-res2-6-10-2-dim512\\modelnew2_onnx.onnx");


        // 人脸处理器,结果检测识别输出人脸特征向量
        FaceImageWorker worker = new FaceImageWorker("C:\\Users\\tyf\\Desktop\\1.png");
        worker.doWork();


        JFrame frame = new JFrame("Image");
        frame.setSize(worker.face.width(), worker.face.height());
        JLabel label = new JLabel(new ImageIcon(worker.mat2BufferedImage(worker.face)));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);



    }






}
