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
import java.util.*;

/**
*   @desc : 人脸检测与识别 sface ： 人脸检测(yunet) + 人脸编码128维(sface)
 *
 *      这两个模块是由人脸识别领域的两位大牛设计的，
 *      其中人脸检测是南科大的于仕琪老师设计的，人脸识别模块是北邮的邓伟洪教授设计的，
 *      其研究成果SFace发表于 图像处理顶级期刊IEEE Transactions on Image Processing。
 *
*   @auth : tyf
*   @date : 2022-05-23  16:17:37
*/
public class face_rec_det_sface {


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


    public static class FaceEmbed{

        String path;
        Mat src; // 原始图片
        int w; // 图片原始宽高
        int h; // 图片原始宽高
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);
        // 用于人脸检测的先验框
        private List<float[]> priors;
        private float[] variance;
        // 保存人脸的得分/边框/关键点
        ArrayList<float[]> faceBoxPoint;
        // 保存人脸对齐后的矩阵
        ArrayList<Mat> faceAffine;
        // 保存人脸编码
        ArrayList<float[]> faceCode;
        public FaceEmbed(String path){
            // 检测模型不用resize
            this.path = path;
            this.src = this.readImg(path);
            this.w = this.src.width();
            this.h = this.src.height();
            // 计算先验框
            this.getPriors(w,h);
        }
        // 计算先眼眶
        public void getPriors(int inputW, int inputH){

            this.priors = new ArrayList<>();
            this.variance = new float[]{0.1f, 0.2f};

            Size[] featureMapSizes = {
                    new Size((inputW + 1) / 2 / 2 / 2, (inputH + 1) / 2 / 2 / 2),
                    new Size((inputW + 1) / 2 / 2 / 2 / 2, (inputH + 1) / 2 / 2 / 2 / 2),
                    new Size((inputW + 1) / 2 / 2 / 2 / 2/ 2, (inputH + 1) / 2 / 2 / 2 / 2 / 2),
                    new Size((inputW + 1) / 2 / 2 / 2 / 2 / 2/ 2, (inputH + 1) / 2 / 2 / 2 / 2 / 2 / 2)
            };

            List<List<Float>> minSizes = new ArrayList<>();
            minSizes.add(Arrays.asList(10.0f, 16.0f, 24.0f));
            minSizes.add(Arrays.asList(32.0f, 48.0f));
            minSizes.add(Arrays.asList(64.0f, 96.0f));
            minSizes.add(Arrays.asList(128.0f, 192.0f, 256.0f));

            List<Integer> steps = Arrays.asList(8, 16, 32, 64);

            for (int i = 0; i < featureMapSizes.length; i++) {
                Size featureMapSize = featureMapSizes[i];
                List<Float> minSize = minSizes.get(i);
                for (int h = 0; h < featureMapSize.height; h++) {
                    for (int w = 0; w < featureMapSize.width; w++) {
                        for (float size : minSize) {
                            float s_kx = size / inputW;
                            float s_ky = size / inputH;
                            float cx = (w + 0.5f) * steps.get(i) / inputW;
                            float cy = (h + 0.5f) * steps.get(i) / inputH;
                            priors.add(new float[]{cx, cy, s_kx, s_ky});
                        }
                    }
                }
            }
        }
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(netWidth, netHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }
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

        // 人脸检测
        public void doDet(){


            float confThreshold = 0.7f;
            float nmsThreshold = 0.3f;

            Mat input = src.clone();

            // 模型输入不需要resize
            // BGR -> RGB
            Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
            input.convertTo(input, CvType.CV_32FC1);

            float[] whc = new float[ 3 * input.width() * input.height() ];
            input.get(0, 0, whc);

            // 得到最终的图片转 float 数组
            float[] chw = whc2chw(whc);

            try {
                // ---------模型1输入-----------
                // input -> [-1, 3, -1, -1] -> FLOAT
                OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,input.height(),input.width()});

                // 推理
                OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));

                // ---------模型1输出-----------
                // loc -> [-1, 14] -> FLOAT
                // conf -> [-1, 2] -> FLOAT
                // iou -> [-1, 1] -> FLOAT
                float[][] loc = ((float[][])(res.get(0)).getValue());
                float[][] conf = ((float[][])(res.get(1)).getValue());
                float[][] iou = ((float[][])(res.get(2)).getValue());


                // 用于nms过滤
                ArrayList<float[]> tmp = new ArrayList<>();

//                System.out.println("priors:"+priors.size());
//                System.out.println("loc:"+loc.length);
//                System.out.println("conf:"+conf.length);
//                System.out.println("iou:"+iou.length);

                // 按照先验框个数
                for (int i = 0; i < priors.size(); ++i) {
                    // Get score
                    float clsScore = conf[i][1];
                    float iouScore = iou[i][0];
                    // Clamp
                    if (iouScore < 0.f) {
                        iouScore = 0.f;
                    }
                    else if (iouScore > 1.f) {
                        iouScore = 1.f;
                    }
                    // 得分
                    float score = Double.valueOf(Math.sqrt(clsScore * iouScore)).floatValue();

                    // 边框
                    float cx = (priors.get(i)[0] + loc[i][0] * variance[0] * priors.get(i)[2])  * src.width();
                    float cy = (priors.get(i)[1] + loc[i][1] * variance[0] * priors.get(i)[3]) * src.height();
                    float w = Double.valueOf(priors.get(i)[2]  * Math.exp(loc[i][2] * variance[0]) * src.width()).floatValue();
                    float h = Double.valueOf(priors.get(i)[3] * Math.exp(loc[i][0] * variance[1]) * src.height()).floatValue();
                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;
                    float x2 = cx + w / 2;
                    float y2 = cy + h / 2;

                    // 关键点
                    float p1_x = (priors.get(i)[0] + loc[i][4] * variance[0] * priors.get(i)[2])  * src.width();  // right eye, x
                    float p1_y = (priors.get(i)[1] + loc[i][5] * variance[0] * priors.get(i)[3]) * src.height();  // right eye, y
                    float p2_x = (priors.get(i)[0] + loc[i][6] * variance[0] * priors.get(i)[2])  * src.width();  // left eye, x
                    float p2_y = (priors.get(i)[1] + loc[i][7] * variance[0] * priors.get(i)[3]) * src.height();  // left eye, y
                    float p3_x = (priors.get(i)[0] + loc[i][8] * variance[0] * priors.get(i)[2])  * src.width();  // nose tip, x
                    float p3_y = (priors.get(i)[1] + loc[i][9] * variance[0] * priors.get(i)[3]) * src.height();  // nose tip, y
                    float p4_x = (priors.get(i)[0] + loc[i][10] * variance[0] * priors.get(i)[2])  * src.width(); // right corner of mouth, x
                    float p4_y = (priors.get(i)[1] + loc[i][11] * variance[0] * priors.get(i)[3]) * src.height(); // right corner of mouth, y
                    float p5_x = (priors.get(i)[0] + loc[i][12] * variance[0] * priors.get(i)[2])  * src.width(); // left corner of mouth, x
                    float p5_y = (priors.get(i)[1] + loc[i][13] * variance[0] * priors.get(i)[3]) * src.height(); // left corner of mouth, y

                    // 根据阈值进行首次过滤
                    if(score>confThreshold){
                        // 保存得分 边框、关键点
                        tmp.add(new float[]{
                                score,
                                x1,y1,x2,y2,
                                p1_x,p1_y,
                                p2_x,p2_y,
                                p3_x,p3_y,
                                p4_x,p4_y,
                                p5_x,p5_y
                        });

                    }
                }

                System.out.println("NMS过滤前边框:"+tmp.size());

                // 用于nms过滤后保存的
                ArrayList<float[]> box = new ArrayList<>();


                // 开始nms过滤先按照得分排序
                tmp.sort((o1, o2) -> Float.compare(o2[0],o1[0]));

                while (!tmp.isEmpty()){
                    float[] max = tmp.get(0);
                    box.add(max);
                    Iterator<float[]> it = tmp.iterator();
                    while (it.hasNext()) {
                        // 交并比
                        float[] obj = it.next();
                        double iouValue = calculateIoU(
                                new float[]{max[1],max[2],max[3],max[4]},
                                new float[]{obj[1],obj[2],obj[3],obj[4]}
                        );
//                        System.out.println("iouValue:"+iouValue);
                        if (iouValue > nmsThreshold) {
                            it.remove();
                        }
                    }
                }

                System.out.println("NMS过滤后边框:"+box.size());


                this.faceBoxPoint = box;

            }
            catch (Exception e){
                e.printStackTrace();
            }
        }

        // 计算两个框的交并比
        private double calculateIoU(float[] box1, float[] box2) {

            //  getXYXY() 返回 xmin-0 ymin-1 xmax-2 ymax-3

            double x1 = Math.max(box1[0], box2[0]);
            double y1 = Math.max(box1[1], box2[1]);
            double x2 = Math.min(box1[2], box2[2]);
            double y2 = Math.min(box1[3], box2[3]);
            double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
            double box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
            double box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
            double unionArea = box1Area + box2Area - intersectionArea;
            return intersectionArea / unionArea;
        }

        // 特征提取
        public void doRec(){

            this.faceCode = new ArrayList<>();

            // 对每个人脸进行特征提取
            faceAffine.stream().forEach(face->{


                Mat input = face.clone();

                // BGR -> RGB
                Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
                input.convertTo(input, CvType.CV_32FC1);

                float[] whc = new float[ 3 * 112 * 112 ];
                input.get(0, 0, whc);

                // 得到最终的图片转 float 数组
                float[] chw = whc2chw(whc);

                try {
                    // ---------模型2输入-----------
                    // data -> [1, 3, 112, 112] -> FLOAT
//                    OnnxTensor tensor = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1, 3, 112, 112});
//                    OrtSession.Result out = session2.run(Collections.singletonMap("data", tensor));
//
//                    // ---------模型2输出-----------
//                    // fc1 -> [1, 128] -> FLOAT
//                    float[] embeds = ((float[][]) out.get(0).getValue())[0];
//
//                    // 保存人脸编码
//                    this.faceCode.add(embeds);

                }catch (Exception e){
                    e.printStackTrace();
                }


            });


        }

        // 进行人脸编码全部流程
        public void doWork(){

            // 人脸检测
            doDet();

            // 人脸截取并仿射变换对齐
            doAffineTransform();

            // 特征提取
            doRec();
        }


        // 进行所有人脸的仿射变换
        public void doAffineTransform(){

            faceAffine = new ArrayList<>();

            faceBoxPoint.stream().forEach(n->{


                // 从原始图片中将五个关键点进行对齐并得到 112*112的对齐后的人脸图片
                float[] p1 = new float[]{n[5],n[6]};
                float[] p2 = new float[]{n[7],n[8]};
                float[] p3 = new float[]{n[9],n[10]};
                float[] p4 = new float[]{n[11],n[12]};
                float[] p5 = new float[]{n[13],n[14]};

                // 定义仿射变换的目标尺寸,也就是下一个模型的输入尺寸
                // 注意这里没有传入人脸检测得到的边框坐标来截取,而是根据关键点仿射变换比例自动从原图中获取人脸,这样保证人脸截取更全面
                Mat dst = alignAndCrop(src,p1,p2,p3,p4,p5,112,112);

                // 保存对齐后的矩阵
                faceAffine.add(dst);
            });

        }

        // 从原始图片中按照 x1y1x2y2截取,截取后按照 p1 p2 p3 p4 p5 五个关键点仿射变换进行对齐,最后返回对齐后的Mat, 其中 dsth/dstw 是变换后目标宽高
        public Mat alignAndCrop(Mat src,float[] p1, float[] p2, float[] p3, float[] p4, float[] p5, float dsth, float dstw){

            // 定义对齐后的关键点坐标， opencv 的对其函数只需要3点
            Point[] alignedPoints = new Point[3];
            alignedPoints[0] = new Point(p1[0], p1[1]);
            alignedPoints[1] = new Point(p2[0], p2[1]);
            alignedPoints[2] = new Point(p3[0], p3[1]);

            // 定义对齐后的目标关键点坐标 ,也就是将5个点落到固定的坐标位置,以 112 * 112 为例 五个点的位置可以通过比例进行计算
            // 尽量选择合适的点将人脸框住例如 112*112 尺寸下的5点对齐矩阵:
            // double[][] dst_points_5 = new double[][]{
            //                {30.2946f + 8.0000f, 51.6963f},
            //                {65.5318f + 8.0000f, 51.6963f},
            //                {48.0252f + 8.0000f, 71.7366f},
            //                {33.5493f + 8.0000f, 92.3655f},
            //                {62.7299f + 8.0000f, 92.3655f}
            //        };
            // 这部分对齐目标点在一些特征提取网络中都能找到 python 代码:
            //         self._dst = np.array([
            //            [38.2946, 51.6963],
            //            [73.5318, 51.5014],
            //            [56.0252, 71.7366],
            //            [41.5493, 92.3655],
            //            [70.7299, 92.2041]
            //        ], dtype=np.float32)
            Point[] targetPoints = new Point[3];
            targetPoints[0] = new Point(30.2946f + 8.0000f, 51.6963f); // 左眼角 ： 上半部分左边的正中心
            targetPoints[1] = new Point(65.5318f + 8.0000f, 51.6963f); // 右眼角 ： 上半部分右边的正中心
            targetPoints[2] = new Point(48.0252f + 8.0000f, 71.7366f); // 鼻尖 ： 正中心
            // 将关键点坐标转换为MatOfPoint2f类型
            MatOfPoint2f srcPoints = new MatOfPoint2f(alignedPoints);
            MatOfPoint2f dstPoints = new MatOfPoint2f(targetPoints);

            // 计算仿射变换矩阵
            Mat transformation = Imgproc.getAffineTransform(srcPoints, dstPoints);

            // 对齐图像
            Mat aligned = new Mat();
            Imgproc.warpAffine(src, aligned, transformation, new Size(dstw, dsth));

            return aligned;
        }

        // 显示
        public void show(){


            JPanel root = new JPanel();
            root.setLayout(new BoxLayout(root, BoxLayout.Y_AXIS));

            // 画边框
            faceBoxPoint.stream().forEach(n->{
                Imgproc.rectangle(
                        src,
                        new Point(n[1],n[2]),
                        new Point(n[3],n[4]),
                        color1,
                        2);
            });

            // 画关键点
            faceBoxPoint.stream().forEach(n->{
                Imgproc.circle(src, new Point(n[5],n[6]), 2, color2, 2);
                Imgproc.circle(src, new Point(n[7],n[8]), 2, color2, 2);
                Imgproc.circle(src, new Point(n[9],n[10]), 2, color2, 2);
                Imgproc.circle(src, new Point(n[11],n[12]), 2, color2, 2);
                Imgproc.circle(src, new Point(n[13],n[14]), 2, color2, 2);
            });

            // 显示原始图片
            JPanel img_panel = new JPanel();
            img_panel.add(new JLabel(new ImageIcon(mat2BufferedImage(src))));

            // 显示对齐的人脸
            JPanel face_panel = new JPanel();
            faceAffine.stream().forEach(n->{
                face_panel.add(new JLabel(new ImageIcon(mat2BufferedImage(n))));
            });


            root.add(img_panel);
            root.add(face_panel);


            JFrame frame = new JFrame("Image");
            frame.getContentPane().add(root);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        }


    }

    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // input -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型1输出-----------
        // loc -> [-1, 14] -> FLOAT
        // conf -> [-1, 2] -> FLOAT
        // iou -> [-1, 1] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_rec_det_sface\\face_detection_yunet.onnx");

        // ---------模型2输入-----------
        // data -> [1, 3, 112, 112] -> FLOAT
        // ---------模型2输出-----------
        // fc1 -> [1, 128] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_rec_det_sface\\face_recognition_sface.onnx");



        // 创建人脸编码器
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_rec_det_sface\\pic1.jpg";
        FaceEmbed embed = new FaceEmbed(pic);


        // 处理
        embed.doWork();

        // 显示
        embed.show();



    }



}
