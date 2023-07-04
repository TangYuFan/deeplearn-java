package tool.deeplearning;


import ai.onnxruntime.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.dnn.Dnn;
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
import java.util.List;

/**
 *   @desc : 人脸检测（ insightface） + 人脸年龄性别检测 + 关键点识别 + 人脸特征提取  + 人脸匹配
 *
 *      参考代码：
 *      https://gitee.com/open-visual/face-search
 *
 *      模型 insightface：
 *      https://github.com/deepinsight/insightface
 *
 *      模型 pcn：
 *      https://github.com/Rock-100/FaceKit/tree/master/PCN
 *
 *
 *      模型介绍:
 *      attribute_gender_age\insight_gender_age.onnx  性别年龄识别
 *      detection_face_pcn\pcn1_sd.onnx  人脸检测(旋转)
 *      detection_face_pcn\pcn2_sd.onnx  人脸检测(旋转)
 *      detection_face_pcn\pcn3_sd.onnx  人脸检测(旋转)
 *      detection_face_scrfd\scrfd_500m_bnkps.onnx  人脸检测
 *      keypoint_coordinate\coordinate_106_mobilenet_05.onnx  人脸关键点
 *      recognition_face_arc\glint360k_cosface_r18_fp16_0.1.onnx  人脸特征提取
 *
 *      处理步骤：
 *      https://code.exmay.com/detail/1176
 *          人脸检测    =>
 *          人脸裁剪    =>
 *          关键点检测   =>
 *          关键点对齐   =>
 *          遮罩人脸以外区域    =>
 *          特征提取    =>
 *          相似度、属性等匹配检索
 *
 *
 *
*   @auth : tyf
*   @date : 2022-05-17  17:01:27
*/
public class face_rec_det_insightface_pcn_1 {

    public static OrtEnvironment env1;
    public static OrtSession session1;

    public static OrtEnvironment env2;
    public static OrtSession session2;

    public static OrtEnvironment env3;
    public static OrtSession session3;

    public static OrtEnvironment env4;
    public static OrtSession session4;

    public static OrtEnvironment env5;
    public static OrtSession session5;

    public static void init(String weight[]) throws Exception{

        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight[0], new OrtSession.SessionOptions());

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight[1], new OrtSession.SessionOptions());

        env3 = OrtEnvironment.getEnvironment();
        session3 = env3.createSession(weight[2], new OrtSession.SessionOptions());

        env4 = OrtEnvironment.getEnvironment();
        session4 = env4.createSession(weight[3], new OrtSession.SessionOptions());


        env5 = OrtEnvironment.getEnvironment();
        session5 = env5.createSession(weight[4], new OrtSession.SessionOptions());

    }

    public static void moduleInfo() throws Exception{

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

        System.out.println("---------模型[3]输入-----------");
        session3.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[3]输出-----------");
        session3.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session3.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

        System.out.println("---------模型[4]输入-----------");
        session4.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[4]输出-----------");
        session4.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session4.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });


        System.out.println("---------模型[5]输入-----------");
        session5.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型[5]输出-----------");
        session5.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session5.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });
    }

    public static class FaceEmbed{

        Mat src; // 原始图片
        int w; // 图片原始宽高
        int h; // 图片原始宽高

        // scrfd
        List<float[]> faceInfo_scrfd; // 人脸检测
        List<Mat> face_scrfd; // 人脸检测后对box扩展后截取的face
        List<List<float[]>> faceKeyPoint_scrfd; // 对截取face进行关键点检测得到的关键点
        List<Mat> face_aligned_scrfd;// 对截取的人脸和关键点进行对齐,对齐后是112*112
        List<Mat> face_hide_bg_scrfd;// 对齐后的人脸进行人脸分割去掉背景,112*112
        List<String> faceAttribute_scrfd;// 年龄和性别
        List<float[]> faceCode_scrfd;// 特征提取

        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0,255, 0);
        Scalar color3 = new Scalar(255, 0, 0);

        //5对对齐矩阵
        double[][] dst_points_5 = new double[][]{
                {30.2946f + 8.0000f, 51.6963f},
                {65.5318f + 8.0000f, 51.6963f},
                {48.0252f + 8.0000f, 71.7366f},
                {33.5493f + 8.0000f, 92.3655f},
                {62.7299f + 8.0000f, 92.3655f}
        };

        public FaceEmbed(String path){

            this.src = this.readImg(path);
            this.w = this.src.width();
            this.h = this.src.height();

            // 集合初始化
            face_scrfd = new ArrayList<>();
            faceInfo_scrfd = new ArrayList<>();
            faceKeyPoint_scrfd = new ArrayList<>();
            face_aligned_scrfd = new ArrayList<>();
            face_hide_bg_scrfd = new ArrayList<>();
            faceAttribute_scrfd = new ArrayList<>();
            faceCode_scrfd = new ArrayList<>();


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


        // 人脸检测 scrfd
        public void doDetection_scrfd(float score_thres,float iou_thres){

            // 模型输入最大尺寸
            int net_wh = 640;
            // 检测步长
            int[] net_stride = new int[]{8, 16, 32};

            // 通过原始图片尺寸和模型最大尺寸计算缩放比例
            float imgScale = 1.0f;

            // 模型输入张量
            Mat mat = null;
            int imageWidth = (int)src.size().width;
            int imageHeight = (int)src.size().height;
            int modelWidth = imageWidth, modelHeight = imageHeight;
            if(imageWidth > net_wh || imageHeight > net_wh){
                if(imageWidth > imageHeight){
                    modelWidth = net_wh;
                    imgScale = 1.0f * imageWidth / net_wh;
                    modelHeight = imageHeight * net_wh / imageWidth;
                }else {
                    modelHeight = net_wh ;
                    imgScale = 1.0f * imageHeight / net_wh;
                    modelWidth = modelWidth * net_wh / imageHeight;
                }
                mat = resizeWithoutPadding(src,modelWidth,modelHeight);
            }else{
                mat = src.clone();
            }


            Scalar mean = new Scalar(127.5, 127.5, 127.5); // 减去均值 -127.5~127.5
            double scale = 1.0/128; // 除均方差
            boolean swapRB = true; // 交换红蓝通道

            // 使用 opencv.dnn 模块进行预处理
            Mat dst = Dnn.blobFromImage(mat, scale, new Size(mat.cols(),mat.rows()),mean, swapRB);
            List<Mat> mats = new ArrayList<>();
            Dnn.imagesFromBlob(dst, mats);

            // 预处理之后的输入mat
            Mat input = mats.get(0);
            int width = input.cols();
            int height = input.rows();
            int channel = input.channels();

            // 转为4维数组
            float[][][][] array = new float[1][channel][height][width];
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    // 获取 input rgb
                    double[] c = input.get(i, j);
                    // 三个通道
                    for(int k=0; k< channel; k++){
                        float ck = (float) c[k];
                        array[0][k][i][j] = ck;
                    }
                }
            }


            // 转为张量并推理
            try {


                // ---------模型[2]输入-----------
                // input.1 -> [1, 3, -1, -1] -> FLOAT
                // ---------模型[2]输出-----------
                // 447 -> [12800, 1] -> FLOAT
                // 473 -> [3200, 1] -> FLOAT
                // 499 -> [800, 1] -> FLOAT
                // 450 -> [12800, 4] -> FLOAT
                // 476 -> [3200, 4] -> FLOAT
                // 502 -> [800, 4] -> FLOAT
                // 453 -> [12800, 10] -> FLOAT
                // 479 -> [3200, 10] -> FLOAT
                // 505 -> [800, 10] -> FLOAT

                OnnxTensor ten = OnnxTensor.createTensor(env2, array);

                // 打印输入张量shape
                long[] shape = ten.getInfo().getShape();

                OrtSession.Result out = session2.run(Collections.singletonMap("input.1", ten));

                // 按照每个步长解析face框,模型输出9个张量,也就是3个步长,每个步长有置信度、边框坐标、点坐标
                List<float[]> temp = new ArrayList<>();
                for(int index = 0;index < net_stride.length;index++){
                    int stride = net_stride[index];
                    float[][] scores = (float[][]) out.get(index).getValue();
                    float[][] boxes = (float[][]) out.get(index + 3).getValue();
                    float[][] points = (float[][]) out.get(index + 6).getValue();
                    int ws = (int) Math.ceil(1.0f * shape[3] / stride);
                    // 人脸框的个数
                    int count = scores.length;
                    for(int i=0;i<count;i++){
                        float score = scores[i][0];// 分数
                        if(score >= score_thres){
                            int anchorIndex = i / 2;
                            int rowNum = anchorIndex / ws;
                            int colNum = anchorIndex % ws;
                            //计算人脸框,坐标缩放到原始图片中
                            float anchorX = colNum * net_stride[index];
                            float anchorY = rowNum * net_stride[index];
                            float x1 = (anchorX - boxes[i][0] * net_stride[index])  * imgScale;
                            float y1 = (anchorY - boxes[i][1] * net_stride[index])  * imgScale;
                            float x2 = (anchorX + boxes[i][2] * net_stride[index])  * imgScale;
                            float y2 = (anchorY + boxes[i][3] * net_stride[index])  * imgScale;
                            // 关键点集合
                            float [] point = points[i];
                            // 5个关键点,坐标缩放到原始图片中
                            float pointX_1 = (point[0] * net_stride[index] + anchorX)  * imgScale;
                            float pointY_1 = (point[1] * net_stride[index] + anchorY)  * imgScale;
                            float pointX_2 = (point[2] * net_stride[index] + anchorX)  * imgScale;
                            float pointY_2 = (point[3] * net_stride[index] + anchorY)  * imgScale;
                            float pointX_3 = (point[4] * net_stride[index] + anchorX)  * imgScale;
                            float pointY_3 = (point[5] * net_stride[index] + anchorY)  * imgScale;
                            float pointX_4 = (point[6] * net_stride[index] + anchorX)  * imgScale;
                            float pointY_4 = (point[7] * net_stride[index] + anchorY)  * imgScale;
                            float pointX_5 = (point[8] * net_stride[index] + anchorX)  * imgScale;
                            float pointY_5 = (point[8] * net_stride[index] + anchorY)  * imgScale;
                            // 保存到tmp中
                            temp.add(new float[]{
                                    score,
                                    x1,y1,x2,y2,
                                    pointX_1,pointY_1,
                                    pointX_2,pointY_2,
                                    pointX_3,pointY_3,
                                    pointX_4,pointY_4,
                                    pointX_5,pointY_5
                            });
                        }
                    }
                }
                // temp 中 进行IOU 过滤,先按照 score 排序
                temp.sort((o1, o2) -> Float.compare(o2[0],o1[0]));
                // 保存最终的 face 信息
                List<float[]> faceInfo = new ArrayList<>();
                while (!temp.isEmpty()){
                    float[] maxObj = temp.get(0);
                    faceInfo.add(maxObj);
                    Iterator<float[]> it = temp.iterator();
                    while (it.hasNext()) {
                        float[] obj = it.next();
                        double iou = calculateIoU(maxObj, obj);
                        if (iou > iou_thres) {
                            it.remove();
                        }
                    }
                }
                // 遍历最终的face信息,截取等等
                faceInfo.stream().forEach(n->{
                    faceInfo_scrfd.add(n);
                    // 人脸边框,为了防止人脸部分未被框住造成后续关键点计算误差,通常按照中心点扩大1.5倍
                    float x1 = n[1];
                    float y1 = n[2];
                    float x2 = n[3];
                    float y2 = n[4];
                    // 计算中心点,然后按照中心点坐标朝四周扩大1.5倍再截图保存
                    float expend = 1.5f;
                    float[] box = scala_x1_y1_x2_y2(x1,y1,x2,y2,expend);
                    x1 = box[0];
                    y1 = box[1];
                    x2 = box[2];
                    y2 = box[3];
                    // 截图保存 xywh
                    face_scrfd.add(
                            new Mat(src, new Rect(
                                            Float.valueOf(x1).intValue(),
                                            Float.valueOf(y1).intValue(),
                                            Float.valueOf(x2 - x1).intValue(),
                                            Float.valueOf(y2 - y1).intValue())).clone()
                    );
                });
            }
            catch (Exception e){
                e.printStackTrace();
            }

            // 释放所有资源
            dst.release();
            mat.release();
            input.release();
        }

        // 人脸关键点检测
        public void doKeyPointRec_scrfd(){

            // 遍历每个人脸框
            face_scrfd.stream().forEach(face->{

                // 模型的输入宽高 192*192
                int netwh = 192;

                // 截图的宽高
                int face_w = face.width();
                int face_h = face.height();

                float scale_w = Float.valueOf(netwh) / Float.valueOf(face_w);
                float scale_h = Float.valueOf(netwh) / Float.valueOf(face_h);

                // 转为模型输入
                Mat face_resize = resizeWithoutPadding(face,192,192);

                Scalar mean = new Scalar(0, 0, 0);// 均值
                double scale = 1.0; // 均方差
                boolean swapRB = true; // 交换红蓝通道

                // 使用 opencv.dnn 模块进行预处理
                Mat dst = Dnn.blobFromImage(face_resize, scale, new Size(face_resize.cols(),face_resize.rows()),mean, swapRB);
                List<Mat> mats = new ArrayList<>();
                Dnn.imagesFromBlob(dst, mats);

                // 预处理之后的输入mat
                Mat input = mats.get(0);
                int width = input.cols();
                int height = input.rows();
                int channel = input.channels();

                // 转为4维数组
                float[][][][] array = new float[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        // 获取 input rgb
                        double[] c = input.get(i, j);
                        // 三个通道
                        for(int k=0; k< channel; k++){
                            float ck = (float) c[k];
                            array[0][k][i][j] = ck;
                        }
                    }
                }


                // 转为张量并进行推理
                try {

                    // ---------模型[3]输入-----------
                    // data -> [-1, 3, 192, 192] -> FLOAT
                    // ---------模型[3]输出-----------
                    // fc1 -> [1, 212] -> FLOAT


                    OnnxTensor ten = OnnxTensor.createTensor(env3, array);
                    OrtSession.Result out = session3.run(Collections.singletonMap("data", ten));
                    // 解析 106关键点
                    float[] value = ((float[][]) out.get(0).getValue())[0];
                    List<float[]> p = new ArrayList<>();
                    for(int i=0; i< 106; i++){
                        float x = (value[2*i] + 1) * 96;
                        float y = (value[2*i + 1] + 1) * 96;
                        // 保存当前人脸当前关键点,缩放到原始人脸尺寸中
                        p.add(new float[]{
                                x/scale_w,
                                y/scale_h
                        });
                    }
                    // 保存当前人脸所有关键点
                    faceKeyPoint_scrfd.add(p);
                }
                catch (Exception e){
                    e.printStackTrace();
                }


            });

        }

        // 将人脸以外的背景部分进行屏蔽
        public void doHideFaceBg_scrfd(float thres){

            // 遍历每个截取的人脸进行人脸实例分割,去掉背景
            // TODO 这个模型用于分割肖像对头发、手等人体信息依然进行了保留,后续可以替换换纯人脸的分割模型
            face_aligned_scrfd.stream().forEach(face->{

                Mat input = this.resizeWithoutPadding(face,192,192);
                Mat mat = input.clone();

                // 只需要做 BGR -> RGB
                Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
                //  归一化 0-255 转 0-1
                input.convertTo(input, CvType.CV_32FC1, 1. / 255);

                // 减去均值再除以均方差
                double[] meanValue = {0.5, 0.5, 0.5};
                double[] stdValue = {0.5, 0.5, 0.5};
                Core.subtract(input, new Scalar(meanValue), input);
                Core.divide(input, new Scalar(stdValue), input);

                // 初始化一个输入数组 channels * netWidth * netHeight
                float[] whc = new float[ 3 * 192 * 192 ];
                input.get(0, 0, whc);
                // 得到最终的图片转 float 数组
                float[] chw = new float[whc.length];
                int j = 0;
                for (int ch = 0; ch < 3; ++ch) {
                    for (int i = ch; i < whc.length; i += 3) {
                        chw[j] = whc[i];
                        j++;
                    }
                }

                // 推理
                try {

                    // ---------模型[5]输入-----------
                    // x -> [1, 3, 192, 192] -> FLOAT
                    // ---------模型[5]输出-----------
                    // tf.identity -> [1, 192, 192, 2] -> FLOAT

                    OnnxTensor tensor = OnnxTensor.createTensor(env5, FloatBuffer.wrap(chw), new long[]{1,3,192,192});
                    OrtSession.Result res = session5.run(Collections.singletonMap("x", tensor));
                    float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];
                    for(int y=0;y<192;y++){
                        for(int x=0;x<192;x++){
                            float d1 = data[y][x][0];
                            float d2 = data[y][x][1];
                            // 背景
                            if(d1>thres){
                                // 修改颜色
                                mat.put(y,x,new double[]{255,255,255});
                            }
                        }
                    }
                }
                catch (Exception e){
                    e.printStackTrace();
                }


                // 显示去掉背景的人脸后转为 112*112 保存
                face_hide_bg_scrfd.add(resizeWithoutPadding(mat,112,112));

            });

        }

        // 人脸对齐
        public void doFaceAligned_scrfd(){

            // scrfd_500m_bnkps.onnx 在人脸检测的同时就能输出5个关键点就能进行对齐,但是偶尔关键点有点飘
            // coordinate_106_mobilenet_05.onnx 这个模型检测的106点更加精确,使用 pi==38||pi==88||pi==80||pi==52||pi==61 这五个点进行对齐


            // 遍历每个人脸和关键点进行对齐,生成新的人脸
            int count = face_scrfd.size();
            for(int i=0;i<count;i++){
                Mat face = face_scrfd.get(i);// 人脸矩阵
                // 人脸关键点,对齐可以5点对齐,也可以1067对齐,这里仅取5点对齐
                List<float[]> point = faceKeyPoint_scrfd.get(i);
                float[] p1 = point.get(38);
                float[] p2 = point.get(88);
                float[] p3 = point.get(80);
                float[] p4 = point.get(52);
                float[] p5 = point.get(61);
                double[][] alignedPoint = new double[][]{
                        new double[]{p1[0],p1[1]},
                        new double[]{p2[0],p2[1]},
                        new double[]{p3[0],p3[1]},
                        new double[]{p4[0],p4[1]},
                        new double[]{p5[0],p5[1]},
                };
                // 进行对齐生成新的人脸矩阵,同时转为112*112
                Mat alignMat = this.alignedImage(face, alignedPoint, 112, 112, dst_points_5);

                // 保存对齐后的人脸
                face_aligned_scrfd.add(alignMat);
            }


        }

        // 人脸属性识别
        public void doFaceAttribute_scrfd(){

            face_scrfd.stream().forEach(n->{

                //  96*96
                Mat face = resizeWithoutPadding(n,96,96);

                Scalar mean = new Scalar(0,0,0);// 均值
                double scale = 1.0; // 均方差
                boolean swapRB = true; // 交换红蓝通道

                // 使用 opencv.dnn 模块进行预处理
                Mat dst = Dnn.blobFromImage(face, scale, new Size(face.cols(),face.rows()),mean, swapRB);
                List<Mat> mats = new ArrayList<>();
                Dnn.imagesFromBlob(dst, mats);

                // 预处理之后的输入mat
                Mat input = mats.get(0);
                int width = input.cols();
                int height = input.rows();
                int channel = input.channels();

                // 转为4维数组
                float[][][][] array = new float[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        // 获取 input rgb
                        double[] c = input.get(i, j);
                        // 三个通道
                        for(int k=0; k< channel; k++){
                            float ck = (float) c[k];
                            array[0][k][i][j] = ck;
                        }
                    }
                }

                try {

                    // ---------模型[1]输入-----------
                    // data -> [-1, 3, 96, 96] -> FLOAT
                    // ---------模型[1]输出-----------
                    // fc1 -> [1, 3] -> FLOAT

                    // 转为张量并推理
                    OnnxTensor ten = OnnxTensor.createTensor(env1, array);
                    OrtSession.Result out = session1.run(Collections.singletonMap("data", ten));

                    float[] value = ((float[][]) out.get(0).getValue())[0];
                    String age = Double.valueOf(Math.floor(value[2] * 100)).intValue() +"";
                    String gender = (value[0] > value[1]) ? "women" : "men";

                    faceAttribute_scrfd.add(gender+":"+age);

                }
                catch (Exception e){
                    e.printStackTrace();
                }

            });

        }

        // 人脸编码
        public void doFaceCode_scrfd(){

            // 遍历每个人脸 112*112
            face_hide_bg_scrfd.stream().forEach(face->{

                Scalar mean = new Scalar(127.5, 127.5, 127.5);// 均值
                double scale = 1.0/127.5; // 均方差
                boolean swapRB = true; // 交换红蓝通道

                // 使用 opencv.dnn 模块进行预处理
                Mat dst = Dnn.blobFromImage(face, scale, new Size(face.cols(),face.rows()),mean, swapRB);
                List<Mat> mats = new ArrayList<>();
                Dnn.imagesFromBlob(dst, mats);

                // 预处理之后的输入mat
                Mat input = mats.get(0);
                int width = input.cols();
                int height = input.rows();
                int channel = input.channels();

                // 转为4维数组
                float[][][][] array = new float[1][channel][height][width];
                for(int i=0; i<height; i++){
                    for(int j=0; j<width; j++){
                        // 获取 input rgb
                        double[] c = input.get(i, j);
                        // 三个通道
                        for(int k=0; k< channel; k++){
                            float ck = (float) c[k];
                            array[0][k][i][j] = ck;
                        }
                    }
                }

                try {

                    // ---------模型[4]输入-----------
                    // input.1 -> [-1, 3, 112, 112] -> FLOAT
                    // ---------模型[4]输出-----------
                    // 267 -> [1, 512] -> FLOAT

                    // 转为张量并推理
                    OnnxTensor ten = OnnxTensor.createTensor(env4, array);
                    OrtSession.Result out = session4.run(Collections.singletonMap("input.1", ten));

                    float[] embeds = ((float[][]) out.get(0).getValue())[0];
                    faceCode_scrfd.add(embeds);
                }
                catch (Exception e){
                    e.printStackTrace();
                }
            });

        }


        // 左上+右下两点的矩形框缩放后得到新的坐标
        public float[] scala_x1_y1_x2_y2(float x1,float y1,float x2,float y2,float scale){
            // 计算原始矩形框的宽度和高度
            float width = x2 - x1;
            float height = y2 - y1;
            // 计算缩放后的宽度和高度
            float newWidth = width * scale;
            float newHeight = height * scale;
            // 计算左上角和右下角点的新坐标
            float newX1 = x1 + (width - newWidth) / 2;
            float newY1 = y1 + (height - newHeight) / 2;
            float newX2 = newX1 + newWidth;
            float newY2 = newY1 + newHeight;
            // 返回新的四个点的坐标
            return new float[]{newX1, newY1, newX2, newY2};
        }

        // 交并比
        private double calculateIoU(float[] box1, float[] box2) {
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

        // 图像对齐
        public Mat alignedImage(Mat image, double[][] imagePoint, int stdWidth, int stdHeight, double[][] stdPoint){
            Mat warp = null;
            Mat rectMat = null;
            try {
                warp = warpAffine(image, imagePoint, stdPoint);
                double imgWidth = warp.size().width;
                double imgHeight = warp.size().height;
                if(stdWidth <= imgWidth && stdHeight <= imgHeight){
                    Mat crop = new Mat(warp, new Rect(0, 0, stdWidth, stdHeight));
                    return crop;
                }
                //计算需要裁剪的宽和高
                int h, w;
                if((1.0*imgWidth/imgHeight) >= (1.0 * stdWidth/stdHeight)){
                    h = (int) Math.floor(1.0 * imgHeight);
                    w = (int) Math.floor(1.0 * stdWidth * imgHeight / stdHeight);

                }else{
                    w = (int) Math.floor(1.0 * imgWidth);
                    h = (int) Math.floor(1.0 * stdHeight * imgWidth / stdWidth);
                }
                //需要裁剪图片
                rectMat = new Mat(warp, new Rect(0, 0, w, h));
                Mat crop = new Mat();
                Imgproc.resize(rectMat, crop, new Size(stdWidth, stdHeight), 0, 0, Imgproc.INTER_NEAREST);
                return crop;
            }finally {
                if(null != rectMat){
                    rectMat.release();
                }
                if(null != warp){
                    warp.release();
                }
            }
        }

        // 图像放射变换
        public Mat warpAffine(Mat image, double[][] imgPoint, double[][] stdPoint){
            Mat matM = null;
            Mat matMTemp = null;
            try {
                //转换为矩阵
                RealMatrix imgPointMatrix = createMatrix(imgPoint);
                RealMatrix stdPointMatrix = createMatrix(stdPoint);
                //判断数据的行列是否一致
                int row = imgPointMatrix.getRowDimension();
                int col = imgPointMatrix.getColumnDimension();
                if(row <= 0 || col <=0 || row != stdPointMatrix.getRowDimension() || col != stdPointMatrix.getColumnDimension()){
                    throw new RuntimeException("row or col is not equal");
                }
                //求列的均值
                RealVector imgPointMeanVector = mean(imgPointMatrix, 0);
                RealVector stdPointMeanVector = mean(stdPointMatrix, 0);
                //对关键点进行减去均值
                RealMatrix imgPointMatrix1 = imgPointMatrix.subtract(createMatrix(row, imgPointMeanVector.toArray()));
                RealMatrix stdPointMatrix1 = stdPointMatrix.subtract(createMatrix(row, stdPointMeanVector.toArray()));
                //计算关键点的标准差
                double imgPointStd = std(imgPointMatrix1);
                double stdPointStd = std(stdPointMatrix1);
                //对关键点除以标准差
                RealMatrix imgPointMatrix2 = scalarDivision(imgPointMatrix1, imgPointStd);
                RealMatrix stdPointMatrix2 = scalarDivision(stdPointMatrix1, stdPointStd);
                //获取矩阵的分量
                RealMatrix pointsT = imgPointMatrix2.transpose().multiply(stdPointMatrix2);
                SingularValueDecomposition svdH = new SingularValueDecomposition(pointsT);
                RealMatrix U = svdH.getU(); RealMatrix S = svdH.getS(); RealMatrix Vt = svdH.getVT();
                //计算仿射矩阵
                RealMatrix R = U.multiply(Vt).transpose();
                RealMatrix R1 = R.scalarMultiply(stdPointStd/imgPointStd);
                RealMatrix v21 = createMatrix(1, stdPointMeanVector.toArray()).transpose();
                RealMatrix v22 = R.multiply(createMatrix(1, imgPointMeanVector.toArray()).transpose());
                RealMatrix v23 = v22.scalarMultiply(stdPointStd/imgPointStd);
                RealMatrix R2 = v21.subtract(v23);
                RealMatrix M = hstack(R1, R2);
                //变化仿射矩阵为Mat
                matMTemp = new MatOfDouble(flatMatrix(M, 1).toArray());
                matM = new Mat(2, 3, CvType.CV_32FC3);
                matMTemp.reshape(1,2).copyTo(matM);
                //使用open cv做仿射变换
                Mat dst = new Mat();
                Imgproc.warpAffine(image, dst, matM, image.size());
                return dst;
            }finally {
                if(null != matM){
                    matM.release();
                }
                if(null != matMTemp){
                    matMTemp.release();
                }
            }
        }

        public RealVector flatMatrix(RealMatrix matrix, int axis){
            RealVector vector = new ArrayRealVector();
            if(0 == axis){
                for(int i=0; i< matrix.getColumnDimension(); i++){
                    vector = vector.append(matrix.getColumnVector(i));
                }
            }else{
                for(int i=0; i< matrix.getRowDimension(); i++){
                    vector = vector.append(matrix.getRowVector(i));
                }
            }
            return vector;
        }

        public RealMatrix hstack(RealMatrix matrix1, RealMatrix matrix2){
            int row = matrix1.getRowDimension();
            int col = matrix1.getColumnDimension()+matrix2.getColumnDimension();
            double[][] data = new double[row][col];
            for(int i=0;i<matrix1.getRowDimension(); i++){
                for(int j=0;j<matrix1.getColumnDimension(); j++){
                    data[i][j] = matrix1.getEntry(i, j);
                }
                for(int j=0;j<matrix2.getColumnDimension(); j++){
                    data[i][matrix1.getColumnDimension()+j] = matrix2.getEntry(i, j);
                }
            }
            return new Array2DRowRealMatrix(data);
        }

        public RealMatrix scalarDivision(RealMatrix matrix, double value){
            return matrix.scalarMultiply(1.0/value);
        }

        public double std(RealMatrix matrix){
            double[] data = new double[matrix.getColumnDimension() * matrix.getRowDimension()];
            for(int i=0;i<matrix.getRowDimension(); i++){
                for(int j=0;j<matrix.getColumnDimension(); j++){
                    data[i*matrix.getColumnDimension()+j] = matrix.getEntry(i, j);
                }
            }
            return new StandardDeviation(false).evaluate(data);
        }

        public RealVector mean(RealMatrix matrix, int axis){
            if(axis == 0){
                double[] means = new double[matrix.getColumnDimension()];
                for(int i=0;i<matrix.getColumnDimension(); i++){
                    means[i] = new Mean().evaluate(matrix.getColumn(i));
                }
                return new ArrayRealVector(means);
            }else {
                double[] means = new double[matrix.getRowDimension()];
                for(int i=0;i<matrix.getRowDimension(); i++){
                    means[i] = new Mean().evaluate(matrix.getRow(i));
                }
                return new ArrayRealVector(means);
            }
        }

        public RealMatrix createMatrix(double[][] array){
            return new Array2DRowRealMatrix(array);
        }

        public RealMatrix createMatrix(int rows, double[] array){
            double[][] data = new double[rows][array.length];
            for(int i=0; i<rows;i++){
                data[i] = array;
            }
            return new Array2DRowRealMatrix(data);
        }

        public void show_scrfd(){

            // 整个面板
            JPanel content = new JPanel();
            content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS)); // 设置垂直布局

            // 人脸个数
            int count = faceInfo_scrfd.size();


            // 遍历人脸检测信息
            for (int i = 0; i < count; i++) {
                // 原始人脸框信息
                float[] n = faceInfo_scrfd.get(i);
                float score = n[0];
                float x1 = n[1];
                float y1 = n[2];
                float x2 = n[3];
                float y2 = n[4];
                float p1_x = n[5];
                float p1_y = n[6];
                float p2_x = n[7];
                float p2_y = n[8];
                float p3_x = n[9];
                float p3_y = n[10];
                float p4_x = n[11];
                float p4_y = n[12];
                float p5_x = n[13];
                float p5_y = n[14];
                Imgproc.rectangle(src, new Point(x1,y1), new Point(x2,y2), color1, 2);
                Imgproc.circle(src, new Point(p1_x,p1_y), 2, color2, 2);
                Imgproc.circle(src, new Point(p2_x,p2_y), 2, color2, 2);
                Imgproc.circle(src, new Point(p3_x,p3_y), 2, color2, 2);
                Imgproc.circle(src, new Point(p4_x,p4_y), 2, color2, 2);
                Imgproc.circle(src, new Point(p5_x,p5_y), 2, color2, 2);
                // 年龄和性别
                String attribute = faceAttribute_scrfd.get(i);
                Imgproc.putText(src, attribute, new Point(x1,y1), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2);
            }


            JPanel src_img = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 5));
            src_img.add(new JLabel(new ImageIcon(mat2BufferedImage(src))));
            content.add(src_img);

            for(int i=0;i<count;i++){

                // 人脸矩阵(对齐前的)
                Mat face = face_scrfd.get(i);
                // 人脸关键点
                List<float[]> point = faceKeyPoint_scrfd.get(i);
                // 人脸矩阵(对齐后的)
                Mat face_aligned = face_aligned_scrfd.get(i);
                // 人脸矩阵(去掉背景)
                Mat face_hide_bg = face_hide_bg_scrfd.get(i);
                // 特征向量
                float[] embed = faceCode_scrfd.get(i);
                // 在对齐前的人脸上进行关键点标注
                for(int pi=0;pi<point.size();pi++){
                    float x = point.get(pi)[0];
                    float y = point.get(pi)[1];
                    // 特殊五点红色其他点蓝色
                    if(pi==38||pi==88||pi==80||pi==52||pi==61){
                        Imgproc.circle(face, new Point(x,y), 1, color2, 2);
                    }else{
                        Imgproc.circle(face, new Point(x,y), 1, color3, 1);
                    }
                }
                // 显示对齐前后的人脸
                JPanel line = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 5));
                line.add(new JLabel("关键点 =>"));
                line.add(new JLabel(new ImageIcon(mat2BufferedImage(face))));// 人脸截图
                line.add(new JLabel("对齐后 =>"));
                line.add(new JLabel(new ImageIcon(mat2BufferedImage(face_aligned))));// 人脸截图(对齐后)
                line.add(new JLabel("去背景 =>"));
                line.add(new JLabel(new ImageIcon(mat2BufferedImage(face_hide_bg))));// 人脸截图(去掉背景)
                line.add(new JLabel("特征向量 =>"));
                line.add(new JLabel("[" + embed[0] + ",...," + embed[511] + "]"));
                content.add(line);
            }


            // 自适应大小
            JFrame frame = new JFrame();
            frame.add(content);
            frame.setVisible(true);
            frame.setResizable(false);
            frame.pack();
        }


    }


    public static void main(String[] args) throws Exception{

        // ---------模型[1]输入-----------
        // data -> [-1, 3, 96, 96] -> FLOAT
        // ---------模型[1]输出-----------
        // fc1 -> [1, 3] -> FLOAT

        // ---------模型[2]输入-----------
        // input.1 -> [1, 3, -1, -1] -> FLOAT
        // ---------模型[2]输出-----------
        // 447 -> [12800, 1] -> FLOAT
        // 473 -> [3200, 1] -> FLOAT
        // 499 -> [800, 1] -> FLOAT
        // 450 -> [12800, 4] -> FLOAT
        // 476 -> [3200, 4] -> FLOAT
        // 502 -> [800, 4] -> FLOAT
        // 453 -> [12800, 10] -> FLOAT
        // 479 -> [3200, 10] -> FLOAT
        // 505 -> [800, 10] -> FLOAT

        // ---------模型[3]输入-----------
        // data -> [-1, 3, 192, 192] -> FLOAT
        // ---------模型[3]输出-----------
        // fc1 -> [1, 212] -> FLOAT

        // ---------模型[4]输入-----------
        // input.1 -> [-1, 3, 112, 112] -> FLOAT
        // ---------模型[4]输出-----------
        // 267 -> [1, 512] -> FLOAT

        // ---------模型[5]输入-----------
        // x -> [1, 3, 192, 192] -> FLOAT
        // ---------模型[5]输出-----------
        // tf.identity -> [1, 192, 192, 2] -> FLOAT


        // 资源路径
        String root  = new File("").getCanonicalPath() + "\\model\\deeplearning\\face_rec_det_insightface_pcn\\";

        // 模型加载
        String[] weights = new String[]{
                root + "attribute_gender_age\\insight_gender_age.onnx",// 性别年龄识别
                root + "detection_face_scrfd\\scrfd_500m_bnkps.onnx",// 人脸检测(效率高但旋转人脸检测效果低一些)
                root + "keypoint_coordinate\\coordinate_106_mobilenet_05.onnx",// 人脸关键点检测用于对齐
                root + "recognition_face_arc\\glint360k_cosface_r18_fp16_0.1.onnx",// 人脸特征提取用于编码
                root + "pp_human_seg\\model_float32.onnx" // 肖像分割,去掉人脸截图的背景
        };
        init(weights);

        // 模型信息打印
        moduleInfo();

        // 人脸图片,支持多个人脸
        FaceEmbed faceEmbed = new FaceEmbed( root + "pic2.png" );

        // scrfd
        faceEmbed.doDetection_scrfd(0.5f,0.7f);// 人脸检测,扩展人脸区域并截取
        faceEmbed.doFaceAttribute_scrfd();// 人脸年龄、性别检测
        faceEmbed.doKeyPointRec_scrfd();// 截取的人脸区域进行关键点检测,得到106点
        faceEmbed.doFaceAligned_scrfd();// 截取的人脸区域进行5点对齐,并转为112*112,得到对齐后的人脸
        faceEmbed.doHideFaceBg_scrfd(0.9f);// 对齐后的肖像分割去掉背景,得到最终需要进行编码的人脸 112*112
        faceEmbed.doFaceCode_scrfd();// 人脸编码提取特征向量
        faceEmbed.show_scrfd();// 弹窗显示


    }





}
