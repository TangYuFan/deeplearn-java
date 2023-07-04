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
import java.util.Iterator;

/**
*   @desc : 使用ONNXRuntime部署yolov5旋转目标检测
 *          输出矩形框的中心点坐标(x, y)，矩形框的高宽(h, w)，矩形框的倾斜角的余弦值和正弦值
*   @auth : tyf
*   @date : 2022-05-08  13:59:59
*/
public class yolov5_rotate {

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
        // 阈值
        float conf_thres = 0.5f;
        float iou_thres = 0.5f;
        // 边框置信度过滤后得到
        ArrayList<float[]> datas = new ArrayList<>();
        // 颜色
        Scalar color1 = new Scalar(0, 0, 255);
        Scalar color2 = new Scalar(0, 255, 0);

        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src.clone(),1024,1024);
            this.tensor = this.transferTensor(this.dst.clone(),3,1024,1024); // 转张量
            this.run(); // 执行推理
//            this.nms(); // 执行nms过滤
        }
        // 使用 opencv 读取图片到 mat
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        // YOLOv5的输入是RGB格式的3通道图像，图像的每个像素需要除以255来做归一化，并且数据要按照CHW的顺序进行排布
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
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,channels,netWidth,netHeight});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }

        // 中心点坐标转 xin xmax ymin ymax
        public static float[] xywh2xyxy(float[] bbox,float maxWidth,float maxHeight) {
            // 中心点坐标
            float x = bbox[0];
            float y = bbox[1];
            float w = bbox[2];
            float h = bbox[3];
            // 计算
            float x1 = x - w * 0.5f;
            float y1 = y - h * 0.5f;
            float x2 = x + w * 0.5f;
            float y2 = y + h * 0.5f;
            // 限制在图片区域内
            return new float[]{
                    x1 < 0 ? 0 : x1,
                    y1 < 0 ? 0 : y1,
                    x2 > maxWidth ? maxWidth:x2,
                    y2 > maxHeight? maxHeight:y2};
        }

        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("images", tensor));

                // 定义 anchorGrid 数组
                float[][] anchorGrid = new float[3][];
                anchorGrid[0] = new float[] { 27, 26, 20, 40, 44, 19, 34, 34, 25, 47 };
                anchorGrid[1] = new float[] { 55, 24, 44, 38, 31, 61, 50, 50, 63, 45 };
                anchorGrid[2] = new float[] { 65, 62, 88, 60, 84, 79, 113, 85, 148, 122 };
                // 定义 stride 数组
                float[] stride = new float[]{8, 16, 32};
                // 模型输入的宽高
                int netw = 1024;
                int neth = 1024;

                // 模型输出 [ 107520, 9]
                float[][] out = ((float[][][])(res.get(0)).getValue())[0];

                // 107520 个预选框
                for(int i=0;i< out.length;i++){

                    float[] data = out[i];

                    // 边框置信度
                    float score = data[4];

                    // 置信度阈值
                    if(score>=conf_thres){
                        // 解析预测框在特征图中的坐标
                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];
                        // 3中尺寸的特征图
                        for(int feature_map_index=0;feature_map_index<3;feature_map_index++){

                            // TODO

                            // 计算预测框在原始图片中的左上角和右下角坐标
//                            float xmin = centerX - width / 2;
//                            float ymin = centerY - height / 2;
//                            float xmax = centerX + width / 2;
//                            float ymax = centerY + height / 2;
//
//                            this.datas.add(new float[]{
//                                    xmin,
//                                    ymin,
//                                    xmax,
//                                    ymax
//                            });
                        }
                    }
                }

            }
            catch (Exception e){
                e.printStackTrace();
            }
        }
        // 定义sigmoid函数
        private float sigmoid(float x) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
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

        // 执行nms过滤
        public void nms(){

            ArrayList<float[]> temp = new ArrayList<>();

            // 将 datas 按照置信度从大到小顺序排放
            datas.sort((o1, o2) -> Float.compare(
                    o2[4]*o2[5],
                    o1[4]*o1[5]
            ));

            while (!datas.isEmpty()){

                float[] max = datas.get(0);
                temp.add(max);
                Iterator<float[]> it = datas.iterator();
                while (it.hasNext()) {
                    // 交并比
                    float[] obj = it.next();
                    double iou = calculateIoU(
                            new float[]{max[0],max[1],max[2],max[3]},
                            new float[]{obj[0],obj[1],obj[2],obj[3]}
                    );
                    if (iou > iou_thres) {
                        it.remove();
                    }
                }

            }

            this.datas = temp;

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

        // 弹窗显示
        public void show(){

            // 遍历所有边框
            datas.stream().forEach(n->{

                float x1 = n[0];
                float y1 = n[1];
                float x2 = n[2];
                float y2 = n[3];

                System.out.println("xxx:"+Arrays.toString(n));

                // 画边框
                Imgproc.rectangle(
                        dst,
                        new Point(Float.valueOf(x1).intValue(), Float.valueOf(y1).intValue()),
                        new Point(Float.valueOf(x2).intValue(), Float.valueOf(y2).intValue()),
                        color1,
                        2);
            });

            JFrame frame = new JFrame("Image");
            frame.setSize(dst.width(), dst.height());

            // 图片转为原始大小
            BufferedImage img = mat2BufferedImage(dst);
            img = resize(img,src.width(),src.height());

            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
    }

    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // images -> [1, 3, 1024, 1024] -> FLOAT
        // ---------模型输出-----------
        // output -> [1, 107520, 9] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_rotate\\best.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\yolov5_rotate\\pic.png");

        // 显示
        image.show();

    }


}
