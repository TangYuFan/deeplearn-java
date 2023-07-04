package tool.deeplearning;


import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
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
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;


/**
 *  @Desc: 目标检测 + 实例分割  视频
 *  @Date: 2022-03-24 16:12:52
 *  @auth: TYF
 */
public class yolov5_predict_segment {


    // onnxruntime 环境
    public static OrtEnvironment env;
    public static OrtSession session;

    // 模型的类别信息,从权重读取
    public static JSONObject names;

    // 模型的输入shape,从权重读取
    public static long count;//1 模型每次处理一张图片
    public static long channels;//3 模型通道数
    public static long netHeight;//640 模型高
    public static long netWidth;//640 模型宽

    // 检测框筛选阈值,参考 detect.py 中的设置
    public static float confThreshold = 0.25f;
    public static float nmsThreshold = 0.45f;

    // mask掩膜二值化阈值
    public static float maskThreshold = 0.5f;

    static DecimalFormat decimalFormat = new DecimalFormat("#.##");

    // 目标类 一共25200 个
    public static class Detection{
        float[] data_117;// 117长度的数组,保存类别、xyxy、置信度、mask信息
        float[] srcXYXY;// 保存还原到原始图片中的坐标
        List<int[]> pointSet;// mask 阈值判断后的对象点集合
        public Detection(float[] data_117) {
            this.data_117 = data_117;
        }
        // 返回中心点坐标以及框图宽高
        public float[] getXYWH(){
            return new float[]{
                    data_117[0],
                    data_117[1],
                    data_117[2],
                    data_117[3]};
        }
        // 返回边框坐标
        public float[] getXYXY(){
            return xywh2xyxy(this.getXYWH());
        }
        // 返回置信度
        public float getPercentage(){
            return data_117[4];
        }
        // 返回最有可能的类别
        public String getTypeMaxName(){
            // 5~85 为所有类别(80个)的概率得分,需要找出最大值以及所在索引
            float[] classInfo = Arrays.copyOfRange(this.data_117, 5, 85);
            int maxIndex = getMaxIndex(classInfo);// 概率最高的类被的索引
            String maxClass = (String)names.get(Integer.valueOf(maxIndex));// 概率最高的类别的label
            return maxClass;
        }
        // 返回最有可能的类别的概率
        public float getTypeMaxPercent(){
            // 5~85 为所有类别(80个)的概率得分,需要找出最大值以及所在索引
            float[] classInfo = Arrays.copyOfRange(this.data_117, 5, 85);
            int maxIndex = getMaxIndex(classInfo);// 概率最高的类被的索引
            float maxValue = classInfo[maxIndex];// 概率最高的类被的概率
            return maxValue;
        }
        // 返回32位mask信息
        public float[] getMaskWeight(){
            return Arrays.copyOfRange(this.data_117, 85, 117);
        }
        public void setSrcXYXY(float[] srcXYXY) {
            this.srcXYXY = srcXYXY;
        }
        public float[] getSrcXYXY(){
            return this.srcXYXY;
        }
        public void setPointSet(List<int[]> pointSet) {
            this.pointSet = pointSet;
        }

        public List<int[]> getPointSet() {
            return pointSet;
        }
    }

    // onnxruntime 环境初始化
    static {
        try{

            // 权重导出时目前 com.microsoft.onnxruntime 这个库只支持到 opset<=16 所以在导出模型时需要设置 --opset 16
            String weight = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolo\\yolov5s-seg.onnx";

            env = OrtEnvironment.getEnvironment();
            session = env.createSession(weight, new OrtSession.SessionOptions());

            // 保存一些模型信息 例如输入宽高、类别等
            // 3.打印模型,getCustomMetadata 里面有类别信息、模型输入宽高等
            OnnxModelMetadata metadata = session.getMetadata();
            Map<String, NodeInfo> infoMap = session.getInputInfo();
            TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
            String nameClass = metadata.getCustomMetadata().get("names");
            System.out.println("-------打印模型信息开始--------");
            System.out.println("getProducerName="+metadata.getProducerName());
            System.out.println("getGraphName="+metadata.getGraphName());
            System.out.println("getDescription="+metadata.getDescription());
            System.out.println("getDomain="+metadata.getDomain());
            System.out.println("getVersion="+metadata.getVersion());
            System.out.println("getCustomMetadata="+metadata.getCustomMetadata());
            System.out.println("getInputInfo="+infoMap);
            System.out.println("nodeInfo="+nodeInfo);
            System.out.println("-------打印模型信息结束--------");

            // 4.从里面读取类别信息 {0: 'person', 1: 'bicycle', 2: 'car'}
            names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
            System.out.println("类别信息:"+names);

            // 5.需要从模型信息中读出输入张量的shape,任意图片都需要转换到这个尺寸之后才能输入模型,并且模型输出得到的检测框坐标还需要反变换回去,yolov5是 640*640
            count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
            channels = nodeInfo.getShape()[1];//3 模型通道数
            netHeight = nodeInfo.getShape()[2];//640 模型高
            netWidth = nodeInfo.getShape()[3];//640 模型宽
            System.out.println("模型通道数="+channels+",网络输入高度="+netHeight+",网络输入宽度="+netWidth);

            // opencv 库,将 opencv\build\java\x64\opencv_java455.dll 复制到 Java JDK安装的bin目录下
            // 从 org.openpnp.opencv 的依赖中获取
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }

    }

    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }


    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,添加留白保存宽高比为1
    public static Mat resizeWithPadding(Mat src) {
        Mat dst = new Mat();
        int oldW = src.width();
        int oldH = src.height();
        double r = Math.min((double) netWidth / oldW, (double) netHeight / oldH);
        int newUnpadW = (int) Math.round(oldW * r);
        int newUnpadH = (int) Math.round(oldH * r);
        int dw = (Long.valueOf(netWidth).intValue() - newUnpadW) / 2;
        int dh = (Long.valueOf(netHeight).intValue() - newUnpadH) / 2;
        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);
        Imgproc.resize(src, dst, new Size(newUnpadW, newUnpadH));
        Core.copyMakeBorder(dst, dst, top, bottom, left, right, Core.BORDER_CONSTANT);
        return dst;
    }

    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,不添加留白
    public static Mat resizeWithoutPadding(Mat src){
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(netWidth, netHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }


    // 将图片矩阵转化为 onnxruntime 需要的 tensor
    // 根据yolo的输入张量的预处理,需要进行归一化、BGR -> RGB 等超做 具体可以看 detect.py 脚本
    public static OnnxTensor transferTensor(Mat dst){

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
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{count,channels,netWidth,netHeight});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }

    // YOLOv5的输入是RGB格式的3通道图像，图像的每个像素需要除以255来做归一化，并且数据要按照CHW的顺序进行排布
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


    // 计算两个框的交并比
    private static double calculateIoU(Detection box1, Detection box2) {

        //  getXYXY() 返回 xmin-0 ymin-1 xmax-2 ymax-3

        double x1 = Math.max(box1.getXYXY()[0], box2.getXYXY()[0]);
        double y1 = Math.max(box1.getXYXY()[1], box2.getXYXY()[1]);
        double x2 = Math.min(box1.getXYXY()[2], box2.getXYXY()[2]);
        double y2 = Math.min(box1.getXYXY()[3], box2.getXYXY()[3]);
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1.getXYXY()[2] - box1.getXYXY()[0] + 1) * (box1.getXYXY()[3] - box1.getXYXY()[1] + 1);
        double box2Area = (box2.getXYXY()[2] - box2.getXYXY()[0] + 1) * (box2.getXYXY()[3] - box2.getXYXY()[1] + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }


    // 遍历 25200个框根据置信度初步赛选
    public static List<Detection> filter1(float[][][] outpot){
        // 取 25200 * (85+32) 矩阵
        float[][] data = outpot[0];
        List<Detection> res = new ArrayList<>();
        // 遍历 25200 检测框
        for (float[] bbox : data){
            // 4 个位置表示检测框置信度得分
            float confidence = bbox[4];
            // 首先根据框图置信度粗选
            if(confidence>=confThreshold){
                Detection det = new Detection(bbox);
                res.add(det);
            }
        }
        return res;
    }

    // 将初筛后的检测框去重,nms算法
    public static List<Detection> filter2(List<Detection> input){

        // 先按照类概率进行排序
        input.sort((o1, o2) -> Float.compare(o2.getTypeMaxPercent(),o1.getTypeMaxPercent()));

        List<Detection> res = new ArrayList<>();

        while (!input.isEmpty()){
            Detection maxObj = input.get(0);
            res.add(maxObj);
            Iterator<Detection> it = input.iterator();
            // 计算这个检测框和其他所有检测框的iou,如果超过阈值也就是重叠过大则从原集合中去除
            while (it.hasNext()) {
                Detection obj = it.next();
                double iou = calculateIoU(maxObj, obj);
                if (iou > nmsThreshold) {
                    it.remove();
                }
            }
        }

        return res;
    }

    // 中心点坐标转 xin xmax ymin ymax
    public static float[] xywh2xyxy(float[] bbox) {
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
                x2 > netWidth ? netWidth:x2,
                y2 > netHeight? netHeight:y2};
    }


    // 获取数组中最大值所在的下标,求 80个类别中概率最大的类别
    public static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    // 将网络输出的两个点坐标转换到原始图片的坐标 根据原始宽高和网络输入宽高确定缩放比
    // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
    public static void transferSrc2DstWithPadding(List<Detection> data, int srcw, int srch){
        /*
        具体来说，代码中的 srcw 和 srch 分别表示原始图片的宽度和高度，
        gain 是缩放比例，计算方式为将原始图片缩放到指定大小所需的缩放比例和按原始图片宽高比缩放所需的缩放比例中的最小值。
        padW 和 padH 分别表示在水平和竖直方向上留白的大小，计算方式为将指定大小减去缩放后的图片的大小（即缩放前的大小乘以缩放比例）再除以2。
        这段代码的作用在于计算出留白大小，使得在将原始图片缩放到指定大小后，缩放后的图片与指定大小具有相同的宽高比，同时留白大小也可以用于将缩放后的图片放置在指定大小的画布上的正确位置。
        */
        float gain = Math.min((float) netWidth / srcw, (float) netHeight / srch);
        float padW = (netWidth - srcw * gain) * 0.5f;
        float padH = (netHeight - srch * gain) * 0.5f;

        data.stream().forEach(detection ->{
            // 边框坐标缩放
            float xmin = detection.getXYXY()[0];
            float ymin = detection.getXYXY()[1];
            float xmax = detection.getXYXY()[2];
            float ymax = detection.getXYXY()[3];
            // 缩放过后的坐标
            float xmin_ = Math.max(0, Math.min(srcw - 1, (xmin - padW) / gain));
            float ymin_ = Math.max(0, Math.min(srch - 1, (ymin - padH) / gain));
            float xmax_ = Math.max(0, Math.min(srcw - 1, (xmax - padW) / gain));
            float ymax_ = Math.max(0, Math.min(srch - 1, (ymax - padH) / gain));
            // 保存到目标对象中
            detection.setSrcXYXY(new float[]{xmin_,ymin_,xmax_,ymax_});
        });
    }

    // 将网络输出的两个点坐标转换到原始图片的坐标 根据原始宽高和网络输入宽高确定缩放比
    // xmin, ymin, xmax, ymax -> (xmin_org, ymin_org, xmax_org, ymax_org)
    public static void transferSrc2DstWithOutPadding(List<Detection> data, int srcw, int srch){
        // 没有padding,直接确定宽高缩放比就可以了
        float w_scala = Float.valueOf(netWidth)/srcw;
        float h_scala = Float.valueOf(netHeight)/srch;
        data.stream().forEach(detection ->{
            // 边框坐标缩放
            float xmin = detection.getXYXY()[0];
            float ymin = detection.getXYXY()[1];
            float xmax = detection.getXYXY()[2];
            float ymax = detection.getXYXY()[3];
            // 缩放过后的坐标
            float xmin_ = xmin / w_scala;
            float ymin_ = ymin / h_scala;
            float xmax_ = xmax / w_scala;
            float ymax_ = ymax / h_scala;
            if(xmax_ >= srcw){
                xmax_ = srcw;
            }
            if(ymax_ >= srch){
                ymax_ = srch;
            }
            // 保存到目标对象中
            detection.setSrcXYXY(new float[]{xmin_,ymin_,xmax_,ymax_});
        });

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

    // 将 BufferedImage 缩小到一定比例
    public static BufferedImage scaleImage(BufferedImage originalImage, double scale) {
        int newWidth = (int) (originalImage.getWidth() * scale);
        int newHeight = (int) (originalImage.getHeight() * scale);
        // 创建新的BufferedImage对象
        BufferedImage newImage = new BufferedImage(newWidth, newHeight, originalImage.getType());
        // 绘制原始图像并缩小它
        Graphics2D g = newImage.createGraphics();
        g.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return newImage;
    }

    // 在原始 Mat 上进行标注
    public static void pointBox(Mat dst,List<Detection> box){

        if(box.size()==0){
            return;
        }
        // 类别名称
        box.stream().forEach(detection -> {

            String name = detection.getTypeMaxName();
            float percentage = detection.getTypeMaxPercent();// 概率转两位小数
            String percentString = decimalFormat.format(percentage);

            // 画边框
            float xmin = detection.getSrcXYXY()[0];
            float ymin = detection.getSrcXYXY()[1];
            float xmax = detection.getSrcXYXY()[2];
            float ymax = detection.getSrcXYXY()[3];
            Imgproc.rectangle(
                    dst,
                    new Point(Float.valueOf(xmin).intValue(), Float.valueOf(ymin).intValue()),
                    new Point(Float.valueOf(xmax).intValue(), Float.valueOf(ymax).intValue()),
                    new Scalar(0, 0, 255),
                    1);
            // 画类别和名称
            Imgproc.putText(
                    dst,
                    name+" "+percentString,
                    new Point(Float.valueOf(xmin-1).intValue(), Float.valueOf(ymin-3).intValue()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    new Scalar(0, 0, 255),
                    1);

            // 画mask 每个目标有这些点,将这些点的像素值改变一下就可以了
            List<int[]> pointSet = detection.getPointSet();
            // 遍历点坐标列表并绘制圆形
            for (int[] point : pointSet) {
                int x = point[0];
                int y = point[1];
                // 获取原来的颜色,注意矩阵都是行列表示,xy对应的是列行
                double[] c = dst.get(y,x);
                c[1] = c[1]+80;// 将RGB任意通道修改一下就有透明效果了
                if(c[1]>=255){
                    c[1]=255;
                }
                // 修改颜色后设置回去
                dst.put(y,x,c);// 注意矩阵都是行列表示,xy对应的是列行
            }

        });

        // Mat 转 BufferedImage
        BufferedImage imageDst = mat2BufferedImage(dst);

        // BufferedImage 缩小一定比例
        BufferedImage image = scaleImage(imageDst,1);

        // 弹窗显示
        JFrame frame = new JFrame("Image");
        frame.setSize(image.getWidth(), image.getHeight());
        JLabel label = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }

    public static double[] floatArray2doubleArray(float[] data){
        double[] maskDouble = new double[data.length];
        for (int j = 0; j < data.length; j++) {
            maskDouble[j] = (double) data[j];
        }
        return maskDouble;
    }

    public static double[][] floatArray2doubleArray(float[][] data){
        double[][] maskDouble = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for(int j=0; j<data[0].length;j++){
                maskDouble[i][j] = data[i][j];
            }
        }
        return maskDouble;
    }

    public static float[][] floatArray2floatArray(float[][][] data){
        float[][] flattenedData = new float[data.length][data[0].length * data[0][0].length];
        for (int i = 0; i < data.length; i++) {
            float[][] slice = data[i];
            for (int j = 0; j < slice.length; j++) {
                System.arraycopy(slice[j], 0, flattenedData[i], j * slice[j].length, slice[j].length);
            }
        }
        return flattenedData;
    }

    // 再将 1*25600 转回 160*160
    public static RealMatrix transfer_25600_To_160_160(RealMatrix data){
        RealMatrix res = new Array2DRowRealMatrix(160, 160);
        for (int i = 0; i < 160; i++) {
            for (int j = 0; j < 160; j++) {
                int index = i * 160 + j;
                double value = data.getEntry(0, index);
                res.setEntry(i, j, value);
            }
        }
        return res;
    }

    // 对矩阵中每个元素进行sigmod
    public static RealMatrix getSigmod(RealMatrix data){
        int numRows = data.getRowDimension();
        int numCols = data.getColumnDimension();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double x = data.getEntry(i, j);
                double sigmoid = 1 / (1 + Math.exp(-x));
                data.setEntry(i, j, sigmoid);
            }
        }
        return data;
    }

    // 将掩膜mask矩阵扩展到图片原始尺寸,需要考虑原始图片宽高比不为1的情况防止转换后上下留白
    public static RealMatrix resizeRealMatrix(RealMatrix matrix, int newRows, int newCols) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        RealMatrix resizedMatrix = MatrixUtils.createRealMatrix(newRows, newCols);

        for (int i = 0; i < newRows; i++) {
            for (int j = 0; j < newCols; j++) {
                int origI = (int) Math.floor(i * rows / newRows);
                int origJ = (int) Math.floor(j * cols / newCols);
                double d = matrix.getEntry(origI, origJ);
//                if(d>=maskThreshold){
//                    d = 1;
//                }else{
//                    d = 0;
//                }
                resizedMatrix.setEntry(i, j, d);
            }
        }

        return resizedMatrix;
    }


    // 弹窗显示一个 showMatrix
    public static void showMatrix(RealMatrix matrix){
        // 转换 RealMatrix to BufferedImage
        int numRows = matrix.getRowDimension();
        int numCols = matrix.getColumnDimension();
        BufferedImage image = new BufferedImage(numCols, numRows, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double value = matrix.getEntry(i, j);
                int grayValue = (int) Math.round(value * 255.0);
                grayValue = Math.min(grayValue, 255);
                grayValue = Math.max(grayValue, 0);
                int pixelValue = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(j, i, pixelValue);
            }
        }

        // Create JFrame and JLabel to display the image
        JFrame frame = new JFrame("Matrix Image");
        JLabel label = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(label);
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    // 弹窗显示一个 showMatrix 并画框
    public static void showMatrixWithBox(RealMatrix matrix,float xmin,float ymin,float xmax,float ymax){

        // 转换 RealMatrix to BufferedImage
        int numRows = matrix.getRowDimension();
        int numCols = matrix.getColumnDimension();
        BufferedImage image = new BufferedImage(numCols, numRows, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double value = matrix.getEntry(i, j);
                int grayValue = (int) Math.round(value * 255.0);
                grayValue = Math.min(grayValue, 255);
                grayValue = Math.max(grayValue, 0);
                int pixelValue = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(j, i, pixelValue);
            }
        }

        // image 上画框
        Graphics2D graph = image.createGraphics();
        graph.setStroke(new BasicStroke(3));// 线粗细
        graph.setColor(Color.RED);

        // 画矩形
        graph.drawRect(
                Float.valueOf(xmin).intValue(),
                Float.valueOf(ymin).intValue(),
                Float.valueOf(xmax-xmin).intValue(),
                Float.valueOf(ymax-ymin).intValue());
        // 提交画框
        graph.dispose();

        // 弹窗显示
        JFrame frame = new JFrame("Image Dialog");
        frame.setSize(image.getWidth(), image.getHeight());
        JLabel label = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }

    public static List<int[]> getObjMaskPoint(RealMatrix data, float xmin, float ymin, float xmax, float ymax){

        int _xmin = Float.valueOf(xmin).intValue();
        int _ymin = Float.valueOf(ymin).intValue();
        int _xmax = Float.valueOf(xmax).intValue();
        int _ymax = Float.valueOf(ymax).intValue();

        List<int[]> pointSet = new ArrayList();

        // 遍历 data 中 这块区域,对每个像素点进行二值化,如果1则保存为一个点信息,代表这个像素点是目标
        for(int i=_xmin ; i<_xmax ; i++){
            for(int j=_ymin ; j<_ymax ; j++){
                // getEntry 是获取第几行,第几列
                // x 横坐标对应的列 y是纵坐标对应的行 所以这里是 data.getEntry(j,i) 另外矩阵是从0开始的
                double value = data.getEntry(j,i);
                if(value>=maskThreshold){
                    // 保存这个点的坐标信息
                    int[] point = new int[]{i,j};
                    pointSet.add(point);
                }
            }
        }
        return pointSet;
    }


    // 参考 general.py 中的 process_mask 函数
    public static void generateMaskInfo(List<Detection> detections,float[][][][] proto,int width,int height){

        // 32 * 160 * 160 这个是mask原型 c h w
        float[][][] maskSrc = proto[0];
        // 转为二维矩阵也就是 32 * 25600,也就是 32 行 25600 列,相当于把 160*160展平
        float[][] flattenedData = floatArray2floatArray(maskSrc);
        // 再转为矩阵
        RealMatrix m1 = MatrixUtils.createRealMatrix(floatArray2doubleArray(flattenedData));

        // 每个目标框
        detections.stream().forEach(detection -> {

            // 32 这个是mask 掩膜系数,也就是权重,转为矩阵
            float[] maskWeight = detection.getMaskWeight();
            // 作为一个行向量存储在m1中,也就是 1 行 32 列
            RealMatrix m2 = MatrixUtils.createRowRealMatrix(floatArray2doubleArray(maskWeight));

            // 矩阵乘法 1*32 乘 32*25600 得到 1*25600
            RealMatrix m3 = m2.multiply(m1);

            // 再将 1*25600 转回 160*160 也就是一个缩小的掩膜图
            RealMatrix m4 = transfer_25600_To_160_160(m3);

            // 对每个元素求sigmod限制到0~1
            RealMatrix m5 = getSigmod(m4);

            // 将160*160上采样到图片原始尺寸
            RealMatrix m6 = resizeRealMatrix(m5,height,width);

            // 目标在原始图片上的xyxy
            // showMatrixWithBox(m6,detection.getSrcXYXY()[0],detection.getSrcXYXY()[1],detection.getSrcXYXY()[2],detection.getSrcXYXY()[3]);

            // 根据xyxy,将这一块区域从mask图中截取出来,也就是将这块区域中,大于mask阈值的所有点的坐标拿出来,这一步隐含了二值化
            List<int[]> pointSet = getObjMaskPoint(
                    m6,
                    detection.getSrcXYXY()[0],
                    detection.getSrcXYXY()[1],
                    detection.getSrcXYXY()[2],
                    detection.getSrcXYXY()[3]);

            // 将mask点信息保存到每个目标对象中
            detection.setPointSet(pointSet);

        });

    }

    public static void main(String[] args) throws Exception{

        // 读取图片,保存原始宽高
        String pic = new File("").getCanonicalPath() +"\\model\\deeplearning\\yolo\\car2.png";
        Mat src = readImg(pic);
        int srcw = src.width();
        int srch = src.height();

        // 重写修改为网络输入的宽高,padding指的是留白扩充保持宽高比为1,主要影响的是后续坐标转换
//        Mat dst = resizeWithPadding(src);
        Mat dst = resizeWithoutPadding(src);

        // 输入图片预处理并转为 tensor 根据yolo的输入张量的预处理,需要进行归一化、BGR -> RGB 等超做 具体可以看 detect.py 脚本
        OnnxTensor tensor = transferTensor(dst);

        // 进行推理
        OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));

        // 输出第一个元素和目标检测差不多现状是 1 * 25200 * (85+32) 的矩阵
        // 输出的第二个元素就是proto网络输出形状是 1 * 32 * 160 * 160  需要将前面得到的 1*32 和 32 * 160 * 160 做矩阵惩罚然后转换为轮廓坐标点信息
        OnnxTensor tensor1 = (OnnxTensor)result.get(0);
        float[][][] data1 = (float[][][])tensor1.getValue();
        OnnxTensor tensor2 = (OnnxTensor)result.get(1);
        float[][][][] data2 = (float[][][][])tensor2.getValue();

        // 遍历25200个框,保存目标对象,并根据置信度进行初步筛选
        List<Detection> filter1 = filter1(data1);

        // 遍历所有目标根据nms去掉重叠框
        List<Detection> filter2 = filter2(filter1);

        // 坐标转换到原始图片,注意前面输入模型的图片是否使用padding,会影响坐标转换
//        transferSrc2DstWithPadding(filter2,srcw,srch);
        transferSrc2DstWithOutPadding(filter2,srcw,srch);

        // 计算所有目标的mask
        generateMaskInfo(filter2,data2,srcw,srch);

        // 所有目标画框最后弹窗显示
        pointBox(src,filter2);



    }

}
