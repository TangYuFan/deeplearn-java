package tool.deeplearning;


import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import java.io.File;
import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.util.List;
import java.util.*;


/**
 *  @Desc: 目标检测 + 实例分割  视频
 *  @Date: 2022-03-24 16:12:52
 *  @auth: TYF
 */
public class yolov5_predict_video {


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

    static DecimalFormat decimalFormat = new DecimalFormat("#.##");

    // 目标类 一共25200 个
    public static class Detection{
        float[] data_117;// 117长度的数组,保存类别、xyxy、置信度、mask信息
        float[] srcXYXY;// 保存还原到原始图片中的坐标
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
    }

    // onnxruntime 环境初始化
    static {
        try{

            // 权重导出时目前 com.microsoft.onnxruntime 这个库只支持到 opset<=16 所以在导出模型时需要设置 --opset 16
            String weight = new File("").getCanonicalPath() + "\\model\\yolo\\yolov5s-seg.onnx";

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


    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
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
    public static void transferSrc2Dst(List<Detection> data,int srcw,int srch){
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
                    2);
            // 画类别和名称
//            Imgproc.putText(
//                    dst,
//                    name+" "+percentString,
//                    new Point(Float.valueOf(xmin-1).intValue(), Float.valueOf(ymin-3).intValue()),
//                    Imgproc.FONT_HERSHEY_SIMPLEX,
//                    1,
//                    new Scalar(0, 255, 0),
//                    1);
            // 画轮廓

        });

    }


    public static void generateMaskInfo(List<Detection> detections,float[][][][] proto){

        // 32 * 160 * 160 这个是mask原型
        // [mask_dim, mask_h, mask_w]
        float[][][] maskSrc = proto[0];

        detections.stream().forEach(detection -> {

            // 32 这个是mask 掩膜系数
            float[] maskWeight = detection.getMaskWeight();

            // 矩阵乘法得到图像的掩码信息,计算之后设置到 detectron 中
            //

        });

    }



    // 输入Mat进行目标检测、标注 返回耗时毫秒
    public static long runRec(Mat src,int srcw,int srch) throws Exception{

        Long t1 = System.currentTimeMillis();

        // 重写修改为网络输入的宽高
        Mat dst = resizeWithPadding(src);

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

        // 计算所有目标的mask
        generateMaskInfo(filter2,data2);

        // 坐标转换到原始图片
        transferSrc2Dst(filter2,srcw,srch);

        // 所有目标画框、画轮廓最后弹窗显示
        pointBox(src,filter2);

        Long t2 = System.currentTimeMillis();


        return t2-t1;

    }


    public static void main(String[] args) throws Exception{

        // 视频、rtsp流等
        String video = new File("").getCanonicalPath() + "\\model\\yolo\\1.mp4";

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
                // 推理、标注
                long timeuse = runRec(frame,width,height);
                // 显示处理后的图像
                org.opencv.highgui.HighGui.imshow("Video", frame);

                // 如果fps=25的话每一帧间隔 1000/25 = 40 毫秒 如果推理标注花费时间超过40毫秒则直接处理下一帧,否则等到40-timeuse 毫秒
                if(timeuse>=40){
                    org.opencv.highgui.HighGui.waitKey(1);
                }else {
                    org.opencv.highgui.HighGui.waitKey(Long.valueOf(40-timeuse).intValue());
                }

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        // 推流 TOOD


        Runtime.getRuntime().addShutdownHook(new Thread(()->{
            // 释放资源
            System.out.println("程序退出!");
            cap.release();
        }));


    }

}
