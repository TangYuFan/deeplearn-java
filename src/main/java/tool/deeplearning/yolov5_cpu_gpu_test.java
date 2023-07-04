package tool.deeplearning;

import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

/**
 *  @Desc: 分别使用cpu和gpu进行推理速度比较 目标检测
 *  @Date: 2022-03-22 16:18:29
 *  @auth: TYF
 */
public class yolov5_cpu_gpu_test {


    public static void cpu(String weight,String pic) throws Exception{

        // onnxruntime 环境
        OrtEnvironment env;
        OrtSession session;

        // 模型的类别信息,从权重读取
        JSONObject names;
        // 模型的输入shape,从权重读取
        long count;//1 模型每次处理一张图片
        long channels;//3 模型通道数
        long netHeight;//640 模型高
        long netWidth;//640 模型宽
        // 检测框筛选阈值,参考 detect.py 中的设置
        float confThreshold = 0.25f;
        float nmsThreshold = 0.45f;

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());
        Map<String, NodeInfo> infoMap = session.getInputInfo();
        OnnxModelMetadata metadata = session.getMetadata();
        TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
        String nameClass = metadata.getCustomMetadata().get("names");
        names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
        count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
        channels = nodeInfo.getShape()[1];//3 模型通道数
        netHeight = nodeInfo.getShape()[2];//640 模型高
        netWidth = nodeInfo.getShape()[3];//640 模型宽
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = readImg(pic);// 读取图片
        Mat dst = resizeWithPadding(src,netWidth,netHeight);// 重写修改为网络输入的宽高
        OnnxTensor tensor = transferTensor(dst,env,count,channels,netHeight,netWidth);// 输入图片预处理并转为 tensor

        Long t1 = System.currentTimeMillis();


        // 进行100次推理
        float n = 500l;
        for(int i=1;i<=n;i++){
            OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));
        }

        Long t2 = System.currentTimeMillis();

        float total = (t2-t1)/1000;// 秒
        float fps = (n/total);

        System.out.println("CPU推理总次数:"+n+",总耗时:"+total+"秒"+",fps:"+fps);

    }

    public static void gpu(String weight,String pic) throws Exception{


        // onnxruntime 环境
        OrtEnvironment env;
        OrtSession session;

        // 模型的类别信息,从权重读取
        JSONObject names;
        // 模型的输入shape,从权重读取
        long count;//1 模型每次处理一张图片
        long channels;//3 模型通道数
        long netHeight;//640 模型高
        long netWidth;//640 模型宽
        // 检测框筛选阈值,参考 detect.py 中的设置
        float confThreshold = 0.25f;
        float nmsThreshold = 0.45f;

        env = OrtEnvironment.getEnvironment();

        // 设置gpu deviceId=0
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addCUDA(0);
        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

        session = env.createSession(weight, options);
        Map<String, NodeInfo> infoMap = session.getInputInfo();
        OnnxModelMetadata metadata = session.getMetadata();
        TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
        String nameClass = metadata.getCustomMetadata().get("names");
        names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
        count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
        channels = nodeInfo.getShape()[1];//3 模型通道数
        netHeight = nodeInfo.getShape()[2];//640 模型高
        netWidth = nodeInfo.getShape()[3];//640 模型宽
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src = readImg(pic);// 读取图片
        Mat dst = resizeWithPadding(src,netWidth,netHeight);// 重写修改为网络输入的宽高
        OnnxTensor tensor = transferTensor(dst,env,count,channels,netHeight,netWidth);// 输入图片预处理并转为 tensor

        Long t1 = System.currentTimeMillis();


        // 进行100次推理
        float n = 500l;
        for(int i=1;i<=n;i++){
            OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));
        }

        Long t2 = System.currentTimeMillis();

        float total = (t2-t1)/1000;// 秒
        float fps = (n/total);

        System.out.println("GPU推理总次数:"+n+",总耗时:"+total+"秒"+",fps:"+fps);
    }



    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }

    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
    public static Mat resizeWithPadding(Mat src,long netWidth,long netHeight) {
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
    public static OnnxTensor transferTensor(Mat dst,OrtEnvironment env,long count,long channels,long netHeight,long netWidth){

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


    public static void main(String[] args) throws Exception{

        // 权重
        String weight = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolov5\\yolov5s.onnx";
        // 图片
        String pic = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolov5\\muzo.png";


        // 实际上有两个依赖,前者只能cpu推理,后者可以使用cpu或gpu推理
        // <dependency>
        //			<groupId>com.microsoft.onnxruntime</groupId>
        //			<artifactId>onnxruntime_gpu</artifactId>
        //			<version>1.11.0</version>
        //		</dependency>
        // <dependency>
        //			<groupId>com.microsoft.onnxruntime</groupId>
        //			<artifactId>onnxruntime_gpu</artifactId>
        //			<version>1.11.0</version>
        //		</dependency>


        // 使用cpu测试
        cpu(weight,pic);

        // 使用gpu测试
        gpu(weight,pic);


    }





}
