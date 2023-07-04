package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : 人像抠图 + 视频背景替换
*   @auth : tyf
*   @date : 2022-04-28  17:43:07
*/
public class mod_net_video {

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

    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }

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

    public static OnnxTensor transferTensor(Mat dst, int channels, int netWidth, int netHeight){
        // BGR -> RGB
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

        //  归一化 0-255 转 0-1
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);

        // 减去均值，除以标准差
        Core.subtract(dst, new Scalar(0.5, 0.5, 0.5), dst);
        Core.divide(dst, new Scalar(0.5, 0.5, 0.5), dst);

        // 初始化一个输入数组 channels * netWidth * netHeight
        float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
        dst.get(0, 0, whc);

        // 得到最终的图片转 float 数组
        float[] chw = whc2cwh(whc);

        // 创建 onnxruntime 需要的 tensor
        // 传入输入的图片 float 数组并指定数组shape
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

    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
    public static Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(netWidth, netHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }

    public static BufferedImage matToBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((java.awt.image.DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }


    public static BufferedImage scala(BufferedImage scaledImage,int newWidth,int newHeight){
        BufferedImage resized = new BufferedImage(newWidth, newHeight, scaledImage.getType());
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(scaledImage, 0, 0, newWidth, newHeight, 0, 0, scaledImage.getWidth(), scaledImage.getHeight(), null);
        g.dispose();
        return resized;
    }

    // 将 BufferedImage 缩小一半
    public static BufferedImage shrinkByHalf(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        BufferedImage shrunkImage = new BufferedImage(w / 2, h / 2, image.getType());
        Graphics2D g = shrunkImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        AffineTransform at = AffineTransform.getScaleInstance(0.5, 0.5);
        AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        scaleOp.filter(image, shrunkImage);
        g.dispose();
        return shrunkImage;
    }


    // 将人和背景进行融合
    public static Mat doMix(Mat m1,Mat m2) throws Exception{

        // 保存原始人像
        Mat src = m1.clone();

        // 从m1抠出人像
        // 转为张量 同样减去均值然后除标准差
        OnnxTensor tensor = transferTensor(m1,3,512,512);

        // 推理输出 1 * 1 * 512 * 512
        OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));
        float[][] data = ((float[][][][])(res.get(0)).getValue())[0][0];

        for(int i=0;i<512;i++){
            for(int j=0;j<512;j++){
                // 该点是人像取值是0.9xxx 如果不是人像取值是0.000
                float mask = data[i][j];
                // 人
                if(mask>0.99){
                    // 获取m1在该点额颜色修改背景m2中对应像素点的颜色
                    m2.put(i,j,src.get(i, j));
                }
            }
        }

        // 返回背景
        return m2;
    }



    public static void main(String[] args) throws Exception{

        // https://github.com/ZHKKKe/MODNet

        /*
        ---------模型输入-----------
        input -> [-1, 3, -1, -1] -> FLOAT
        ---------模型输出-----------
        output -> [-1, 1, -1, -1] -> FLOAT
         */
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\mod_net2\\modnet.onnx");


        // 视频a,用于提取人像
        String video_a = new File("").getCanonicalPath() + "\\model\\deeplearning\\mod_net2\\people.mp4";
        VideoCapture cap_a = new VideoCapture(video_a);

        // 视频b,用于提取背景
        String video_b = new File("").getCanonicalPath() + "\\model\\deeplearning\\mod_net2\\background.mp4";
        VideoCapture cap_b = new VideoCapture(video_b);

        // 缓存两个视频读取的当前帧
        Mat cache1 = new Mat((int) cap_a.get(Videoio.CAP_PROP_FRAME_HEIGHT), (int) cap_a.get(Videoio.CAP_PROP_FRAME_WIDTH), CvType.CV_8UC3);
        Mat cache2 = new Mat((int) cap_b.get(Videoio.CAP_PROP_FRAME_HEIGHT), (int) cap_b.get(Videoio.CAP_PROP_FRAME_WIDTH), CvType.CV_8UC3);

        // 打开两个视频,一帧一帧进行处理
        while ( cap_a.read(cache1) && cap_b.read(cache2) ) {

            // 都转为 512*512
            Mat m1 = resizeWithoutPadding(cache1.clone(),512,512);
            Mat m2 = resizeWithoutPadding(cache2.clone(),512,512);

            // 将人和背景进行融合
            Mat mix = doMix(m1.clone(),m2.clone());

            // 将三个mat显示在同一个窗口中,创建一个大的Mat对象,将3个Mat对象拼接起来
            Mat displayMat = Mat.zeros(512, 512 * 3, CvType.CV_8UC3);

            // 左
            Mat leftSide = displayMat.colRange(0,512);
            m2.copyTo(leftSide);
            // 中
            Mat centerSide = displayMat.colRange(512, 512 * 2);
            m1.copyTo(centerSide);
            // 右
            Mat rightSide = displayMat.colRange(512 * 2, 512 * 3);
            mix.copyTo(rightSide);

            // 显示拼接后的帧
            org.opencv.highgui.HighGui.imshow("Video", displayMat);
            org.opencv.highgui.HighGui.waitKey(1);

        }


    }

}
