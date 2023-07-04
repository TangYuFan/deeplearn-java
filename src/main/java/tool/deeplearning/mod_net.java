package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : 实时人像抠图模型 背景实时替换
*   @auth : tyf
*   @date : 2022-04-28  17:43:07
*/
public class mod_net {

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

    public static void main(String[] args) throws Exception{

        // https://github.com/ZHKKKe/MODNet

        /*
        ---------模型输入-----------
        input -> [-1, 3, -1, -1] -> FLOAT
        ---------模型输出-----------
        output -> [-1, 1, -1, -1] -> FLOAT
         */
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\mod_net\\modnet.onnx");

        // 原始图像
        Mat src = readImg(new File("").getCanonicalPath()+"\\model\\deeplearning\\mod_net\\ren.png");
        BufferedImage src_img_1 = matToBufferedImage(src.clone());

        // 转为 512*512 模型信息不体现输入尺寸,需要看源码确认尺寸以及预处理方法
        Mat dst = resizeWithoutPadding(src,512,512);

        // 原始图像 512*512,作为底图获取人像部分的颜色值
        BufferedImage src_img_2 = matToBufferedImage(dst.clone());

        // 转为张量 同样减去均值然后除标准差
        OnnxTensor tensor = transferTensor(dst.clone(),3,512,512);

        // 推理输出 1 * 1 * 512 * 512
        OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

        float[][] data = ((float[][][][])(res.get(0)).getValue())[0][0];

        // 空图片来显示人像,也可以使用一个背景图片进行背景替换
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);

        for(int i=0;i<512;i++){
            for(int j=0;j<512;j++){
                // 该点是人像取值是0.9xxx 如果不是人像取值是0.000
                float mask = data[i][j];
                // 像素点是人,则设置原来的颜色
                if(mask>0.99){
                    // 从原始图片中获取该点的颜色
                    img.setRGB(j, i,src_img_2.getRGB(j,i));
                }
                // 像素点不是人,则设置为白色
                else{
                    img.setRGB(j, i,
                            new Color(255,255,255).getRGB()
                    );
                }
            }
        }

        // 512*512缩放到原始尺寸
        BufferedImage src_img_3 = scala(img, src_img_1.getWidth(), src_img_1.getHeight());

        // 弹窗显示 原始图片和新的图片
        // 一行两列
        JPanel content = new JPanel(new GridLayout(1,2,5,5));
        // display the image in a window
        ImageIcon icon = new ImageIcon(shrinkByHalf(src_img_1));
        JLabel le1 = new JLabel(icon);
        ImageIcon ico2 = new ImageIcon(shrinkByHalf(src_img_3));
        JLabel le2 = new JLabel(ico2);

        content.add(le1);
        content.add(le2);

        JFrame frame = new JFrame();
        frame.add(content);
        frame.pack();
        frame.setVisible(true);

        // 视频背景替换：
        // 将两段同样尺寸、时长的视频分别遍历每一帧,视频1实时抠图、视频2实时画人图像



    }

}
