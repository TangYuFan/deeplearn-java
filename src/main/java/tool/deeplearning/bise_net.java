package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
*   @desc : 使用BiSeNet做人脸面部解析
*   @auth : tyf
*   @date : 2022-04-28  16:29:21
*/
public class bise_net {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 19种颜色
    public static Color[] colors = new Color[]{
            new Color(0, 0, 255),
            new Color(0, 255, 0),
            new Color(0, 255, 255),
            new Color(255, 0, 0),
            new Color(255, 255, 0),
            new Color(255, 0, 255),
            new Color(128, 128, 128),
            new Color(255, 165, 0),
            new Color(128, 0, 0),
            new Color(255, 192, 203),
            new Color(46, 139, 87),
            new Color(30, 144, 255),
            new Color(255, 20, 147),
            new Color(218, 112, 214),
            new Color(255, 99, 71),
            new Color(255, 215, 0),
            new Color(64, 224, 208),
            new Color(0, 255, 127),
            new Color(147, 112, 219)
    };


    // 环境初始化
    public static void init1(String weight) throws Exception{
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
    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中
    public static Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(netWidth, netHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
    }

    public static OnnxTensor transferTensor(Mat dst, int channels, int netWidth, int netHeight){
        // BGR -> RGB
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);

        //  归一化 0-255 转 0-1
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);
        // 减去均值再除以均方差
        double[] meanValue = {0.485f, 0.456f, 0.406f};
        double[] stdValue = {0.229f, 0.224f, 0.225f};
        Core.subtract(dst, new Scalar(meanValue), dst);
        Core.divide(dst, new Scalar(stdValue), dst);

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

    public static BufferedImage copy(BufferedImage source) {
        BufferedImage destination = new BufferedImage(source.getWidth(), source.getHeight(), source.getType());
        Graphics2D g = destination.createGraphics();
        g.drawImage(source, 0, 0, null);
        g.dispose();
        return destination;
    }


    public static void main(String[] args) throws Exception{

        // https://github.com/hpc203/Face-Parsing-pytorch-opencv-onnxruntime

        /*
            BiSeNet是一个语义分割网络，
            人脸面部解析的本质是对人脸的不同器官做分割或者说像素级分类。
            BiseNet（全称为Bilateral Segmentation Network）是一种用于图像语义分割的神经网络模型。
            它由来自北京大学的研究人员在2018年提出，旨在处理大规模高分辨率图像的语义分割问题，
            如道路、人、车等物体的分割。BiseNet的架构采用了双边滤波（Bilateral Filter）和两个分支（branch）的设计。
            双边滤波是一种图像处理技术，它能够平滑图像并保留图像边缘信息。
            BiseNet中使用双边滤波器对原始图像进行处理，从而得到一个具有多尺度信息的特征图。
            BiseNet的两个分支分别称为上下文分支（Context Branch）和细粒度分支（Detail Branch）。
            上下文分支是一种深度卷积神经网络（DCNN），可以学习全局上下文信息，对于大物体的分割非常有效。
            细粒度分支则采用了轻量级卷积神经网络（LCNN）来学习图像的细节信息，对于小物体和细节分割有很好的效果。
         */

        /*
        ---------模型输入-----------
        input -> [1, 3, 512, 512] -> FLOAT
        ---------模型输出-----------
        out -> [1, 19, 512, 512] -> FLOAT
         */
        init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\bise_net\\my_param.onnx");

        // 原始图片
        Mat src = readImg(new File("").getCanonicalPath()+"\\model\\deeplearning\\bise_net\\face.png");

        // 缩放 512*512
        Mat dst = resizeWithoutPadding(src.clone(),512,512);

        // 预处理 先除255、再减去均值、再除均方差
        OnnxTensor tensor = transferTensor(dst.clone(),3,512,512);

        // 推理
        OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

        // 19 * 512* 512
        float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];

        // 输出的是含义
        // 512*512 就是一个图像
        // 19 代表当前位置的像素点的19个分类的概率,可以直接按照19种颜色,根据概率指定一个颜色进行可视化
        // 也就是完全的像素级别分割


        // 原始图片
        BufferedImage img_src = matToBufferedImage(dst);

        // 19个原始图片备份
        BufferedImage img_src_0 = copy(img_src);
        BufferedImage img_src_1 = copy(img_src);
        BufferedImage img_src_2 = copy(img_src);
        BufferedImage img_src_3 = copy(img_src);
        BufferedImage img_src_4 = copy(img_src);
        BufferedImage img_src_5 = copy(img_src);
        BufferedImage img_src_6 = copy(img_src);
        BufferedImage img_src_7 = copy(img_src);
        BufferedImage img_src_8 = copy(img_src);
        BufferedImage img_src_9 = copy(img_src);
        BufferedImage img_src_10 = copy(img_src);
        BufferedImage img_src_11 = copy(img_src);
        BufferedImage img_src_12 = copy(img_src);
        BufferedImage img_src_13 = copy(img_src);
        BufferedImage img_src_14 = copy(img_src);
        BufferedImage img_src_15 = copy(img_src);
        BufferedImage img_src_16 = copy(img_src);
        BufferedImage img_src_17 = copy(img_src);
        BufferedImage img_src_18 = copy(img_src);

        // 空图片,显示19中颜色
        BufferedImage img = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);


        // 按照像素位置遍历
        for(int i=0;i<512;i++){
            for(int j=0;j<512;j++){

                // 取当前像素位置概率最大值的下标
                float[] gailv = new float[]{
                        data[0][i][j],
                        data[1][i][j],
                        data[2][i][j],
                        data[3][i][j],
                        data[4][i][j],
                        data[5][i][j],
                        data[6][i][j],
                        data[7][i][j],
                        data[8][i][j],
                        data[9][i][j],
                        data[10][i][j],
                        data[11][i][j],
                        data[12][i][j],
                        data[13][i][j],
                        data[14][i][j],
                        data[15][i][j],
                        data[16][i][j],
                        data[17][i][j],
                        data[18][i][j]
                };

                int max = getMaxIndex(gailv);

                // 设置当前位置的rgb颜色
                Color col = colors[max];
                img.setRGB(j, i, col.getRGB());

                // 在备份图片上进行绘制
                if(max==0)
                    img_src_0.setRGB(j, i, col.getRGB());
                else if(max==1)
                    img_src_1.setRGB(j, i, col.getRGB());
                else if(max==2)
                    img_src_2.setRGB(j, i, col.getRGB());
                else if(max==3)
                    img_src_3.setRGB(j, i, col.getRGB());
                else if(max==4)
                    img_src_4.setRGB(j, i, col.getRGB());
                else if(max==5)
                    img_src_5.setRGB(j, i, col.getRGB());
                else if(max==6)
                    img_src_6.setRGB(j, i, col.getRGB());
                else if(max==7)
                    img_src_7.setRGB(j, i, col.getRGB());
                else if(max==8)
                    img_src_8.setRGB(j, i, col.getRGB());
                else if(max==9)
                    img_src_9.setRGB(j, i, col.getRGB());
                else if(max==10)
                    img_src_10.setRGB(j, i, col.getRGB());
                else if(max==11)
                    img_src_11.setRGB(j, i, col.getRGB());
                else if(max==12)
                    img_src_12.setRGB(j, i, col.getRGB());
                else if(max==13)
                    img_src_13.setRGB(j, i, col.getRGB());
                else if(max==14)
                    img_src_14.setRGB(j, i, col.getRGB());
                else if(max==15)
                    img_src_15.setRGB(j, i, col.getRGB());
                else if(max==16)
                    img_src_16.setRGB(j, i, col.getRGB());
                else if(max==17)
                    img_src_17.setRGB(j, i, col.getRGB());
                else if(max==18)
                    img_src_18.setRGB(j, i, col.getRGB());
            }
        }

        // 弹窗显示
        JFrame frame = new JFrame("face");

        // 一共21个图片 3*7
        JPanel parent = new JPanel(new GridLayout(3,7,1,1));

        // 原始图片
        parent.add(new JLabel(new ImageIcon(img_src)));

        // 19分类的图片
        parent.add(new JLabel(new ImageIcon(img)));

        // 剩下单独绘制的图片
        parent.add(new JLabel(new ImageIcon(img_src_0)));
        parent.add(new JLabel(new ImageIcon(img_src_1)));
        parent.add(new JLabel(new ImageIcon(img_src_2)));
        parent.add(new JLabel(new ImageIcon(img_src_3)));
        parent.add(new JLabel(new ImageIcon(img_src_4)));
        parent.add(new JLabel(new ImageIcon(img_src_5)));
        parent.add(new JLabel(new ImageIcon(img_src_6)));
        parent.add(new JLabel(new ImageIcon(img_src_7)));
        parent.add(new JLabel(new ImageIcon(img_src_8)));
        parent.add(new JLabel(new ImageIcon(img_src_9)));
        parent.add(new JLabel(new ImageIcon(img_src_10)));
        parent.add(new JLabel(new ImageIcon(img_src_11)));
        parent.add(new JLabel(new ImageIcon(img_src_12)));
        parent.add(new JLabel(new ImageIcon(img_src_13)));
        parent.add(new JLabel(new ImageIcon(img_src_14)));
        parent.add(new JLabel(new ImageIcon(img_src_15)));
        parent.add(new JLabel(new ImageIcon(img_src_16)));
        parent.add(new JLabel(new ImageIcon(img_src_17)));
        parent.add(new JLabel(new ImageIcon(img_src_18)));


        frame.getContentPane().add(parent);
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


    }
}
