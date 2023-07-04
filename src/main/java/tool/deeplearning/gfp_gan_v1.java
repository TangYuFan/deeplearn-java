package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

import static org.opencv.core.CvType.CV_32FC3;

/**
*   @desc : GFP_GAN_V1 人脸照片修复,增加清晰度
 *
 *
*   @auth : tyf
*   @date : 2022-05-11  15:29:59
*/
public class gfp_gan_v1 {

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
        // 模型输出的图片
        public Mat out;
        public BufferedImage out_img;
        // 输入张量
        OnnxTensor tensor;
        public ImageObj(String image) throws Exception{
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,512,512);
            this.tensor = this.transferTensor(this.dst.clone()); // 转张量
            this.run(); // 执行推理
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
        public OnnxTensor transferTensor(Mat dst) throws Exception{

            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            dst.convertTo(dst, CvType.CV_32FC1);

            // 数组
            float[] whc = new float[ 3 * 512 * 512 ];
            dst.get(0, 0, whc);

            // 将图片维度从 HWC 转换为 CHW
            float[] chw = new float[whc.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < whc.length; i += 3) {
                    chw[j] = whc[i];
                    // 除 255 [0, 1]
                    chw[j] = chw[j] / 255f;
                    // 减 0.5 [-0.5, 0.5]
                    chw[j] = chw[j] - 0.5f;
                    // 乘 2 [-1, 1]
                    chw[j] = chw[j] * 2;
                    j++;
                }
            }
            // 张量 C=3 H=512 W=512
            // input -> [1, 3, 512, 512] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,3,512,512});
            return tensor;
        }
        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));
                // 解析 output -> [3, 512, 512] -> FLOAT
                float[][][] data = ((float[][][][])(res.get(0)).getValue())[0];

                // 处理每个元素
                for(int channel=0;channel<3;channel++){
                    for(int i=0;i<512;i++){
                        for(int j=0;j<512;j++){
                            // -1 ~ 1
                            if(data[channel][i][j]>1){
                                data[channel][i][j] = 1;
                            }
                            if(data[channel][i][j]<-1){
                                data[channel][i][j] = -1;
                            }
                            // 0 ~ 1
                            data[channel][i][j] = data[channel][i][j] + 1;
                            // 0 ~ 255
                            data[channel][i][j] = data[channel][i][j] * 255;
                        }
                    }
                }

                // 全新的图片
                Mat mat = new Mat(512, 512, CV_32FC3);
                for (int y = 0; y < 512; y++) {
                    for (int x = 0; x < 512; x++) {
                        float r = data[0][x][y];
                        float g = data[1][x][y];
                        float b = data[2][x][y];
                        mat.put(x, y, b, g, r);
                    }
                }

                this.out = mat;

                // 缩放到原始尺寸并保存
                this.out_img = mat2BufferedImage(this.out);
                // 缩放到原始尺寸并保存
                this.out_img = resize(out_img,src.width(),src.height());
            }
            catch (Exception e){
                e.printStackTrace();
            }
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

        // 弹窗显示
        public void show(){

            // 一行两列
            JPanel content = new JPanel(new GridLayout(1,2,5,5));

            ImageIcon icon = new ImageIcon(mat2BufferedImage(this.dst));
            JLabel le1 = new JLabel(icon);

            ImageIcon ico2 = new ImageIcon(mat2BufferedImage(this.out));
            JLabel le2 = new JLabel(ico2);

            content.add(le1);
            content.add(le2);

            JFrame frame = new JFrame();
            frame.add(content);
            frame.pack();
            frame.setVisible(true);

        }
    }



    public static void main(String[] args) throws Exception{



        // ---------模型输入-----------
        // input -> [1, 3, 512, 512] -> FLOAT
        // ---------模型输出-----------
        // 1288 -> [1, 3, 512, 512] -> FLOAT
        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\gfp_gan_v1\\GFPGANv1.3.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\face.JPG");

        // 显示
        image.show();

    }
}
