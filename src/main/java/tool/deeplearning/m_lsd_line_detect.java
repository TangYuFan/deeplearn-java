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

/**
*   @desc : M-LSD直线检测
 *          paper是顶会AAAI2022里的一篇文章《Towards Light-weight and Real-time Line Segment Detection》
 *          githup:https://github.com/hpc203/M-LSD-onnxrun-cpp-py
 *
*   @auth : tyf
*   @date : 2022-05-08  16:33:48
*/
public class m_lsd_line_detect {


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
        // 保存的线段
        ArrayList<float[]> line = new ArrayList<>();
        // 阈值1
        float score_thres = 0.5f;// 点置信度阈值
        // 阈值2
        float dist_thres = 20f;// 线段长度阈值
        Scalar color1 = new Scalar(0, 255, 0);
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst = this.resizeWithoutPadding(this.src,512,512);
            // input_image_with_alpha:0 -> [1, 512, 512, 4] -> FLOAT
            this.tensor = this.transferTensor(this.dst.clone(),4,512,512); // 转张量
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
        public OnnxTensor transferTensor(Mat dst,int channels,int netWidth,int netHeight){

            //  def prepare_input(self, image):
            //        resized_image = cv2.resize(image, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
            //        input_image = np.concatenate([resized_image, np.ones([self.input_height, self.input_width, 1])], axis=-1)
            //        input_image = np.expand_dims(input_image, axis=0).astype('float32')
            //        return input_image

            // 输入尺寸
            // input_image_with_alpha:0 -> [1, 512, 512, 4] -> FLOAT

            // cv2也是使用的opencv读取,也就是BGR顺序
            dst.convertTo(dst, CvType.CV_32FC3);

            // 创建一个 全为1的Mat对象。这里的数据类型使用CV_32F，与Python中的np.ones操作相匹配。
            Mat onesMat = Mat.ones(netHeight, netWidth, CvType.CV_32F);

            ArrayList<Mat> all = new ArrayList<>();
            all.add(dst);
            all.add(onesMat);

            // 创建一个大小为netHeight x netWidth的4通道Mat对象。这里的数据类型使用
            // 合并前面的图片和全1矩阵到这个新矩阵中
            Mat mergedImageMat = new Mat(netHeight, netWidth, CvType.CV_32FC4);
            Core.merge(all, mergedImageMat);

            // 展平
            float[] inputArray = new float[(int) mergedImageMat.total() * mergedImageMat.channels()];
            mergedImageMat.get(0, 0, inputArray);

            // python 代码也是使用cv2也就是opencv读取
            // 如果python代码中没有进行通道转换那么这里也不需要


            OnnxTensor tensor = null;
            try {
                tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputArray), new long[]{1,netHeight,netWidth,channels});
            }
            catch (Exception e){
                e.printStackTrace();
                System.exit(0);
            }
            return tensor;
        }
        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("input_image_with_alpha:0", tensor));

                // Identity -> [1, 200, 2] -> INT32
                // Identity_1 -> [1, 200] -> FLOAT
                // Identity_2 -> [1, 256, 256, 4] -> FLOAT

                // pts -> [ 200, 2] -> INT32
                int[][] pts  = ((int[][][])(res.get(0)).getValue())[0];

                // pts_score -> [ 200] -> FLOAT
                float[] pts_score  = ((float[][])(res.get(1)).getValue())[0];

                // vmap  -> [ 256, 256, 4] -> FLOAT
                float[][][] vmap  = ((float[][][][])(res.get(2)).getValue())[0];


                for (int i = 0; i < pts.length; i++) {
                    //
                    int[] center = pts[i];
                    int y = center[0];
                    int x = center[1];
                    float score = pts_score[i];

                    float[] disp = vmap[y][x];
                    float disp_x_start = disp[0];
                    float disp_y_start = disp[1];
                    float disp_x_end = disp[2];
                    float disp_y_end = disp[3];

                    float x_start = x + disp_x_start;
                    float y_start = y + disp_y_start;
                    float x_end = x + disp_x_end;
                    float y_end = y + disp_y_end;

                    if (score > score_thres && getDistance(x_start, y_start, x_end, y_end) > dist_thres) {
                        // 解析上面的输出保存线段的起始点 x1y1x2y2
                        // vmap 是256*256 的也就是相对模型输入只有一半,坐标需要乘2
                        line.add(new float[]{
                                x_start * 2,
                                y_start * 2,
                                x_end * 2,
                                y_end * 2
                        });
                    }
                }

            }
            catch (Exception e){
                e.printStackTrace();
            }
        }

        // 计算两点之间的距离
        private float getDistance(float x1, float y1, float x2, float y2) {
            float dx = x2 - x1;
            float dy = y2 - y1;
            return (float) Math.sqrt(dx * dx + dy * dy);
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


            // 画线
            this.line.stream().forEach(n->{

                float x1 = n[0];
                float y1 = n[1];
                float x2 = n[2];
                float y2 = n[3];

                Point startPoint = new Point(x1, y1);
                Point endPoint = new Point(x2, y2);

                Imgproc.line(dst, startPoint, endPoint, color1, 2);

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
        // input_image_with_alpha:0 -> [1, 512, 512, 4] -> FLOAT
        // ---------模型输出-----------
        // Identity -> [1, 200, 2] -> INT32
        // Identity_1 -> [1, 200] -> FLOAT
        // Identity_2 -> [1, 256, 256, 4] -> FLOAT

        init(new File("").getCanonicalPath()+"\\model\\deeplearning\\m_lsd_line_detect\\model_512x512_large.onnx");

        // 加载图片
        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\m_lsd_line_detect\\test1.jpg";
        ImageObj image = new ImageObj(pic);

        // 显示
        image.show();

    }
}
