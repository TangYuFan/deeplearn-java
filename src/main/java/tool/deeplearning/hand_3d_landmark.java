package tool.deeplearning;

import ai.onnxruntime.*;
import javafx.animation.Animation;
import javafx.animation.RotateTransition;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.geometry.Point3D;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
import javafx.scene.shape.Cylinder;
import javafx.scene.shape.Sphere;
import javafx.scene.transform.Rotate;
import javafx.util.Duration;
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
import java.util.concurrent.atomic.AtomicBoolean;

/**
*   @desc : 手关键点检测 3d 注意模型需要结合手掌检测来完成,这里直接进行关键点检测所以要求直接输入手掌图片
*   @auth : tyf
*   @date : 2022-05-12  10:14:34
*/
public class hand_3d_landmark {


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

    public static class Hand{
        float score;// 置信度
        float lr;//左右
        float[] points;//21个点xyz坐标
        public Hand(float score, float lr, float[] points) {
            this.score = score;
            this.lr = lr;
            this.points = points;
        }
    }

    public static class ImageObj{
        // 原始图片(原始尺寸)
        Mat src;
        // 原始图片(模型尺寸的)
        Mat dst;
        // 模型输出的图片
        public Mat out;
        // 输入张量
        OnnxTensor tensor;
        // 保存所有手的信息
        ArrayList<Hand> hands = new ArrayList<>();
        // 关节连线
        static int[][] connect = new int[][]{

                // 手指1
                new int[]{0,1},
                new int[]{1,2},
                new int[]{2,3},
                new int[]{3,4},
                // 手指2
                new int[]{5,6},
                new int[]{6,7},
                new int[]{7,8},
                // 手指3
                new int[]{9,10},
                new int[]{10,11},
                new int[]{11,12},
                // 手指4
                new int[]{13,14},
                new int[]{14,15},
                new int[]{15,16},
                // 手指5
                new int[]{17,18},
                new int[]{18,19},
                new int[]{19,20},
                // 手掌
                new int[]{2,5},
                new int[]{5,9},
                new int[]{9,13},
                new int[]{13,17},
                new int[]{0,5},
                new int[]{0,9},
                new int[]{0,13},
                new int[]{0,17},

        };
        // 颜色
        Scalar red = new Scalar(0, 0, 255);
        Scalar green = new Scalar(0, 255, 0);
        public ImageObj(String image) throws Exception{
            this.src = this.readImg(image);
            this.dst = this.resizeWithPadding(this.src,224,224);
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

        public Mat resizeWithPadding(Mat src, int netWidth, int netHeight) {
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

        public OnnxTensor transferTensor(Mat dst) throws Exception{

            // BGR -> RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
            dst.convertTo(dst, CvType.CV_32FC1);

            // 数组
            float[] whc = new float[ 3 * 224 * 224 ];
            dst.get(0, 0, whc);

            // 将图片维度从 HWC 转换为 CHW
            float[] chw = new float[whc.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < whc.length; i += 3) {
                    chw[j] = whc[i];
                    // 除 255 [0, 1]
                    chw[j] = chw[j] / 255f;
                    j++;
                }
            }
            // 张量 C=3 H=224 W=224
            // input -> [1, 3, 224, 224] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,3,224,224});
            return tensor;
        }
        // 执行推理
        public void run(){
            try {
                OrtSession.Result res = session.run(Collections.singletonMap("input", tensor));

                // xyz_x21 -> [-1, 63] -> FLOAT  21点,三位坐标
                float[][] xyz_x21 = ((float[][])(res.get(0)).getValue());

                // hand_score -> [-1, 1] -> FLOAT 置信度,手的置信度
                float[][] hand_score = ((float[][])(res.get(1)).getValue());

                // lefthand_0_or_righthand_1 -> [-1, 1] -> FLOAT 左手0右手1
                float[][] lefthand_0_or_righthand_1 = ((float[][])(res.get(2)).getValue());


                // 手的个数
                int count = xyz_x21.length;

                for(int i=0;i<count;i++){
                    // 手的xyz信息
                    float[] xyz = xyz_x21[i];
                    // 手的置信度
                    float score = hand_score[i][0];
                    // 左右
                    float lr = lefthand_0_or_righthand_1[i][0];
//                    System.out.println("hand置信度:"+score+",左右:"+lr+",21关键点xyz:"+Arrays.toString(xyz));
                    if(score>=0.5){
                        // 保存手
                        hands.add(new Hand(
                                score,
                                lr,
                                xyz));
                    }
                }

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

        // 判断是否应该连接线
        public boolean shouldConnect(int i, int j){
            AtomicBoolean res = new AtomicBoolean(false);
            // 特定点进行连接
            Arrays.stream(connect).forEach(n->{
                int index1 = n[0];
                int index2 = n[1];
                if(index1==i&&index2==j){
                    res.set(true);
                }
            });
            return res.get();
        }

        // 创建一个Jpanel来三维展示一个手,传入旋转角度

        // 在3d图片上进行画线
        public JPanel draw3d(){

            // 返回一个 panel 旋转显示手势
            JFXPanel fxPanel = new JFXPanel();
            fxPanel.setPreferredSize(new Dimension(224, 224));

            // 创建Swing的面板用于包含JavaFX的Panel
            JPanel panel = new JPanel(new BorderLayout());
            panel.setPreferredSize(new Dimension(224, 224));
            panel.add(fxPanel, BorderLayout.CENTER);

            // 创建JavaFX的场景和根节点
            Group root = new Group();
            Scene scene = new Scene(root);
            // 将JavaFX的场景设置到JavaFX的Panel中
            fxPanel.setScene(scene);

            // 在Swing的Frame可见之后，初始化JavaFX
            Platform.runLater(() -> {


                // 遍历每个手
                hands.stream().forEach(hand -> {

                    float[] xyz = hand.points;
                    float score = hand.score;
                    float lr = hand.lr;

                    // 保存21个点
                    ArrayList<float[]> hand_point_224_224_224 = new ArrayList<>();

                    // xyz坐标
                    for(int j=1;j<=21;j++){
                        float x = xyz[(j-1)*3];
                        float y = xyz[(j-1)*3+1];
                        float z = xyz[(j-1)*3+2];
                        // 保存这些点
                        hand_point_224_224_224.add(new float[]{x,y,z});
                    }


                    // 画点
                    int m=0;
                    for (float[] point : hand_point_224_224_224) {
                        Sphere sphere = new Sphere(3); // 创建球体作为网格点
                        sphere.setTranslateX(point[0]); // 设置x坐标
                        sphere.setTranslateY(point[1]); // 设置y坐标
                        sphere.setTranslateZ(point[2]); // 设置z坐标
                        PhongMaterial material = new PhongMaterial();
                        // 手指为绿红色,其他为绿色
                        if(m==4||m==8||m==12||m==16||m==20){
                            material.setDiffuseColor(javafx.scene.paint.Color.RED); // 设置球体的漫反射颜色为红色
                        }
                        else {
                            material.setDiffuseColor(Color.GREEN); // 设置球体的漫反射颜色为红色
                        }
                        sphere.setMaterial(material); // 应用材质到球体
                        root.getChildren().add(sphere); // 将球体添加到根节点
                        m++;
                    }


                    // 链接两个点
                    for(int i=0;i<21;i++){
                        for(int j=0;j<21;j++){
                            float[] point1 = hand_point_224_224_224.get(i);
                            float[] point2 = hand_point_224_224_224.get(j);
                            // 关键点链接
                            if(shouldConnect(i,j)){
                                // javafx中没有3d的线,只能画圆柱体,或者通过点插值的方式进行,这里根据距离创建一些点进行绘制
                                Point3D p1 = new Point3D(point1[0], point1[1], point1[2]);
                                Point3D p2 = new Point3D(point2[0], point2[1], point2[2]);
                                // 计算两个点的直线距离
                                double length = p1.distance(p2);
                                int numPoints = (int) length; // 将距离转换为整数，作为要插入的点的数量
                                for (int k = 1; k <= numPoints; k++) {
                                    double t = k / (double) (numPoints + 1);
                                    double x = (1 - t) * p1.getX() + t * p2.getX();
                                    double y = (1 - t) * p1.getY() + t * p2.getY();
                                    double z = (1 - t) * p1.getZ() + t * p2.getZ();
                                    // 在这里使用插入的点坐标(x, y, z)进行绘制操
                                    Sphere sphere = new Sphere(1); // 创建球体作为网格点
                                    sphere.setTranslateX(x); // 设置x坐标
                                    sphere.setTranslateY(y); // 设置y坐标
                                    sphere.setTranslateZ(z); // 设置z坐标
                                    PhongMaterial material = new PhongMaterial();
                                    material.setDiffuseColor(Color.GREEN);
                                    sphere.setMaterial(material); // 应用材质到球体
                                    root.getChildren().add(sphere); // 将球体添加到根节点
                                }
                            }
                        }
                    }

                });


                // 创建旋转动画
                RotateTransition rotateTransition = new RotateTransition(Duration.seconds(10), root);
                rotateTransition.setAxis(Rotate.Y_AXIS);
                rotateTransition.setByAngle(360);
                rotateTransition.setCycleCount(Animation.INDEFINITE);
                rotateTransition.setAutoReverse(false);
                rotateTransition.play();

                // 将JavaFX的场景设置到JavaFX的Panel中
                fxPanel.setScene(scene);

            });

            return panel;

        }

        // 在原始图片上进行画线
        public JPanel draw2d(){

            Mat background = dst.clone();

            // 所有手
            for(int i=0;i<hands.size();i++){

                Hand hand = hands.get(i);

                float[] xyz = hand.points;
                float score = hand.score;
                float lr = hand.lr;

                // xyz坐标
                for(int j=1;j<=21;j++){
                    float x = xyz[(j-1)*3];
                    float y = xyz[(j-1)*3+1];
                    float z = xyz[(j-1)*3+2];
                    // 画点
                    Imgproc.circle(background, new Point(x, y), 2, red,2);
                }

                // 画线
                for(int a=0;a<21;a++){
                    for(int b=0;b<21;b++){
                        if(shouldConnect(a,b)){
                            float x1 = xyz[(a)*3];
                            float y1 = xyz[(a)*3+1];
                            float z1 = xyz[(a)*3+2];
                            float x2 = xyz[(b)*3];
                            float y2 = xyz[(b)*3+1];
                            float z2 = xyz[(b)*3+2];
                            // 画线
                            Imgproc.line(background, new Point(x1,y1), new Point(x2,y2), green, 2);
                        }
                    }
                }

            }

            JLabel label = new JLabel(new ImageIcon(mat2BufferedImage(background)));
            JPanel res = new JPanel();
            res.add(label);
            return res;
        }

        // 原始图片
        public JPanel getSrcImg(){
            JLabel label = new JLabel(new ImageIcon(mat2BufferedImage(dst)));
            JPanel res = new JPanel();
            res.add(label);
            return res;
        }

        // 弹窗显示
        public void show(){

            // 弹窗显示
            JFrame frame = new JFrame("Hand");
            frame.setSize(dst.width(), dst.height());

            JPanel all = new JPanel();
            all.add(getSrcImg()); // 原始图片
            all.add(draw2d()); // 2d标注
            all.add(draw3d()); // 3d标注

            frame.getContentPane().add(all);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.pack();

        }
    }



    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // input -> [-1, 3, 224, 224] -> FLOAT
        // ---------模型输出-----------
        // xyz_x21 -> [-1, 63] -> FLOAT
        // hand_score -> [-1, 1] -> FLOAT
        // lefthand_0_or_righthand_1 -> [-1, 1] -> FLOAT
        init(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\hand_3d_landmark\\hand_landmark_sparse_Nx3x224x224.onnx");


        // 加载图片
        ImageObj image = new ImageObj(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\hand_3d_landmark\\hand.png");

        // 显示
        image.show();


    }
}
