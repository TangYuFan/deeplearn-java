package tool.deeplearning;


import ai.onnxruntime.*;
import javafx.animation.Animation;
import javafx.animation.RotateTransition;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.paint.Color;
import javafx.scene.paint.PhongMaterial;
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

/**
 *   @desc : 人脸检测 + 对齐（密集点的3d重建也就是mesh重建）
 *
 *         对应的paper是ECCV2020里的一篇文章《Towards Fast, Accurate and Stable 3D Dense Face Alignment》

 *         三个onnx：
 *         RFB-320_240x320_post.onnx    人脸检测,也就是第一步
 *         dense_face_Nx3x120x120.onnx  密集点(dense)重建,也就是3d重建
 *         sparse_face_Nx3x120x120.onnx  稀疏点(sparse)重建,也就是pose重建
 *
 *
 *   @auth : tyf
 *   @date : 2022-05-10  17:06:04
 */
public class face_alignment_mesh_pose {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;


    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 模型2
    public static OrtEnvironment env3;
    public static OrtSession session3;


    // 环境初始化
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }


    // 环境初始化
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }

    public static class ImageObj{

        // 原始图片(原始尺寸)
        Mat src;

        // 截取的脸部
        Mat face;

        // 截图的脸部以及xy描点
        Mat face_point;

        // 保存人脸所有三维点,320*320*320的空间中
        ArrayList<float[]> face_point_320_320_320 = new ArrayList<>();

        Scalar color = new Scalar(0, 0, 255);

        public ImageObj(String image) {
            this.src = this.readImg(image);
        }

        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }

        public static Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(inputWidth, inputHeight);
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

        public static float[] transferPointWithPadding(float[] xy,int imgWidth,int imgHeight,int netWidth,int netHeight){
            float gain = Math.min((float) netWidth / imgWidth, (float) netHeight / imgHeight);
            float padW = (netWidth - imgWidth * gain) * 0.5f;
            float padH = (netHeight - imgHeight * gain) * 0.5f;
            float xmin = xy[0];
            float ymin = xy[1];
            // 缩放过后的坐标
            float xmin_ = Math.max(0, Math.min(imgWidth - 1, (xmin - padW) / gain));
            float ymin_ = Math.max(0, Math.min(imgHeight - 1, (ymin - padH) / gain));
            return new float[]{xmin_,ymin_};
        }

        // 人脸检测
        public void doFaceDetection() throws Exception{

            // ---------模型1输入-----------
            // input -> [1, 3, 240, 320] -> FLOAT 模型输入是 C=3 H=240 W=320
            Mat inputMat = resizeWithoutPadding(src.clone(),320,240);

            // BGR -> RGB
            Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_BGR2RGB);
            inputMat.convertTo(inputMat, CvType.CV_32FC1);

            // 数组
            float[] whc = new float[ 3 * 240 * 320 ];
            inputMat.get(0, 0, whc);

            // 将图片维度从 HWC 转换为 CHW
            float[] chw = new float[whc.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < whc.length; i += 3) {
                    chw[j] = whc[i];

                    // 减去 127.5f 再除 127.5f
                    chw[j] = chw[j] - 127.5f;
                    chw[j] = chw[j] / 127.5f;

                    j++;
                }
            }

            // 张量 C=3 H=240 W=320
            // input -> [1, 3, 240, 320] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,240,320});

            // 推理
            OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));


            // ---------模型1输出-----------
            // batchno_classid_score_x1y1x2y2 -> [-1, 7] -> FLOAT
            float[] data = ((float[][])(res.get(0)).getValue())[0];

            float batchno = data[0];
            float classid = data[1];
            float score = data[2];
            float x1 = data[3];
            float y1 = data[4];
            float x2 = data[5];
            float y2 = data[6];

            if(score>=0.7){

                // 坐标缩放到原图
                x1 = x1 * src.width();
                y1 = y1 * src.height();
                x2 = x2 * src.width();
                y2 = y2 * src.height();

                // 在原始图像中进行截图脸部
                this.face = new Mat(
                        src,
                        new Rect(
                                Float.valueOf(x1).intValue(),
                                Float.valueOf(y1).intValue(),
                                Float.valueOf(x2 - x1).intValue(),
                                Float.valueOf(y2 - y1).intValue()
                        )
                ).clone();

                this.face_point = this.face.clone();

            }

        }

        // 人脸重建 密集点
        public void doFaceRebuild1() throws Exception{

            // ---------模型2输入-----------
            // input -> [-1, 3, 120, 120] -> FLOAT
            // ---------模型2输出-----------
            // camera_matrix -> [-1, 3, 4] -> FLOAT
            // landmarks -> [-1, 38365, 3] -> FLOAT

            int face_w = this.face.width();
            int face_h = this.face.height();

            // 缩放为 120*120 这里使用填充来缩放保持脸部比例不变
            Mat face_120 = resizeWithPadding(this.face.clone(),120,120);

            // BGR -> RGB
            Imgproc.cvtColor(face_120, face_120, Imgproc.COLOR_BGR2RGB);
            face_120.convertTo(face_120, CvType.CV_32FC1);

            // 数组
            float[] whc = new float[ 3 * 120 * 120 ];
            face_120.get(0, 0, whc);

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

            // 张量 C=3 H=120 W=120
            // [-1, 3, 120, 120]
            OnnxTensor tensor = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,3,120,120});

            // 推理
            OrtSession.Result res = session2.run(Collections.singletonMap("input", tensor));

            // camera_matrix -> [-1, 3, 4] -> FLOAT  相机投影矩阵用于将三位点投影到二位平面,如果要在二维平面进行展示
            float camera_matrix[][] = ((float[][][])(res.get(0)).getValue())[0];
            // 相机投影矩阵一般如下：
            // camera_matrix = [
            //    [f_x, 0, c_x, t_x],
            //    [0, f_y, c_y, t_y],
            //    [0, 0, 1, t_z]
            //]


            // landmarks -> [-1, 38365, 3] -> FLOAT  三位坐标点
            float landmarks[][] = ((float[][][])(res.get(1)).getValue())[0];

            for(int i=0;i<38365;i++){
                // 模型输出的xyz是在120*120的坐标中
                float[] xyz = landmarks[i];

                // 保存每个点缩放到320的尺寸上
                this.face_point_320_320_320.add(new float[]{
                        xyz[0] * (320f / 120f),
                        xyz[1] * (320f / 120f),
                        xyz[2] * (320f / 120f)
                });

                // 在face上进行画点缩放到原始图片尺寸
                if(i%112==0){
                    Imgproc.circle(this.face_point, new Point(
                            transferPointWithPadding(xyz,face_w,face_h,120,120)[0],
                            transferPointWithPadding(xyz,face_w,face_h,120,120)[1]
                    ),1, color, 1);
                }
            }

        }

        public JPanel showFace(){

            // 返回一个 panel 旋转显示人脸
            JFXPanel fxPanel = new JFXPanel();
            fxPanel.setPreferredSize(new Dimension(320, 320));

            // 创建Swing的面板用于包含JavaFX的Panel
            JPanel panel = new JPanel(new BorderLayout());
            panel.setPreferredSize(new Dimension(320, 320));
            panel.add(fxPanel, BorderLayout.CENTER);

            // 创建JavaFX的场景和根节点
            Group root = new Group();
            Scene scene = new Scene(root);
            // 将JavaFX的场景设置到JavaFX的Panel中
            fxPanel.setScene(scene);

            // 在Swing的Frame可见之后，初始化JavaFX
            Platform.runLater(() -> {


                // 在根节点中绘制脸部网格点
                for (float[] point : face_point_320_320_320) {
                    Sphere sphere = new Sphere(0.7); // 创建球体作为脸部网格点
                    sphere.setTranslateX(point[0]); // 设置x坐标
                    sphere.setTranslateY(point[1]); // 设置y坐标
                    sphere.setTranslateZ(point[2]); // 设置z坐标
                    PhongMaterial material = new PhongMaterial();
                    material.setDiffuseColor(javafx.scene.paint.Color.RED); // 设置球体的漫反射颜色为红色
                    material.setSpecularColor(Color.GREEN); // 设置球体的镜面反射颜色为白色
                    sphere.setMaterial(material); // 应用材质到球体
                    root.getChildren().add(sphere); // 将球体添加到根节点
                }

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

        // 显示结果
        public void show(){


            // 弹窗显示
            JFrame frame = new JFrame("Image");
            frame.setSize(src.width(), src.height());
            JPanel panel = new JPanel(new GridLayout(1,3,10,10));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);


            // 显示3弹窗
            panel.add(new JLabel(new ImageIcon(mat2BufferedImage(src))));
//            panel.add(new JLabel(new ImageIcon(mat2BufferedImage(face))));
            panel.add(new JLabel(new ImageIcon(mat2BufferedImage(face_point))));
            panel.add(showFace());


            frame.pack();



        }

    }




    public static void main(String[] args) throws Exception{


        // 模型1 人脸检测
        // ---------模型1输入-----------
        // input -> [1, 3, 240, 320] -> FLOAT
        // ---------模型1输出-----------
        // batchno_classid_score_x1y1x2y2 -> [-1, 7] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_alignment_mesh_pose\\RFB-320_240x320_post.onnx");


        // 模型2 3d重建 密集点
        // ---------模型2输入-----------
        // input -> [-1, 3, 120, 120] -> FLOAT
        // ---------模型2输出-----------
        // camera_matrix -> [-1, 3, 4] -> FLOAT
        // landmarks -> [-1, 38365, 3] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_alignment_mesh_pose\\dense_face_Nx3x120x120.onnx");


        // 加载图片
        ImageObj imageObj = new ImageObj(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\style_gan_cartoon\\face.JPG");
//                "\\model\\deeplearning\\face_alignment_mesh_pose\\face1.png");


        // 人脸检测
        imageObj.doFaceDetection();

        // 人脸密集点重建
        imageObj.doFaceRebuild1();

        // 显示
        imageObj.show();

    }


}
