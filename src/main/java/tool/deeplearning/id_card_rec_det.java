package tool.deeplearning;


import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

/**
*   @desc : 文本识别 （高精度：文本区域检测+文本识别）
 *
 *
*   @auth : tyf
*   @date : 2022-05-23  15:18:19
*/
public class id_card_rec_det {

    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    public static OrtEnvironment env2;
    public static OrtSession session2;

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
        System.out.println("---------模型输出-----------");
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
        // 原始图片
        Mat src;
        // 模型1宽高需要是32的倍数
        int wh = 960;
        // 模型1输入图片
        Mat dst;
        // 模型1 nms过滤前的文本区域
        ArrayList<float[]> datas;
        // 模型1 nms过滤后的文本区域
        ArrayList<float[]> datas_nms;
        public ImageObj(String pic){
            this.src = this.readImg(pic);
            this.dst = resizeWithoutPadding(src.clone(),wh,wh);
        }
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public Mat resizeWithoutPadding(Mat src, int netWidth, int netHeight) {
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(netWidth, netHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
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

        // 识别文本区域
        public void doDet(){

            Mat input = this.dst.clone();

            // BGR -> RGB
            Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
            input.convertTo(input, CvType.CV_32FC1);

            // 数组
            float[] hwc = new float[ 3 * wh * wh ];
            input.get(0, 0, hwc);

            try {

                // ---------模型1输入-----------
                // input -> [-1, -1, -1, 3] -> FLOAT
                OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(hwc), new long[]{1,wh,wh,3});
                OrtSession.Result res = session1.run(Collections.singletonMap("input", tensor));

                // ---------模型输出-----------
                // geo_map -> [-1, -1, -1, 5] -> FLOAT    1 * 240  * 240 5
                // score_map -> [-1, -1, -1, 1] -> FLOA     1 * 240 * 240 1
                float[][][] geo_map = ((float[][][][])(res.get(0)).getValue())[0];
                float[][][] score_map = ((float[][][][])(res.get(1)).getValue())[0];


                // 其中 240 是原始宽高除以4得到的特征图尺寸
                int height = wh/4; // 240
                int width = wh/4; // 240

                ArrayList<float[]> box = new ArrayList();

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        float d0 =geo_map[i][j][0];
                        float d1 =geo_map[i][j][1];
                        float d2 =geo_map[i][j][2];
                        float d3 =geo_map[i][j][3];
                        float angle =geo_map[i][j][4];
                        float score = score_map[i][j][0];
                        if(score>0.8){

                            // 当前特征点在原图中的坐标
                            int x = j * 4;
                            int y = i * 4;

                            if (angle >= 0) {
                                float a0 = 0;
                                float a1 = -d0 - d2;
                                float a2 = d1 + d3;
                                float a3 = -d0 - d2;
                                float a4 = d1 + d3;
                                float a5 = 0;
                                float a6 = 0;
                                float a7 = 0;
                                float a8 = d3;
                                float a9 = -d2;

                                float rotate_x0 = Double.valueOf(Math.cos(angle)).floatValue();
                                float rotate_x1 = Double.valueOf(Math.sin(angle)).floatValue();
                                float rotate_y0 = Double.valueOf(-Math.sin(angle)).floatValue();
                                float rotate_y1 = Double.valueOf(Math.cos(angle)).floatValue();

                                float bx0 = rotate_x0 * a0 + rotate_x1 * a1;
                                float bx1 = rotate_x0 * a2 + rotate_x1 * a3;
                                float bx2 = rotate_x0 * a4 + rotate_x1 * a5;
                                float bx3 = rotate_x0 * a6 + rotate_x1 * a7;
                                float bx4 = rotate_x0 * a8 + rotate_x1 * a9;

                                float by0 = rotate_y0 * a0 + rotate_y1 * a1;
                                float by1 = rotate_y0 * a2 + rotate_y1 * a3;
                                float by2 = rotate_y0 * a4 + rotate_y1 * a5;
                                float by3 = rotate_y0 * a6 + rotate_y1 * a7;
                                float by4 = rotate_y0 * a8 + rotate_y1 * a9;

                                float org_x = x - bx4;
                                float org_y = y - by4;
                                float new_px0 = bx0 + org_x;
                                float new_py0 = by0 + org_y;
                                float new_px1 = bx1 + org_x;
                                float new_py1 = by1 + org_y;
                                float new_px2 = bx2 + org_x;
                                float new_py2 = by2 + org_y;
                                float new_px3 = bx3 + org_x;
                                float new_py3 = by3 + org_y;

                                box.add(new float[]{
                                        new_px0,new_py0,
                                        new_px1,new_py1,
                                        new_px2,new_py2,
                                        new_px3,new_py3,
                                        score
                                });

                            } else {
                                float a0 = -d1 - d3;
                                float a1 = -d0 - d2;
                                float a2 = 0;
                                float a3 = -d0 - d2;
                                float a4 = 0;
                                float a5 = 0;
                                float a6 = -d1 - d3;
                                float a7 = 0;
                                float a8 = -d1;
                                float a9 = -d2;

                                float rotate_x0 = Double.valueOf(Math.cos(-angle)).floatValue();
                                float rotate_x1 = Double.valueOf(-Math.sin(-angle)).floatValue();
                                float rotate_y0 = Double.valueOf(Math.sin(-angle)).floatValue();
                                float rotate_y1 = Double.valueOf(Math.cos(-angle)).floatValue();

                                float bx0 = rotate_x0 * a0 + rotate_x1 * a1;
                                float bx1 = rotate_x0 * a2 + rotate_x1 * a3;
                                float bx2 = rotate_x0 * a4 + rotate_x1 * a5;
                                float bx3 = rotate_x0 * a6 + rotate_x1 * a7;
                                float bx4 = rotate_x0 * a8 + rotate_x1 * a9;

                                float by0 = rotate_y0 * a0 + rotate_y1 * a1;
                                float by1 = rotate_y0 * a2 + rotate_y1 * a3;
                                float by2 = rotate_y0 * a4 + rotate_y1 * a5;
                                float by3 = rotate_y0 * a6 + rotate_y1 * a7;
                                float by4 = rotate_y0 * a8 + rotate_y1 * a9;

                                float org_x = x - bx4;
                                float org_y = y - by4;
                                float new_px0 = bx0 + org_x;
                                float new_py0 = by0 + org_y;
                                float new_px1 = bx1 + org_x;
                                float new_py1 = by1 + org_y;
                                float new_px2 = bx2 + org_x;
                                float new_py2 = by2 + org_y;
                                float new_px3 = bx3 + org_x;
                                float new_py3 = by3 + org_y;

                                box.add(new float[]{
                                        new_px0,new_py0,
                                        new_px1,new_py1,
                                        new_px2,new_py2,
                                        new_px3,new_py3,
                                        score
                                });

                            }
                        }
                    }
                }


                this.datas = box;

            }
            catch (Exception e){
                e.printStackTrace();
            }

        }


        // 进行文本区域nms
        public void doNms(){

            float nmsThreshold = 0.2f;

            // 用于nms过滤后保存的
            ArrayList<float[]> box = new ArrayList<>();

            // 开始nms过滤先按照得分排序 x1y1 x2y2 x3y3 x4y4 score
            this.datas.sort((o1, o2) -> Float.compare(o2[8],o1[8]));

            while (!datas.isEmpty()){
                float[] max = datas.get(0);
                box.add(max);
                Iterator<float[]> it = datas.iterator();
                while (it.hasNext()) {
                    // 交并比
                    float[] obj = it.next();
                    // 不规则四边形交并比计算
                    double iouValue = calculateIoU(
                            // x1y1 x2y2 x3y3 x4y4
                            new float[]{max[0],max[1],max[2],max[3],max[4],max[5],max[6],max[7]},
                            new float[]{obj[0],obj[1],obj[2],obj[3],obj[4],obj[5],obj[6],obj[7]}
                    );
                    if (iouValue > nmsThreshold) {
                        it.remove();
                    }
                }
            }

            // nsm后的边框就行保存
            this.datas_nms = box;

        }


        // 计算四边形的交并比
        private double calculateIoU(float[] box1, float[] box2) {

            float box1_x1 = box1[0];
            float box1_y1 = box1[1];
            float box1_x2 = box1[2];
            float box1_y2 = box1[3];
            float box1_x3 = box1[4];
            float box1_y3 = box1[5];
            float box1_x4 = box1[6];
            float box1_y4 = box1[7];

            float box2_x1 = box2[0];
            float box2_y1 = box2[1];
            float box2_x2 = box2[2];
            float box2_y2 = box2[3];
            float box2_x3 = box2[4];
            float box2_y3 = box2[5];
            float box2_x4 = box2[6];
            float box2_y4 = box2[7];

            // 计算交集区域的坐标范围
            float inter_x1 = Math.max(box1_x1, box2_x1);
            float inter_y1 = Math.max(box1_y1, box2_y1);
            float inter_x2 = Math.min(box1_x3, box2_x3);
            float inter_y2 = Math.min(box1_y3, box2_y3);

            // 计算交集区域的面积
            float inter_area = Math.max(0, inter_x2 - inter_x1 + 1) * Math.max(0, inter_y2 - inter_y1 + 1);

            // 计算并集区域的面积
            float box1_area = calculateArea(box1_x1, box1_y1, box1_x2, box1_y2, box1_x3, box1_y3, box1_x4, box1_y4);
            float box2_area = calculateArea(box2_x1, box2_y1, box2_x2, box2_y2, box2_x3, box2_y3, box2_x4, box2_y4);
            float union_area = box1_area + box2_area - inter_area;

            // 计算交并比（IoU）
            double iou = inter_area / union_area;

            return iou;
        }

        // 辅助方法：计算矩形区域的面积
        private float calculateArea(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4) {
            // 假设四边形为凸多边形，可以使用 Shoelace Formula 计算面积
            float area = Math.abs(0.5f * ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4)));
            return area;
        }

        // 识别文字
        public void doRec(){


            // ---------模型2输入-----------
            // image -> [-1, -1, -1, -1] -> FLOAT
            // ---------模型2输出-----------
            // output -> [51, 1, 6846] -> FLOAT

        }


        public void show(){


            // 将边框画出来
            Scalar color1 = new Scalar(0, 0, 255);
            datas_nms.stream().forEach(n->{
                Imgproc.line(dst, new Point(n[0],n[1]), new Point(n[2],n[3]), color1, 2);
                Imgproc.line(dst, new Point(n[4],n[5]), new Point(n[6],n[7]), color1, 2);
                Imgproc.line(dst, new Point(n[0],n[1]), new Point(n[6],n[7]), color1, 2);
                Imgproc.line(dst, new Point(n[2],n[3]), new Point(n[4],n[5]), color1, 2);
            });


            // 弹窗显示
            JFrame frame = new JFrame();
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(mat2BufferedImage(dst))));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);


        }


    }
    public static void main(String[] args) throws Exception{


        // ---------模型1输入-----------
        // input -> [-1, -1, -1, 3] -> FLOAT
        // ---------模型输出-----------
        // geo_map -> [-1, -1, -1, 5] -> FLOAT
        // score_map -> [-1, -1, -1, 1] -> FLOA
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_rec_det\\det.onnx");


        // ---------模型2输入-----------
        // image -> [-1, -1, -1, -1] -> FLOAT
        // ---------模型2输出-----------
        // output -> [51, 1, 6846] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_rec_det\\rec.onnx");



        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\id_card_licence\\id_card.png";
        ImageObj imageObj = new ImageObj(img);

        // 识别文本区域
        imageObj.doDet();
        // 文本区域nms
        imageObj.doNms();
        // 识别文字
        imageObj.doRec();
        // 显示
        imageObj.show();
    }

}
