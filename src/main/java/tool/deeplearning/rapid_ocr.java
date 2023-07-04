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
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
*   @desc : 捷智OCR - 文本区域检测 + 文本方向检测+ 文本识别
 *
 *          开源地址：
 *          https://github.com/RapidAI/RapidOCR
 *
 *          模型1 文本检测：
 *          ch_PP-OCRv3_det_infer.onnx
 *          推理参考：
 *          https://github.com/RapidAI/RapidOCR/tree/main/python/rapidocr_onnxruntime/ch_ppocr_v3_det
 *
 *          模型2 中文文本识别：
 *          ch_PP-OCRv3_rec_infer.onnx
 *          推理参考：
 *          https://github.com/RapidAI/RapidOCR/tree/main/python/rapidocr_onnxruntime/ch_ppocr_v3_rec
 *
 *          模型3 英文文本识别：
 *          en_PP-OCRv3_rec_infer.onnx
 *
 *          模型4 文本方向检测
 *          ch_ppocr_mobile_v2.0_cls_infer.onnx
 *          推理参考：
 *          https://github.com/RapidAI/RapidOCR/tree/main/python/rapidocr_onnxruntime/ch_ppocr_v2_cls
 *
 *          其他推理参考：
 *          https://github.com/hpc203/PaddleOCR-v3-onnxrun-cpp-py/tree/main/python/weights
 *
 *
 *
*   @auth : tyf
*   @date : 2022-05-12  16:44:43
*/
public class rapid_ocr {

    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 模型3
    public static OrtEnvironment env3;
    public static OrtSession session3;


    // 模型3
    public static OrtEnvironment env4;
    public static OrtSession session4;

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
//        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

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
//        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }

    public static void init3(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env3 = OrtEnvironment.getEnvironment();
        session3 = env3.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型3输入-----------");
        session3.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型3输出-----------");
        session3.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session3.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }


    // 环境初始化
    public static void init4(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env4 = OrtEnvironment.getEnvironment();
        session4 = env4.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型4输入-----------");
        session4.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型4输出-----------");
        session4.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session4.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }

    public static class ImageObj{

        // 原始图片
        Mat src;
        // 模型1输入
        Mat dst;
        // 字符列表
        ArrayList<String> characterStr;
        ArrayList<String> character;
        // 保存每个文本区域的识别结果
        ArrayList<ArrayList<String>> charsRes;
        public ImageObj(String image) {
            this.src = this.readImg(image);
            // 从模型2的元数据中读取字符列表
            this.characterStr = this.readCharacterDict();
            this.character = addSpecialChar(this.characterStr);
        }
        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        // 保存检测到的文本区域
        ArrayList<Mat> textArea;
        ArrayList<Mat> textArea2;
        // 读取字符列表
        public ArrayList<String> readCharacterDict(){
            // 假设character_dict_path对应的字符列表为 ["a", "b", "c", " "]
            ArrayList<String> characterDict = new ArrayList<>();
            try {
                String chars = session2.getMetadata().getCustomMetadata().get("character");
                char[] charArray = chars.toCharArray();
                for (int i = 0; i < charArray.length; i++) {
                    if(!String.valueOf(charArray[i]).equals("\n")){
                        characterDict.add(String.valueOf(charArray[i]));
                    }
                }
            }
            catch (Exception e){
                e.printStackTrace();
            }
            return characterDict;
        }
        // 文本检测
        public void doDetection() throws Exception{

            // ---------模型1输入-----------
            // x -> [-1, 3, -1, -1] -> FLOAT
            // ---------模型1输出-----------
            // sigmoid_0.tmp_0 -> [-1, 1, -1, -1] -> FLOAT

            //    thresh: 0.3
            //    box_thresh: 0.5
            //    max_candidates: 1000
            //    unclip_ratio: 1.6
            //    use_dilation: true
            //    score_mode: "fast"

            int wh = 736;// 宽高,源码是保持32的倍数即可

            Mat img = src.clone();
            Mat input = resizeWithoutPadding(img,wh,wh);

            this.dst = input.clone();

            // BGR -> RGB
            Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
            input.convertTo(input, CvType.CV_32FC1);

            // 归一化
            input.convertTo(input, CvType.CV_32FC3, 1.0 / 255.0);

            double[] meanValue = {0.485, 0.456, 0.406};
            double[] stdValue = {0.229, 0.224, 0.225};
            Core.subtract(input, new Scalar(meanValue), input);
            Core.divide(input, new Scalar(stdValue), input);

            // 数组
            float[] whc = new float[ 3 * wh * wh];
            input.get(0, 0, whc);

            // 模型要求 chw
            float[] chw = whc2chw(whc);
            OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,wh,wh});
            // 推理
            OrtSession.Result res = session1.run(Collections.singletonMap("x", tensor));

            // ---------模型1输出-----------
            // sigmoid_0.tmp_0 -> [-1, 1, -1, -1] -> FLOAT
            // 输出的 1 * 1 * w * h 也就是一张掩膜图,需要进行二值化然后使用 opencv 查找轮廓
            float[][] pred_logits = ((float[][][][])(res.get(0)).getValue())[0][0];

            float mask_thresh = 0.2f;
            float min_size_thresh = 3;

            // 先用图片显示出来看一下长什么样子,使用灰度图进行弹窗显示
            // this.showDetectionRes(pred_logits,wh,wh,mask_thresh);

            // 根据阈值二值化然后查找轮廓
            ArrayList<MatOfPoint> contours = this.findContours(pred_logits,mask_thresh);

            // 计算每个轮廓的最小外接矩形
            ArrayList<Point[]> contours_points = new ArrayList<>();
            for (int index = 0; index < contours.size(); index++) {
                MatOfPoint contour = contours.get(index);
                // 最小外交矩形四个点保存到 points 中
                RotatedRect boundingBox = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
                // 这个是求最大外接矩形
//                Rect boundingRect = Imgproc.boundingRect(contour);
                Point[] points = new Point[4];
                boundingBox.points(points);
                // 计算最小外接矩形的最小变成
                double sside = getMinSideLength(points);
                if (sside < min_size_thresh) {
                    continue;
                }
                // 保存过滤后的最小外接矩形边框坐标
                contours_points.add(points);
            }

            // 4个关键点按照 左上、左下、右上、右下
            ArrayList<Point[]> contours_points_sort = this.contoursPointsSort(contours_points);

            // 对边框进行扩展,并限制在图片整体区域内,不扩展的话可能边框只能框住文字的一部分,扩展的点不能超出原始图片
            ArrayList<Point[]> contours_points_sort_unclip = this.contoursPointsUnclip(contours_points_sort,src.width(),src.height());

            // 先用图片显示一下过滤后得到的文本边框
            this.showContours(contours_points_sort_unclip);

            // 保存所有文本区域,投影变换到指定的尺寸
            this.saveContours(contours_points_sort_unclip);

        }


        // 文本方向分类
        public void doClassifier() throws Exception{

            // ---------模型4输入-----------
            // x -> [-1, 3, -1, -1] -> FLOAT
            // ---------模型4输出-----------
            // save_infer_model/scale_0.tmp_1 -> [-1, 2] -> FLOAT

            int c = 3;
            int h = 192;
            int w = 192;

            this.textArea2 = new ArrayList<>();

            // 遍历每个文本区域进行处理
            for(int index =0;index <textArea.size();index++){

                Mat in = textArea.get(index);
                // 带有paddding的缩放
                Mat input = resizeWithoutPadding(in.clone(),w,h);

                // BGR -> RGB
                Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
                input.convertTo(input, CvType.CV_32FC1);

                // 转为 whc
                float[] whc = new float[ 3 * h * w];
                input.get(0, 0, whc);

                // 转为 cwh 并归一化道-1~1
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

                // 推理
                OnnxTensor tensor = OnnxTensor.createTensor(env4, FloatBuffer.wrap(chw), new long[]{1,3,h,w});
                // 推理
                OrtSession.Result res = session4.run(Collections.singletonMap("x", tensor));

                // ---------模型4输出-----------
                // save_infer_model/scale_0.tmp_1 -> [-1, 2] -> FLOAT
                float[] data = ((float[][])(res.get(0)).getValue())[0];

                float c1 = data[0]; // 旋转0概率
                float c2 = data[1]; // 旋转180概率

                // 判断图片是否是旋转180度 // TODO
                if( c2>c1 ){
                    if(c2> 0.9){
                        // 需要将图片再旋转180度恢复到正常
                    }
                }

                // 保存最终的需要识别的文本区域
                textArea2.add(in);

            }

        }

        // 文本识别
        public void doRecetion() throws Exception{

            this.charsRes = new ArrayList<>();

            // 文本识别（中文）
            // ---------模型2输入-----------
            // x -> [-1, 3, -1, -1] -> FLOAT
            // ---------模型2输出-----------
            // softmax_5.tmp_0 -> [-1, -1, 6625] -> FLOAT

            int c = 3;
            int h = 48;
            int w = 320;


            // 遍历每个文本区域进行处理
            for(int index =0;index <textArea2.size();index++){

                Mat in = textArea2.get(index);

                // 带有paddding的缩放
                Mat input = resizeWithoutPadding(in.clone(),w,h);

                // BGR -> RGB
                Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2RGB);
                input.convertTo(input, CvType.CV_32FC1);

                // 转为 whc
                float[] whc = new float[ 3 * h * w];
                input.get(0, 0, whc);

                // 转为 cwh 并归一化道-1~1
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

                // 推理
                OnnxTensor tensor = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,3,h,w});
                // 推理
                OrtSession.Result res = session2.run(Collections.singletonMap("x", tensor));

                // ---------模型2输出-----------
                // softmax_5.tmp_0 -> [-1, -1, 6625] -> FLOAT
                // 1,40,6625
                float[][][] data = ((float[][][])(res.get(0)).getValue());

                // 读取当前文本区域的字符
                this.charsRes.add(decode(data));

            }

        }

        public ArrayList<String> decode(float[][][] preds) {
            int[][] predsIdx = new int[preds.length][preds[0].length];
            float[][] predsProb = new float[preds.length][preds[0].length];

            // Get index and probability values from the predictions
            for (int i = 0; i < preds.length; i++) {
                for (int j = 0; j < preds[i].length; j++) {
                    float maxProb = Float.MIN_VALUE;
                    int maxIdx = -1;
                    for (int k = 0; k < preds[i][j].length; k++) {
                        if (preds[i][j][k] > maxProb) {
                            maxProb = preds[i][j][k];
                            maxIdx = k;
                        }
                    }
                    predsIdx[i][j] = maxIdx;
                    predsProb[i][j] = maxProb;
                }
            }

            return ctcDecode(predsIdx, predsProb, true);
        }

        private ArrayList<String> addSpecialChar(ArrayList<String> characterStr) {
            ArrayList<String> character = new ArrayList<>();
            character.add("blank");
            character.addAll(characterStr);
            return character;
        }

        private ArrayList<String> ctcDecode(int[][] textIndex, float[][] textProb, boolean isRemoveDuplicate) {
            ArrayList<String> result = new ArrayList<>();
            ArrayList<Integer> ignoredTokens = getIgnoredTokens();

            for (int batchIdx = 0; batchIdx < textIndex.length; batchIdx++) {
                StringBuilder sb = new StringBuilder();
                for (int idx = 0; idx < textIndex[batchIdx].length; idx++) {
                    int charIndex = textIndex[batchIdx][idx];
                    if (ignoredTokens.contains(charIndex)) {
                        continue;
                    }
                    if (isRemoveDuplicate && idx > 0 && textIndex[batchIdx][idx - 1] == charIndex) {
                        continue;
                    }
                    sb.append(character.get(charIndex));
                }
                result.add(sb.toString());
            }

            return result;
        }

        private ArrayList<Integer> getIgnoredTokens() {
            ArrayList<Integer> ignoredTokens = new ArrayList<>();
            ignoredTokens.add(0); // for ctc blank
            return ignoredTokens;
        }


        public static ArrayList<Point[]> contoursPointsSort(ArrayList<Point[]> contours_points){

            ArrayList<Point[]> tmp = new ArrayList<>();
            // 便利每个边框
            contours_points.stream().forEach(n->{

                // 找到 x+y 最小的  左上
                Point p1 = Arrays.stream(n).min(Comparator.comparingDouble(o -> o.x + o.y)).get();

                // 找到 x+y 最大的  右下
                Point p2 = Arrays.stream(n).max(Comparator.comparingDouble(o -> o.x + o.y)).get();

                // 找到 x 最大的  右上
                Point p3 = Arrays.stream(n).filter(point -> point!=p1&&point!=p2).max(Comparator.comparingDouble(o -> o.x)).get();

                // 找到 x 最小  左下
                Point p4 = Arrays.stream(n).filter(point -> point!=p1&&point!=p2).min(Comparator.comparingDouble(o -> o.x)).get();

                tmp.add(new Point[]{
                        // 左上、左下、右上、右下
                        p1,p4,p3,p2
                });

            });

            return tmp;
        }

        // 对边界框进行扩展,防止文字被拦腰截断
        public static ArrayList<Point[]> contoursPointsUnclip(ArrayList<Point[]> contours_points,int max_w,int max_h){

            ArrayList<Point[]> tmp = new ArrayList<>();

            // 原始图片面积
            double src_area = max_w * max_h;

            contours_points.forEach(n -> {

                // 左上、左下、右上、右下
                Point p1 = n[0];
                Point p2 = n[1];
                Point p3 = n[2];
                Point p4 = n[3];

                // 计算最小的一条边的边长
                double h = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                double w = Math.sqrt(Math.pow(p4.x - p2.x, 2) + Math.pow(p4.y - p2.y, 2));

                // 区域面积
                double area = h * w;

                // 缩放比例
                double ratio = 0;


                // 宽高比是否接近1
                if( w/h >= 0.8 && w/h <= 1.3){
                    // 区域面积占比原图面积很大说明多个文字.多个文字宽高很大需要缩小缩放比例
                    if( area/src_area >= 0.001){
                        ratio =  0.05;
                    }
                    // 说明文字很少
                    else{
                        ratio = 0.1;
                    }
                }
                // 不接近1
                else{
                    // 区域面积占比原图面积很大则缩小缩放距离
                    if( area/src_area >= 0.001){
                        ratio =  0.3;
                    }else{
                        ratio = 0.6;
                    }
                }

                double dis = Math.min(w,h) * ratio;

                // 返回扩展的坐标
                tmp.add(new Point[]{
                        new Point(p1.x-dis,p1.y-dis), // 左上、
                        new Point(p2.x-dis,p2.y+dis), // 左下、
                        new Point(p3.x+dis,p3.y-dis), // 右上、
                        new Point(p4.x+dis,p4.y+dis), // 右下
                });

            });

            return tmp;
        }

        // 执行多边形的扩展
        public static Polygon expandPolygon(Polygon polygon, double horizontalScale, double verticalScale) {
            // 创建一个新的多边形对象
            Polygon expanded = new Polygon();

            // 计算多边形的中心点
            int centerX = 0;
            int centerY = 0;
            for (int i = 0; i < polygon.npoints; i++) {
                centerX += polygon.xpoints[i];
                centerY += polygon.ypoints[i];
            }
            centerX /= polygon.npoints;
            centerY /= polygon.npoints;

            // 执行横向扩展
            for (int i = 0; i < polygon.npoints; i++) {
                int dx = polygon.xpoints[i] - centerX;
                int dy = polygon.ypoints[i] - centerY;
                int expandedX = (int) (polygon.xpoints[i] + dx * horizontalScale);
                int expandedY = (int) (polygon.ypoints[i] + dy * horizontalScale);
                expanded.addPoint(expandedX, expandedY);
            }

            // 计算扩展后多边形的中心点
            int expandedCenterX = 0;
            int expandedCenterY = 0;
            for (int i = 0; i < expanded.npoints; i++) {
                expandedCenterX += expanded.xpoints[i];
                expandedCenterY += expanded.ypoints[i];
            }
            expandedCenterX /= expanded.npoints;
            expandedCenterY /= expanded.npoints;

            // 执行纵向扩展
            for (int i = 0; i < expanded.npoints; i++) {
                int dx = expanded.xpoints[i] - expandedCenterX;
                int dy = expanded.ypoints[i] - expandedCenterY;
                int expandedX = (int) (expanded.xpoints[i] + dx * verticalScale);
                int expandedY = (int) (expanded.ypoints[i] + dy * verticalScale);
                expanded.xpoints[i] = expandedX;
                expanded.ypoints[i] = expandedY;
            }

            return expanded;
        }


        // 计算多边形的面积
        private static double calculatePolygonArea(Polygon polygon) {
            int n = polygon.npoints;
            int[] xPoints = polygon.xpoints;
            int[] yPoints = polygon.ypoints;
            double area = 0.0;
            for (int i = 0; i < n; i++) {
                area += xPoints[i] * yPoints[(i + 1) % n];
                area -= yPoints[i] * xPoints[(i + 1) % n];
            }
            return Math.abs(area / 2.0);
        }

        // 计算多边形的周长
        private static double calculatePolygonPerimeter(Polygon polygon) {
            int n = polygon.npoints;
            int[] xPoints = polygon.xpoints;
            int[] yPoints = polygon.ypoints;
            double perimeter = 0.0;
            for (int i = 0; i < n; i++) {
                int dx = xPoints[(i + 1) % n] - xPoints[i];
                int dy = yPoints[(i + 1) % n] - yPoints[i];
                perimeter += Math.sqrt(dx * dx + dy * dy);
            }
            return perimeter;
        }

        public static Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(inputWidth, inputHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }

        // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,添加留白保存宽高比为1
        public static Mat resizeWithPadding(Mat src,int netWidth,int netHeight) {
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

        public float[] whc2chw(float[] src) {
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
        // 显示模型1的输出
        public void showDetectionRes(float[][] data,int w,int h,float thresh){

            BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);

            for (int y = 0; y < w; y++) {
                for (int x = 0; x < h; x++) {
                    // 设置灰度值
                    float d = data[y][x];
                    if( d > thresh){
                        // 设置白色
                        img.setRGB(x, y, Color.WHITE.getRGB());
                    }
                    else{
                        // 设置黑色
                        img.setRGB(x, y, Color.BLACK.getRGB());
                    }
                }
            }

            // 弹窗显示
            JFrame frame = new JFrame();
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(img)));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);

        }

        // 调用 opencv 进行二值化然后查找轮廓
        public ArrayList<MatOfPoint> findContours(float[][] data,float thresh){

            int w = data.length;
            int h = data[0].length;

            Mat mat = new Mat(w,h, CvType.CV_8U);
            for (int y = 0; y < w; y++) {
                for (int x = 0; x < h; x++) {
                    mat.put(y, x, data[y][x] > thresh ? 255 : 0 );
                }
            }

            // 查找所有轮廓并保存到 contours 中
            ArrayList<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

            return contours;
        }

        // 保存所有文本区域并投影变换到指定区域
        public void saveContours(ArrayList<Point[]> points){
            this.textArea = new ArrayList<>();
            float w_scala = Float.valueOf(src.width()) / Float.valueOf(dst.width());
            float h_scala = Float.valueOf(src.height()) / Float.valueOf(dst.height());
            // 区域区域截取
            points.stream().forEach(n->{
                Point p1 = n[0];
                Point p2 = n[1];
                Point p3 = n[2];
                Point p4 = n[3];
                // 创建一个矩形区域,从原始图片中截取
                Rect roi = Imgproc.boundingRect(new MatOfPoint(
                        new Point(p1.x * w_scala,p1.y * h_scala),
                        new Point(p2.x * w_scala,p2.y * h_scala),
                        new Point(p3.x * w_scala,p3.y * h_scala),
                        new Point(p4.x * w_scala,p4.y * h_scala)
                ));
                // 提取子图像
                Mat subImage = new Mat(src, roi);
                // 保存
                textArea.add(subImage);
            });
        }

        // 显示模型1的输出二值化查找轮廓的结果
        public void showContours(ArrayList<Point[]> points){

            // 复制一个原始尺寸的图片用于标注
            Mat show = src.clone();

            Scalar color1 = new Scalar(0, 0, 255);
            Scalar color2 = new Scalar(0, 255, 0);

            // 在原始图片上进行标注
            float w_scala = Float.valueOf(src.width()) / Float.valueOf(dst.width());
            float h_scala = Float.valueOf(src.height()) / Float.valueOf(dst.height());

            // 先将所有框按照p1坐标排个序
            points.stream().sorted(Comparator.comparingDouble(o -> o[0].x + o[0].y));

            // 遍历所有框
            AtomicInteger index = new AtomicInteger(1);
            points.stream().forEach(n->{
                // 左上、左下、右上、右下
                Point p1 = n[0];
                Point p2 = n[1];
                Point p3 = n[2];
                Point p4 = n[3];
                // 画线,注意这里不是直接使用p2p4矩形画框,因为这个矩形四个点是倾斜的,p2p4画框是和图片平行的
                Imgproc.line(show, new Point(p1.x * w_scala,p1.y * h_scala), new Point(p2.x * w_scala,p2.y * h_scala), color1, 1);
                Imgproc.line(show, new Point(p3.x * w_scala,p3.y * h_scala), new Point(p4.x * w_scala,p4.y * h_scala), color1, 1);
                Imgproc.line(show, new Point(p1.x * w_scala,p1.y * h_scala), new Point(p3.x * w_scala,p3.y * h_scala), color1, 1);
                Imgproc.line(show, new Point(p2.x * w_scala,p2.y * h_scala), new Point(p4.x * w_scala,p4.y * h_scala), color1, 1);
                // 标注一下序号
                Imgproc.putText(show, String.valueOf(index.get()) , new Point(p1.x * w_scala,p1.y * h_scala), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1);
                // 画四个点
//                Imgproc.circle(show, new Point(p1.x * w_scala,p1.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p2.x * w_scala,p2.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p3.x * w_scala,p3.y * h_scala), 2, color2, 1);
//                Imgproc.circle(show, new Point(p4.x * w_scala,p4.y * h_scala), 2, color2, 1);

                index.getAndIncrement();
            });

            // 弹窗显示
            JFrame frame = new JFrame();
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(mat2BufferedImage(show))));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }

        private double getMinSideLength(Point[] points) {
            double minSideLength = Double.MAX_VALUE;
            for (int i = 0; i < 4; i++) {
                Point p1 = points[i];
                Point p2 = points[(i + 1) % 4];
                double sideLength = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
                if (sideLength < minSideLength) {
                    minSideLength = sideLength;
                }
            }
            return minSideLength;
        }

        // Mat 转 BufferedImage
        public static BufferedImage mat2BufferedImage(Mat mat){
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

        public void printChars(){
            for(int i=0;i<charsRes.size();i++){
                ArrayList<String> text = charsRes.get(i);
                System.out.println("文本区域编号:"+(i+1)+",识别文本:"+Arrays.toString(text.toArray()));
            }
        }
    }

    public static void main(String[] args) throws Exception{



        // 文本检测
        // ---------模型1输入-----------
        // x -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型1输出-----------
        // sigmoid_0.tmp_0 -> [-1, 1, -1, -1] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\rapid_ocr\\ch_PP-OCRv3_det_infer.onnx");

        // 文本识别（中文）
        // ---------模型2输入-----------
        // x -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型2输出-----------
        // softmax_5.tmp_0 -> [-1, -1, 6625] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\rapid_ocr\\ch_PP-OCRv3_rec_infer.onnx");


        // 文本识别（英文）
        // ---------模型3输入-----------
        // x -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型3输出-----------
        // softmax_2.tmp_0 -> [-1, -1, 97] -> FLOAT
        init3(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\rapid_ocr\\en_PP-OCRv3_rec_infer.onnx");


        // 文本方向分类
        // ---------模型4输入-----------
        // x -> [-1, 3, -1, -1] -> FLOAT
        // ---------模型4输出-----------
        // save_infer_model/scale_0.tmp_1 -> [-1, 2] -> FLOAT
        init4(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\rapid_ocr\\ch_ppocr_mobile_v2.0_cls_infer.onnx");

        // 发票
        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\rapid_ocr\\fapiao.jpeg";
//        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\rapid_ocr\\book.png";
//        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\rapid_ocr\\baidu.png";
        ImageObj image = new ImageObj(pic);

        // 文本检测
        image.doDetection();

        // 方向分类
        image.doClassifier();

        // 文本识别
        image.doRecetion();

        // 输入识别的文本
        image.printChars();

    }





}
