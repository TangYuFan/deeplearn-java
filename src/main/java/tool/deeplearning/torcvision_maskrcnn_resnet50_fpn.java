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
import java.util.List;

/**
*   @desc : torcvision_maskrcnn_resnet50 实例分割
*   @auth : tyf
*   @date : 2022-05-11  18:44:32
*/
public class torcvision_maskrcnn_resnet50_fpn {

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
        // images -> [1, 3, 1024, 1024] -> FLOAT
        session.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型输出-----------");
        // boxes -> [-1, 4] -> FLOAT
        // labels -> [-1] -> INT64
        // scores -> [-1] -> FLOAT
        // masks -> [-1, -1, -1, -1] -> FLOAT
        session.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });

    }

    // 目标对象
    public static class Obj{
        float[] xyxy;// 边框信息
        long label;// 类别信息
        float score;// 得分
        float[][][] mask;// 掩膜信息
        List<int[]> point;// mask 按照阈值过滤后保留的点
        public Obj(float[] xyxy, long label, float score, float[][][] mask) {
            this.xyxy = xyxy;
            this.label = label;
            this.score = score;
            this.mask = mask;
        }

        public void setPoint(List<int[]> point) {
            this.point = point;
        }
    }

    // 使用 opencv 读取图片到 mat
    public static Mat readImg(String path){
        Mat img = Imgcodecs.imread(path);
        return img;
    }

    // 将一个 src_mat 修改尺寸后存储到 dst_mat 中,不添加留白
    public static Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
        // 调整图像大小
        Mat resizedImage = new Mat();
        Size size = new Size(inputWidth, inputHeight);
        Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
        return resizedImage;
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


    public static OnnxTensor transferTensor(OrtEnvironment env,Mat dst,int inputCount,int inputChannel,int inputWidth,int inputHeight){
        // BGR -> RGB
        // python 中图像通常以rgb加载,java通常以 bgr加载
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
//        dst.convertTo(dst, CvType.CV_32FC1);// 矩阵转单精度浮点型
        dst.convertTo(dst, CvType.CV_32FC1, 1. / 255);// 矩阵转单精度浮点型,并且每个元素除255进行归一化
        // 初始化一个输入数组 channels * netWidth * netHeight
        float[] whc = new float[ inputChannel * inputWidth * inputHeight ];
        dst.get(0, 0, whc);
        // 得到最终的图片转 float 数组 whc 转 chw
        // prtorch 中图片以chw格式加载
        float[] chw = whc2cwh(whc);
        // 创建 onnxruntime 需要的 tensor
        OnnxTensor tensor = null;
        try {
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{inputCount,inputChannel,inputWidth,inputHeight});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }

    // 按照得分过滤一批
    public static void filter1(List<Obj> objs,float scoreThreshold){
        // 删除得分低的
        objs.removeIf(obj -> obj.score<scoreThreshold);
    }

    // 按照重叠过滤一批
    public static void filter2(List<Obj> objs,float nmsThreshold){

        // 先按得分降序
        objs.sort((o1, o2) -> Float.compare(o2.score,o1.score));

        // 需要删除的
        List<Obj> res = new ArrayList<>();

        while (!objs.isEmpty()){
            Obj max = objs.get(0);
            res.add(max);
            Iterator<Obj> it = objs.iterator();
            while (it.hasNext()){
                Obj bi = it.next();
                // 计算交并比
                if(calculateIoU(max.xyxy,bi.xyxy)>=nmsThreshold){
                    it.remove();
                }
            }
        }

        // 保存剩下的
        objs.clear();
        objs.addAll(res);

    }

    // 计算每个目标框中可能包含目标的点的坐标
    public static void filter3(List<Obj> objs,float maskThreshold){
        objs.stream().forEach(n->{
            // mask掩膜 1 * netWidth * netHeight 这里直接取第一个
            float[][] mask = n.mask[0];
            // 边框
            float[] box = n.xyxy;
            // 将原始图片上这些点的位置颜色改变一下
            int xmin = Float.valueOf(box[0]).intValue();
            int ymin = Float.valueOf(box[1]).intValue();
            int xmax = Float.valueOf(box[2]).intValue();
            int ymax = Float.valueOf(box[3]).intValue();
            // 将输出box范围内大于0的点的坐标获取出来,并转成原始图片上面的坐标,注意mask每个元素是 0~1 而不是 0~1,这里按照阈值取感兴趣的点
            List<int[]> point = new ArrayList<>();
            for (int y = ymin; y < ymax; y++) {
                for (int x = xmin; x < xmax; x++) {
                    float pixelValue = mask[y][x];
                    // 大于阈值,说明这个点可能在目标上
                    if(pixelValue>maskThreshold){
                        int[] xy_src = new int[]{x,y};
                        point.add(xy_src);
                    }
                }
            }
            // 保存到对象中,在后面统一标注时使用
            n.setPoint(point);
        });
    }

    // 交并比
    private static double calculateIoU(float[] box1, float[] box2) {
        double x1 = Math.max(box1[0], box2[0]);
        double y1 = Math.max(box1[1], box2[1]);
        double x2 = Math.min(box1[2], box2[2]);
        double y2 = Math.min(box1[3], box2[3]);
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
        double box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }

    // 弹窗显示
    public static void show(Mat src,List<Obj> objs,int imgWidth,int imgHeight,int netWidth,int netHeight){


        objs.stream().forEach(n->{

            // 输出的坐标
            float[] xyxy = n.xyxy;

            // 转换到原图坐标小
            float[] xyxy_ = transferPointWithPadding(xyxy,imgWidth,imgHeight,netWidth,netHeight);

            // 类别
            long type = n.label;

            // 概率
            float score = n.score;

            // 颜色
            Random random = new Random();
            Scalar color = new Scalar(random.nextInt(255), random.nextInt(255), random.nextInt(255));

            // 画边框
            Imgproc.rectangle(
                    src,
                    new Point(Float.valueOf(xyxy_[0]).intValue(), Float.valueOf(xyxy_[1]).intValue()),
                    new Point(Float.valueOf(xyxy_[2]).intValue(), Float.valueOf(xyxy_[3]).intValue()),
                    color,
                    2);
            // 画标签
            Imgproc.putText(
                    src,
                    String.valueOf(score).substring(0,4),// 概率取两位小数
                    new Point(Float.valueOf(xyxy_[0]-1).intValue(), Float.valueOf(xyxy_[1]-7).intValue()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    2);


            // 当前目标可能包含的像素点,mask信息,将这些点标注一下
            List<int[]> point = n.point;
            point.stream().forEach(p->{
                // 坐标转为原始图片
                int[] xy_src = transferXYWithPadding(p[0],p[1],imgWidth,imgHeight,netWidth,netHeight);
                // 在这个点画一个"." 注意坐标是缩放过的也就是说点变稀疏了直接修改像素点颜色可能看不出来
//                Imgproc.putText(
//                        src,
//                        ".",
//                        new Point(xy_src[0], xy_src[1]),
//                        Imgproc.FONT_HERSHEY_SIMPLEX,
//                        1,
//                        color,
//                        2);
                // 修改颜色,修改周围上下左右一共八个点,
                double[] c = src.get(xy_src[1],xy_src[0]);
                c[0] = 255;
                // 修改颜色后设置回去,修改周围上下左右一共八个点,
                src.put(xy_src[1],xy_src[0],c);
                src.put(xy_src[1]-1,xy_src[0],c);
                src.put(xy_src[1],xy_src[0]-1,c);
                src.put(xy_src[1]+1,xy_src[0],c);
                src.put(xy_src[1],xy_src[0]+1,c);
                src.put(xy_src[1]+1,xy_src[0]+1,c);
                src.put(xy_src[1]-1,xy_src[0]-1,c);
                src.put(xy_src[1]+1,xy_src[0]-1,c);
                src.put(xy_src[1]-1,xy_src[0]+1,c);

            });

        });


        // Mat 转 BufferedImage
        BufferedImage imageDst = mat2BufferedImage(src);
        // 缩小
        imageDst = scaleImage(imageDst,0.5);
        // 弹窗显示
        JFrame frame = new JFrame("Image");
        frame.setSize(imageDst.getWidth(), imageDst.getHeight());
        JLabel label = new JLabel(new ImageIcon(imageDst));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }

    // 将 BufferedImage 缩小到一定比例
    public static BufferedImage scaleImage(BufferedImage originalImage, double scale) {
        int newWidth = (int) (originalImage.getWidth() * scale);
        int newHeight = (int) (originalImage.getHeight() * scale);
        // 创建新的BufferedImage对象
        BufferedImage newImage = new BufferedImage(newWidth, newHeight, originalImage.getType());
        // 绘制原始图像并缩小它
        Graphics2D g = newImage.createGraphics();
        g.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return newImage;
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

    public static float[] transferPointWithPadding(float[] xyxy,int imgWidth,int imgHeight,int netWidth,int netHeight){
        float gain = Math.min((float) netWidth / imgWidth, (float) netHeight / imgHeight);
        float padW = (netWidth - imgWidth * gain) * 0.5f;
        float padH = (netHeight - imgHeight * gain) * 0.5f;
        float xmin = xyxy[0];
        float ymin = xyxy[1];
        float xmax = xyxy[2];
        float ymax = xyxy[3];
        // 缩放过后的坐标
        float xmin_ = Math.max(0, Math.min(imgWidth - 1, (xmin - padW) / gain));
        float ymin_ = Math.max(0, Math.min(imgHeight - 1, (ymin - padH) / gain));
        float xmax_ = Math.max(0, Math.min(imgWidth - 1, (xmax - padW) / gain));
        float ymax_ = Math.max(0, Math.min(imgHeight - 1, (ymax - padH) / gain));
        return new float[]{xmin_,ymin_,xmax_,ymax_};
    }

    public static int[] transferXYWithPadding(int x,int y,int imgWidth,int imgHeight,int netWidth,int netHeight){
        float gain = Math.min((float) netWidth / imgWidth, (float) netHeight / imgHeight);
        float padW = (netWidth - imgWidth * gain) * 0.5f;
        float padH = (netHeight - imgHeight * gain) * 0.5f;
        // 缩放过后的坐标
        float x_ = Math.max(0, Math.min(imgWidth - 1, (x - padW) / gain));
        float y_ = Math.max(0, Math.min(imgHeight - 1, (y - padH) / gain));
        return new int[]{
                Float.valueOf(x_).intValue(),
                Float.valueOf(y_).intValue()
        };
    }


    public static void main(String[] args) throws Exception{

        // 加载权重,读取模型
        String weight = new File("").getCanonicalPath() + "\\model\\deeplearning\\maskrcnn_resnet50_fpn\\maskrcnn_resnet50_fpn.onnx";
        init(weight);

        // 模型输入宽高,onnx网站可以看到
        int inputCount = 1;
        int inputChannel = 3;
        int inputWidth = 1024;
        int inputHeight = 1024;

        // 读取图片,保存原始宽高
        String img = new File("").getCanonicalPath() + "\\model\\deeplearning\\yolov5\\img.jpg";
        Mat src = readImg(img);

        System.out.println("---------开始预处理-----------");

        // 图片原始宽高
        int imgWidth = src.width();
        int imgHeight = src.height();

        // 使用 padding 填充后 resize 成模型输入尺寸
        Mat dst = resizeWithPadding(src,inputWidth,inputHeight);

        // 转为 tensor
        // 预处理 主要是 BGR2RGB、归一化、whc2cwh
        OnnxTensor tensor =  transferTensor(env,dst,inputCount,inputChannel,inputWidth,inputHeight);

        System.out.println("---------开始推理-----------");

        // 推理
        OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));

        // 获取四个输出
        OnnxTensor boxes = (OnnxTensor)result.get("boxes").get();
        OnnxTensor labels = (OnnxTensor)result.get("labels").get();
        OnnxTensor scores = (OnnxTensor)result.get("scores").get();
        OnnxTensor masks = (OnnxTensor)result.get("masks").get();

        // 获取输出的张量
        float[][] boxesArray = (float[][])boxes.getValue();
        long[] labelsArray = (long[])labels.getValue();
        float[] scoresArray = (float[])scores.getValue();
        float[][][][] masksArray = (float[][][][])masks.getValue();


        System.out.println("---------开始后处理-----------");


        // 目标个数
        int size = labelsArray.length;

        // 遍历每个目标,生成目标对象
        List<Obj> objs = new ArrayList<>();

        for (int i=0;i<size;i++){
            float[] xyxy = boxesArray[i];// 边框信息
            long label = labelsArray[i];// 类别信息
            float score = scoresArray[i];// 得分
            float[][][] mask = masksArray[i];// 掩膜信息
            objs.add(new Obj(xyxy,label,score,mask));
        }

        // 按照得分过滤一批,小于0.5的丢弃
        filter1(objs,0.3f);

        // 按照nms过滤去重,计算交并比
        filter2(objs,0.45f);

        // 计算目标框中包含目标的点,mask过滤
        filter3(objs,0.7f);

        // 标注原始图片,弹窗显示,传入模型宽高,从src读取原始宽高确定坐标缩放比
        show(src,objs,imgWidth,imgHeight,inputWidth,inputHeight);


    }



}
