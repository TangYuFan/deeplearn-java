package tool.deeplearning;


import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.videoio.VideoWriter;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.nio.Buffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
*   @desc : 蚂蚁雅黑特效  first_order_motion_model  使用 jdl 推理
 *
 *
 *          两个模型:
 *          kp_detector.pt  人脸关键点
 *          generator.pt    视频合成
 *
*   @auth : tyf
*   @date : 2022-05-10  11:37:47
*/
public class first_order_motion_model_jdl {

    // 用于NDArray 操作
    static NDManager manager = NDManager.newBaseManager(Device.cpu());

    static {
//        设置默认DJL引擎
//        System.setProperty("ai.djl.default_engine", "PyTorch");
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 定义模型1关键点检测输入输出处理器
    public static class NPtKTranslator implements Translator<NDArray, Map> {

        public NPtKTranslator() {
        }
        // processInput 方法用于处理输入数据，将其调整为模型需要的格式。
        // 最后，经过处理后的数据以 NDList 的形式返回
        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) {
            return new NDList(input.get(0));
        }
        // processOutput 方法用于处理模型的输出数据，将其转换为一个包含两个 NDArray 对象的 Map。
        // 在这段代码中，模型的输出包括两个 NDArray：jacobian 和 value
        @Override
        public Map<String,NDArray> processOutput(TranslatorContext ctx, NDList list) {
            //默认行为是 情空
            Map a = new HashMap();
            NDArray jacobian = list.get(0);
            jacobian.detach();
            NDArray value = list.get(1);
            value.detach();
            a.put(list.get(0).getName(),jacobian);
            a.put(list.get(1).getName(),value);
            return a;
        }
        // getBatchifier 方法指定了用于批处理的方式，
        // 这里使用的是 Batchifier.STACK，表示将输入数据在批处理时进行堆叠操作。
        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

    }


    // 定义模型2视频合成输入输出处理器
    public static class PtGTranslator implements Translator<List,Image> {

        public PtGTranslator() {
        }
        @Override
        public NDList processInput(TranslatorContext ctx, List input) {
            NDArray source = (NDArray)input.get(0);
            Map<String,NDArray> kp_driving = (Map<String,NDArray>) input.get(1);
            NDArray kp_driving_v = kp_driving.get("value");
            kp_driving_v = kp_driving_v.broadcast(new Shape(1,10,2));
            NDArray kp_driving_j = kp_driving.get("jacobian");
            kp_driving_j = kp_driving_j.broadcast(new Shape(1,10,2,2));
            Map<String,NDArray> kp_source = (Map<String,NDArray>) input.get(2);
            NDArray kp_source_v = kp_source.get("value");
            kp_source_v = kp_source_v.broadcast(new Shape(1,10,2));
            NDArray kp_source_j = kp_source.get("jacobian");
            kp_source_j = kp_source_j.broadcast(new Shape(1,10,2,2));
            Map<String,NDArray> kp_driving_initial = (Map<String,NDArray>) input.get(3);
            NDArray kp_initial_v = kp_driving_initial.get("value");
            kp_initial_v = kp_initial_v.broadcast(new Shape(1,10,2));
            NDArray kp_initial_j = kp_driving_initial.get("jacobian");
            kp_initial_j = kp_initial_j.broadcast(new Shape(1,10,2,2));
            NDList re = new NDList();
            re.add(source);
            re.add(kp_driving_v);
            re.add(kp_driving_j);
            re.add(kp_source_v);
            re.add(kp_source_j);
            re.add(kp_initial_v);
            re.add(kp_initial_j);
            return re;
        }

        @Override
        public Image processOutput(TranslatorContext ctx, NDList list) {
            for(NDArray ig : list){
                if(ig.getName().equals("prediction")){
                    NDArray img = ig.get(0);
                    img = img.mul(255).toType(DataType.UINT8, true);
                    return ImageFactory.getInstance().fromNDArray(img);
                }
            }
            return null;
        }
        @Override
        public Batchifier getBatchifier() {
            return null;
        }

    }

    // 图片转张量
    private static NDArray image2NDArray(Image img){
        NDArray driving0 = img.toNDArray(manager);
        driving0 = NDImageUtils.resize(driving0, 256, 256);
        driving0 = driving0.div(255);
        driving0 = driving0.transpose(2, 0, 1);
        driving0 = driving0.toType(DataType.FLOAT32,false);
        driving0 = driving0.broadcast(new Shape(1,3,256,256));
        return driving0;
    }

    public static List<Image> getKeyFrame(File filePath) throws Exception {
        // 使用rtsp的时候需要使用 FFmpegFrameGrabber，不能再用 FrameGrabber
        FFmpegFrameGrabber grabberI = FFmpegFrameGrabber.createDefault(filePath);
        grabberI.start();
        Java2DFrameConverter converter = new Java2DFrameConverter();
        System.out.println("-----------------------------------");
        // 帧总数
        BufferedImage bImg = null;
        System.out.println("总帧长:"+grabberI.getLengthInFrames());
        int audios = grabberI.getLengthInAudioFrames() >= Integer.MAX_VALUE ? 0 : grabberI.getLengthInAudioFrames();
        int vidoes = grabberI.getLengthInVideoFrames() >= Integer.MAX_VALUE ? 0 : grabberI.getLengthInVideoFrames();
        //获取图片
        int frame_number =  vidoes;
        Frame img = null;
        grabberI.flush();
        List<Image> cvImgs = new ArrayList<>();
        for (int i = 0; i < frame_number; i++) {

            if((img = grabberI.grab()) == null){
                continue;
            }
            if( (bImg = converter.convert(img)) == null){
                continue;
            }
            cvImgs.add(ImageFactory.getInstance().fromImage(copyImg(bImg)));
        }
        grabberI.release();
        return cvImgs;
    }

    public static BufferedImage copyImg(BufferedImage img){
        BufferedImage checkImg = new BufferedImage(img.getWidth(),img.getHeight(),img.getType() == 0 ? 5 : img.getType());
        checkImg.setData(img.getData());
        return checkImg;
    }

    public static BufferedImage resize(BufferedImage img, int newWidth, int newHeight) {
        java.awt.Image scaledImage = img.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH);
        BufferedImage scaledBufferedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = scaledBufferedImage.createGraphics();
        g2d.drawImage(scaledImage, 0, 0, null);
        g2d.dispose();
        return scaledBufferedImage;
    }

    public static Mat bufferedImageToMat(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int type = CvType.CV_8UC3;  // 假设 BufferedImage 是 3 通道的彩色图像

        Mat mat = new Mat(height, width, type);

        if (image.getType() == BufferedImage.TYPE_INT_RGB || image.getType() == BufferedImage.TYPE_INT_ARGB) {
            int[] data = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();

            int pixelIndex = 0;
            byte[] byteData = new byte[width * height * 3];
            for (int i = 0; i < data.length; i++) {
                byteData[pixelIndex++] = (byte) (data[i] & 0xFF);
                byteData[pixelIndex++] = (byte) ((data[i] >> 8) & 0xFF);
                byteData[pixelIndex++] = (byte) ((data[i] >> 16) & 0xFF);
            }

            mat.put(0, 0, byteData);
        } else {
            throw new IllegalArgumentException("Unsupported BufferedImage type: " + image.getType());
        }

        return mat;
    }


    public static void main(String[] args) throws Exception{


        // 人脸检测模型
        Path pt1 = Paths.get("model\\deeplearning\\first_order_motion_model_jdl\\kp_detector.pt");

        // 视频合成模型
        Path pt2 = Paths.get("model\\deeplearning\\first_order_motion_model_jdl\\generator.pt");

        // 原始图片
        Path src_img = Paths.get("model\\deeplearning\\style_gan_cartoon\\face.JPG");

        // 驱动视频
        Path src_video = Paths.get("model\\deeplearning\\first_order_motion_model_jdl\\drive.mp4");

        // 模型1
        ZooModel m1 = ModelZoo.loadModel( Criteria.builder()
                .setTypes(NDArray.class, Map.class)
                .optTranslator(new NPtKTranslator()) // NPtKTranslator 是定义的模型翻译器、专门处理输入和输出
                .optEngine("PyTorch")
                .optDevice(Device.cpu())
                .optModelPath(pt1) // 模型路径
                .build());

        // 模型2
        ZooModel m2 = ModelZoo.loadModel(Criteria.builder()
                .setTypes(List.class, Image.class)
                .optEngine("PyTorch")
                .optTranslator(new PtGTranslator()) // PtGTranslator 是定义的模型翻译器、专门处理输入和输出
                .optDevice(Device.cpu())
                .optModelPath(pt2) // 模型路径
                .build());


        // 创建模型的预测器
        Predictor<NDArray, Map> kPredictor = m1.newPredictor();
        Predictor<List, Image> generator = m2.newPredictor();


        // 读取原始图片
        Image srcImage = ImageFactory.getInstance().fromFile(src_img);

        // 读取驱动视频所有帧
        List<Image> drivingImages = getKeyFrame(src_video.toFile());
        int count = drivingImages.size();

        // 将原始图片转为张量 3*256*256
        NDArray kp_source_tensor = image2NDArray(srcImage);

        // 原始图片进行关键点推理
        Map kp_source_kp = kPredictor.predict(kp_source_tensor);

        // 驱动视频第1帧进行关键点推理
        Map kp_driving_kp = kPredictor.predict(image2NDArray(drivingImages.get(0)));



        // 弹窗显示合成前后的视频
        JFrame frame = new JFrame("Image");
        JPanel panel = new JPanel();
        JLabel label1 = new JLabel();
        JLabel label2 = new JLabel();
        frame.getContentPane().add(panel);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // 写到新的视频中
//        int fps = 30;
//        String filename = "output.avi";
//        VideoWriter videoWriter = new VideoWriter(filename, VideoWriter.fourcc('M', 'J', 'P', 'G'), fps, new Size(srcImage.getWidth(),srcImage.getHeight()));


        // 遍历所有驱动帧
        for(int i=0;i<count;i++){

            System.out.println("关键点以及合成处理,总共帧:"+count+",当前处理到:"+(i+1));

            BufferedImage imgIn = (BufferedImage)drivingImages.get(i).getWrappedImage();

            // 对驱动视频每一帧进行推理
            NDArray im = image2NDArray(drivingImages.get(i));
            Map kp_driving = kPredictor.predict(im);

            // 第二个模型需要输入下面的 tmp
            List<Object> tmp = new ArrayList<>();
            tmp.add(kp_source_tensor);// 保存原始图片的张量
            tmp.add(kp_driving);// 保存每一帧驱动视频的关键点推理结果
            tmp.add(kp_source_kp);// 保存原始图片的关键点推理结果
            tmp.add(kp_driving_kp);// 保存驱动视频第一帧的关键点推理结果

            // 进行合成
            BufferedImage imgOut = (BufferedImage)generator.predict(tmp).getWrappedImage();


            // 弹窗显示
            label1.setIcon(new ImageIcon(imgIn)); // 驱动帧
            label2.setIcon(new ImageIcon(imgOut));// 合成帧
            panel.add(label1);
            panel.add(label2);
            frame.pack();

            // 写入到新的视频中
            BufferedImage imaOut2 = resize(imgOut,srcImage.getWidth(),srcImage.getHeight());
            Mat mat = bufferedImageToMat(imaOut2);
//            videoWriter.write(mat);

        }

        // 释放资源
//        videoWriter.release();
        System.out.println("处理完毕");
        System.exit(0);

    }


}
