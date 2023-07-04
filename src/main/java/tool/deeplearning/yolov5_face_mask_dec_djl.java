package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV5TranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


/**
*   @desc : yolov5 实现人脸口罩检测 , djl 进行onnx引擎推理
*   @auth : tyf
*   @date : 2022-06-13  19:24:41
*/
public class yolov5_face_mask_dec_djl {


    public static class MaskDetection {


        String checkpoint;
        Image out;

        private MaskDetection(String checkpoint) {
            this.checkpoint = checkpoint;
        }


        public DetectedObjects predict(String pa) throws IOException, ModelException, TranslateException {

            Image img = ImageFactory.getInstance().fromFile(new File(pa).toPath());

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(checkpoint).toPath())
//                            .optModelUrls("https://resources.djl.ai/demo/onnxruntime/face_mask_detection.zip")
                            .optEngine("OnnxRuntime")
                            .optTranslatorFactory(new YoloV5TranslatorFactory())
                            .optProgress(new ProgressBar())
                            .optArgument("optApplyRatio", true) // post process
                            .optArgument("rescale", true) // post process
                            .build();

            try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
                try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                    DetectedObjects detection = predictor.predict(img);

                    // 保存识别结果
                    saveBoundingBoxImage(img, detection);
                    return detection;
                }
            }
        }

        private void saveBoundingBoxImage(Image img, DetectedObjects detection)
                throws IOException {

            img.drawBoundingBoxes(detection);
            this.out = img;

        }

        public void show(){
            // 弹窗显示
            BufferedImage img = (BufferedImage)out.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(img.getWidth(), img.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(img)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);

        }
    }


    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // 模型地址,这里从本地映入
        // https://resources.djl.ai/demo/onnxruntime/face_mask_detection.zip
        // face_mask_detection.onnx     模型
        // serving.properties   模型预处理配置
        // synset.txt   类别

        // 模型使用onnx引擎需要加入下面依赖
        // <dependency>
        //			<groupId>ai.djl.onnxruntime</groupId>
        //			<artifactId>onnxruntime-engine</artifactId>
        //			<version>0.22.1</version>
        //		</dependency>

        String checkpoint = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov5_face_mask_dec_djl\\face_mask_detection.onnx";

        // 图片
        String imgfile = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov5_face_mask_dec_djl\\face_mask.png";


        // 预测
        MaskDetection maskDetection = new MaskDetection(checkpoint);
        DetectedObjects detection = maskDetection.predict(imgfile);

        System.out.println("识别结果:");
        System.out.println(detection);

        // 弹窗显示
        maskDetection.show();


    }

}
