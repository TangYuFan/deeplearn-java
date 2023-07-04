package tool.deeplearning;


import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.nio.file.Paths;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import javax.swing.*;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;



/**
*   @desc : yolov4 烟雾火灾检测（paddle模型） ,djl推理
 *
 *
*   @auth : tyf
*   @date : 2022-06-14  18:21:33
*/
public class yolov4_fire_smoke_detect_djl {


    public static class FireSmokeDetect {

        private static final Logger logger = LoggerFactory.getLogger(FireSmokeDetect.class);

        public FireSmokeDetect() {}

        public Criteria<Image, DetectedObjects> criteria(String modelPath) {
            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("width", 640);
            arguments.put("height", 640);
            arguments.put("resize", true);
            arguments.put("rescale", true);
            //    arguments.put("toTensor", false);
            //    arguments.put("range", "0,1");
            //    arguments.put("normalize", "false");
            arguments.put("threshold", 0.2);
            arguments.put("nmsThreshold", 0.5);

            Translator<Image, DetectedObjects> translator = YoloV5Translator.builder(arguments).build();

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .optEngine("PyTorch")
                            .optDevice(Device.cpu())
                            .build();

            return criteria;
        }
    }




    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 模型
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov4_fire_smoke_detect_djl\\fire_smoke.zip";

        // 测试图片
        String picPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\yolov4_fire_smoke_detect_djl\\fire_smoke.png";



        Criteria<Image, DetectedObjects> criteria = new FireSmokeDetect().criteria(modelPath);

        try (ZooModel model = ModelZoo.loadModel(criteria);
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            Path imageFile = new File(picPath).toPath();
            Image image = ImageFactory.getInstance().fromFile(imageFile);

            // 转为图片用于标注
            BufferedImage backup = (BufferedImage)image.getWrappedImage();

            // 识别得到box
            DetectedObjects detections = predictor.predict(image);
            List<DetectedObjects.DetectedObject> items = detections.items();


            for (int i = 0; i < items.size(); i++) {
                DetectedObjects.DetectedObject item = items.get(i);
                if (item.getProbability() < 0.3f) {
                    continue;
                }
                System.out.println("类别:"+item.getClassName()+",得分:"+item.getProbability()+",坐标:"+item.getBoundingBox());
                // 标注
                Graphics2D g2d = backup.createGraphics();
                g2d.setColor(Color.RED);
                g2d.setStroke(new BasicStroke(2));
                // 边框
                g2d.drawRect(
                        Double.valueOf(item.getBoundingBox().getBounds().getX()).intValue() * backup.getWidth() / 640,
                        Double.valueOf(item.getBoundingBox().getBounds().getY()).intValue() * backup.getHeight() / 640,
                        Double.valueOf(item.getBoundingBox().getBounds().getWidth()).intValue() * backup.getWidth() / 640,
                        Double.valueOf(item.getBoundingBox().getBounds().getHeight()).intValue() * backup.getHeight() / 640
                );
                // 类别
                g2d.drawString(
                        item.getClassName(),
                        Double.valueOf(item.getBoundingBox().getBounds().getX()).intValue() * backup.getWidth() / 640,
                        Double.valueOf(item.getBoundingBox().getBounds().getY()).intValue() * backup.getHeight() / 640
                );
                // 释放资源
                g2d.dispose();
            }


            JFrame frame = new JFrame("Image");
            frame.setSize(backup.getWidth(), backup.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(backup)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);


        }
    }

}
