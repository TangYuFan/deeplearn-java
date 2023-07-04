package tool.deeplearning;


import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.nio.file.Path;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import javax.swing.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 *   @desc : 安全帽检测 , mobilenet0.25 ,djl 推理
 *   @auth : tyf
 *   @date : 2022-06-15  09:11:41
 */
public class safety_helmet_detect_mobilenet2_djl {


    public static class SmallSafetyHelmetDetect {

        public SmallSafetyHelmetDetect() {}

        public Criteria<Image, DetectedObjects> criteria(Image image,String modelPath) {
            Map<String, Object> arguments = new ConcurrentHashMap<>();
            int[] size = scale(image.getHeight(), image.getWidth());
            arguments.put("width", size[1]);
            arguments.put("height", size[0]);
            arguments.put("resize", true);
            arguments.put("rescale", true);
            arguments.put("normalize", true);
            arguments.put("threshold", 0.2);

            Translator<Image, DetectedObjects> translator = YoloTranslator.builder(arguments).build();

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .optEngine("MXNet")
                            .build();

            return criteria;
        }

        private static int[] scale(int h, int w) {
            int min = Math.min(h, w);
            float scale = 1.0F;

            scale = (float) 416 * 1.0F / (float) min;

            return new int[] {(int) ((float) h * scale), (int) ((float) w * scale)};
        }
    }



    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // 模型
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\safety_helmet_detect_mobilenet2_djl\\mobilenet0.25.zip";

        // 测试图片
        String picPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\safety_helmet_detect_mobilenet2_djl\\safety_helmet.jpg";



        Path imageFile = new File(picPath).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);

        Criteria<Image, DetectedObjects> criteria = new SmallSafetyHelmetDetect().criteria(image,modelPath);

        try (ZooModel model = ModelZoo.loadModel(criteria);
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detections = predictor.predict(image);
            List<DetectedObjects.DetectedObject> items = detections.items();

            List<String> names = new ArrayList<>();
            List<Double> prob = new ArrayList<>();
            List<BoundingBox> boxes = new ArrayList<>();
            for (int i = 0; i < items.size(); i++) {
                DetectedObjects.DetectedObject item = items.get(i);
                if (item.getProbability() < 0.3f) {
                    continue;
                }
                names.add(item.getClassName() + " " + item.getProbability());
                prob.add(item.getProbability());
                boxes.add(item.getBoundingBox());
            }


            // 标注
            image.drawBoundingBoxes(detections);

            // 弹窗显示
            BufferedImage backup = (BufferedImage)image.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(backup.getWidth(), backup.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(backup)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);


            System.out.println("识别结果：");
            System.out.println(detections);


        }
    }


}
