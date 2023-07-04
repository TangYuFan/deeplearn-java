package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.translator.SimplePoseTranslator;

import javax.swing.*;

/**
*   @desc : 人体姿态估计（关键点）resnet18/resnet50模型  djl 推理
 *
*   @auth : tyf
*   @date : 2022-06-15  12:03:11
*/
public class pose_estimation2_djl {



    public static class PersonDetection {

        private static final Logger logger = LoggerFactory.getLogger(PersonDetection.class);

        String model1;
        public PersonDetection(String model1) {
            this.model1 = model1;
        }


        public DetectedObjects predict(Image img) throws IOException, ModelException, TranslateException {
            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("width", 512);
            arguments.put("height", 512);
            arguments.put("resize", true);
            arguments.put("rescale", true);
            arguments.put("threshold", 0.2);

            Translator<Image, DetectedObjects> translator =
                    SingleShotDetectionTranslator.builder(arguments).build();

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .optEngine("MXNet")
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(model1).toPath())
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<Image, DetectedObjects> ssd = ModelZoo.loadModel(criteria)) {
                try (Predictor<Image, DetectedObjects> predictor = ssd.newPredictor()) {
                    DetectedObjects detectedObjects = predictor.predict(img);
                    List<String> names = new ArrayList<>();
                    List<Double> prob = new ArrayList<>();
                    List<BoundingBox> rect = new ArrayList<>();

                    List<DetectedObjects.DetectedObject> items = detectedObjects.items();
                    for (DetectedObjects.DetectedObject item : items) {
                        if ("person".equals(item.getClassName())) {
                            names.add(item.getClassName());
                            prob.add(item.getProbability());
                            rect.add(item.getBoundingBox());
                        }
                    }
                    return new DetectedObjects(names, prob, rect);
                }
            }
        }
    }


    public static class PoseResnetEstimation {

        String model2;
        public PoseResnetEstimation(String model2) {
            this.model2 = model2;
        }


        public Joints predict(Image img) throws IOException, ModelException, TranslateException {
            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("width", 192);
            arguments.put("height", 256);
            arguments.put("resize", true);
            arguments.put("normalize", true);
            arguments.put("threshold", 0.2);

            Translator<Image, Joints> translator = SimplePoseTranslator.builder(arguments).build();

            Criteria<Image, Joints> criteria =
                    Criteria.builder()
                            .optEngine("MXNet")
                            .setTypes(Image.class, Joints.class)
                            .optModelPath(new File(model2).toPath())
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<Image, Joints> pose = ModelZoo.loadModel(criteria);
                 Predictor<Image, Joints> predictor = pose.newPredictor()) {
                Joints joints = predictor.predict(img);
                return joints;
            }
        }
    }


    public static Image getSubImage(Image img, BoundingBox box, float factor) {
        Rectangle rect = box.getBounds();
        // 左上角坐标 - Upper left corner coordinates
        int x1 = (int) (rect.getX() * img.getWidth());
        int y1 = (int) (rect.getY() * img.getHeight());
        // 宽度，高度 - width, height
        int w = (int) (rect.getWidth() * img.getWidth());
        int h = (int) (rect.getHeight() * img.getHeight());
        // 左上角坐标 - Upper right corner coordinates
        int x2 = x1 + w;
        int y2 = y1 + h;

        // 外扩大100%，防止对齐后人脸出现黑边
        // Expand by 100% to prevent black edges after alignment
        int new_x1 = Math.max((int) (x1 + x1 * factor / 2 - x2 * factor / 2), 0);
        int new_x2 = Math.min((int) (x2 + x2 * factor / 2 - x1 * factor / 2), img.getWidth() - 1);
        int new_y1 = Math.max((int) (y1 + y1 * factor / 2 - y2 * factor / 2), 0);
        int new_y2 = Math.min((int) (y2 + y2 * factor / 2 - y1 * factor / 2), img.getHeight() - 1);
        int new_w = new_x2 - new_x1;
        int new_h = new_y2 - new_y1;

        return img.getSubImage(new_x1, new_y1, new_w, new_h);
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // 模型1  人体检测
        String model1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pose_estimation2_djl\\ssd_512_resnet50_v1_voc.zip";


        // 模型2 关键点   simple_pose_resnet18_v1b.zip/simple_pose_resnet50_v1b.zip
        String model2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pose_estimation2_djl\\simple_pose_resnet50_v1b.zip";


        // 图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pose_estimation2_djl\\ped_vec.jpeg";


        Path imageFile = new File(pic).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);
        PersonDetection personDetection = new PersonDetection(model1);
        DetectedObjects detections = personDetection.predict(image);
        List<DetectedObjects.DetectedObject> persons = detections.items();
        //    List<String> names = new ArrayList<>();
        //    List<Double> prob = new ArrayList<>();
        //    List<BoundingBox> rect = new ArrayList<>();
        int index = 0;

        // 每个人截图
        for (DetectedObjects.DetectedObject person : persons) {

            // 外扩比例 factor = 1, 100%, factor = 0.2, 20%
            // Expand the border of the bounding box by factor=1, 100%; factor=0.2, 20%
            Image subImg = getSubImage(image, person.getBoundingBox(), 0f);

            // 传入模型2进行推理
            Joints joints = new PoseResnetEstimation(model2).predict(subImg);

            // 在抠出的小图中画出关键点
            // Draw the keypoints in the cropped image
            subImg.drawJoints(joints);

            // 弹窗显示每个人的关键点 subImg
            BufferedImage backup = (BufferedImage)subImg.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(backup.getWidth(), backup.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(backup)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);

        }

        System.out.println("识别结果:");
        System.out.println(detections);
    }

}
