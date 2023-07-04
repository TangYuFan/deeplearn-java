package tool.deeplearning;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
*   @desc : 人体姿态估计（17关键点检测）  djl 推理
*   @auth : tyf
*   @date : 2022-06-13  20:53:27
*/
public class pose_estimation_djl {


    public static class PoseEstimation {

        List<Image> pics = new ArrayList<>();
        private PoseEstimation() {}

        public List<Joints> predict(String imagePath) throws IOException, ModelException, TranslateException {
            Path imageFile = new File(imagePath).toPath();
            Image img = ImageFactory.getInstance().fromFile(imageFile);

            List<Image> people = predictPeopleInImage(img);

            if (people.isEmpty()) {
                return Collections.emptyList();
            }

            return predictJointsForPeople(people);
        }


        // 先检测每个人
        private static List<Image> predictPeopleInImage(Image img)
                throws MalformedModelException, ModelNotFoundException, IOException,
                TranslateException {

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .optApplication(Application.CV.OBJECT_DETECTION)
                            .setTypes(Image.class, DetectedObjects.class)
                            .optFilter("size", "512")
                            .optFilter("backbone", "resnet50")
                            .optFilter("flavor", "v1")
                            .optFilter("dataset", "voc")
                            .optEngine("MXNet")
                            .optProgress(new ProgressBar())
                            .build();

            DetectedObjects detectedObjects;
            try (ZooModel<Image, DetectedObjects> ssd = criteria.loadModel()) {
                try (Predictor<Image, DetectedObjects> predictor = ssd.newPredictor()) {
                    detectedObjects = predictor.predict(img);
                }
            }

            List<DetectedObjects.DetectedObject> items = detectedObjects.items();
            List<Image> people = new ArrayList<>();
            for (DetectedObjects.DetectedObject item : items) {
                if ("person".equals(item.getClassName())) {
                    Rectangle rect = item.getBoundingBox().getBounds();
                    int width = img.getWidth();
                    int height = img.getHeight();
                    people.add(
                            img.getSubImage(
                                    (int) (rect.getX() * width),
                                    (int) (rect.getY() * height),
                                    (int) (rect.getWidth() * width),
                                    (int) (rect.getHeight() * height)));
                }
            }
            return people;
        }

        private List<Joints> predictJointsForPeople(List<Image> people)
                throws MalformedModelException, ModelNotFoundException, IOException,
                TranslateException {

            Criteria<Image, Joints> criteria =
                    Criteria.builder()
                            .optApplication(Application.CV.POSE_ESTIMATION)
                            .setTypes(Image.class, Joints.class)
                            .optFilter("backbone", "resnet18")
                            .optFilter("flavor", "v1b")
                            .optFilter("dataset", "imagenet")
                            .build();

            List<Joints> allJoints = new ArrayList<>();
            try (ZooModel<Image, Joints> pose = criteria.loadModel();
                 Predictor<Image, Joints> predictor = pose.newPredictor()) {
                int count = 0;
                for (Image person : people) {
                    Joints joints = predictor.predict(person);
                    saveJointsImage(person, joints, count++);
                    allJoints.add(joints);
                }
            }
            return allJoints;
        }

        private void saveJointsImage(Image img, Joints joints, int count) throws IOException {

            img.drawJoints(joints);

            this.pics.add(img);
        }

        public void show(){


            // 显示每个人的17关键点
            pics.stream().forEach(out->{

                BufferedImage pic = (BufferedImage) out.getWrappedImage();
                JFrame frame = new JFrame("Image");
                frame.setSize(pic.getWidth(), pic.getHeight());
                JPanel panel = new JPanel();
                panel.add(new JLabel(new ImageIcon(pic)));
                frame.getContentPane().add(panel);
                frame.setVisible(true);
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setResizable(false);

            });

        }

    }



    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 自动下载模型 mxnet 模型

        // 目标检测模型的类别
        // https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/mxnet/classes_voc.txt ...

        // 目标检测模型
        // https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/mxnet/ssd/0.0.1/ssd_512_resnet50_v1-symbol.json ...
        // https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/mxnet/ssd/0.0.1/ssd_512_resnet50_v1_voc-0000.params.gz ...

        // 姿态估计模型
        // https://mlrepo.djl.ai/model/cv/pose_estimation/ai/djl/mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b-symbol.json ...
        // https://mlrepo.djl.ai/model/cv/pose_estimation/ai/djl/mxnet/simple_pose/0.0.1/simple_pose_resnet18_v1b-0000.params.gz ...


        // 测试图片
        String imagePath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pose_estimation_djl\\pose_soccer.png";


        PoseEstimation estimation = new PoseEstimation();
        List<Joints> joints = estimation.predict(imagePath);

        System.out.println("预测结果:");
        System.out.println(joints);

        // 显示
        estimation.show();


    }



}
