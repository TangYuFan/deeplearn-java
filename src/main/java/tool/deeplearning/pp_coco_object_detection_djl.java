package tool.deeplearning;


import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import ai.djl.modality.cv.ImageFactory;

import javax.swing.*;

/**
*   @desc : coco数据集 pp开源模型目标检测,djl 推理
 *
 *      参考连接:
 *      https://www.paddlepaddle.org.cn/hubdetail?name=ssd_vgg16_512_coco2017&en_category=ObjectDetection
 *
 *      模型下载地址:
 *      https://github.com/mymagicpower/AIAS/releases/download/apps/traffic.zip
 *
*   @auth : tyf
*   @date : 2022-06-15  12:17:18
*/
public class pp_coco_object_detection_djl {

    public static class CocoDetection {


        String modelPath;
        String labelPath;
        private CocoDetection(String modelPath,String labelPath) {
            this.modelPath = modelPath;
            this.labelPath = labelPath;
        }

        public DetectedObjects predict(Image img)
                throws IOException, ModelException, TranslateException {
            img.getWrappedImage();

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .optEngine("PaddlePaddle")
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optModelName("inference")
                            .optTranslator(new TrafficTranslator(labelPath))
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel model = ModelZoo.loadModel(criteria)) {
                try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                    DetectedObjects objects = predictor.predict(img);
                    return objects;
                }
            }
        }

        private static final class TrafficTranslator implements Translator<Image, DetectedObjects> {

            private List<String> className;

            String label;
            TrafficTranslator(String label) {
                this.label = label;
            }

            @Override
            public void prepare(TranslatorContext ctx) throws IOException {
                Model model = ctx.getModel();
                className = Utils.readLines(new File(label).toPath(), true);
            }

            @Override
            public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
                return processImageOutput(list);
            }

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
                array = NDImageUtils.resize(array, 512, 512);
                if (!array.getDataType().equals(DataType.FLOAT32)) {
                    array = array.toType(DataType.FLOAT32, false);
                }
                //      array = array.div(255f);
                NDArray mean = ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(1, 1, 3));
                NDArray std = ctx.getNDManager().create(new float[] {1f, 1f, 1f}, new Shape(1, 1, 3));
                array = array.sub(mean);
                array = array.div(std);

                array = array.transpose(2, 0, 1); // HWC -> CHW RGB
                array = array.expandDims(0);

                return new NDList(array);
            }

            @Override
            public Batchifier getBatchifier() {
                return null;
            }

            DetectedObjects processImageOutput(NDList list) {
                NDArray result = list.singletonOrThrow();
                float[] probabilities = result.get(":,1").toFloatArray();
                List<String> names = new ArrayList<>();
                List<Double> prob = new ArrayList<>();
                List<BoundingBox> boxes = new ArrayList<>();
                for (int i = 0; i < probabilities.length; i++) {
                    if (probabilities[i] < 0.55) continue;

                    float[] array = result.get(i).toFloatArray();
                    //        [  0.          0.9627503 172.78745    22.62915   420.2703    919.949    ]
                    //        [  0.          0.8364255 497.77234   161.08307   594.4088    480.63745  ]
                    //        [  0.          0.7247823  94.354065  177.53668   169.24417   429.2456   ]
                    //        [  0.          0.5549363  18.81821   209.29712   116.40645   471.8595   ]
                    // 1-person 行人 2-bicycle 自行车 3-car 小汽车 4-motorcycle 摩托车 6-bus 公共汽车 8-truck 货车

                    int index = (int) array[0];
                    names.add(className.get(index));
                    // array[0] category_id
                    // array[1] confidence
                    // bbox
                    // array[2]
                    // array[3]
                    // array[4]
                    // array[5]
                    prob.add((double) probabilities[i]);
                    // x, y , w , h
                    // dt['left'], dt['top'], dt['right'], dt['bottom'] = clip_bbox(bbox, org_img_width,
                    // org_img_height)
                    boxes.add(new Rectangle(array[2], array[3], array[4] - array[2], array[5] - array[3]));
                }
                return new DetectedObjects(names, prob, boxes);
            }
        }

        private static void saveBoundingBoxImage(
                Image img, DetectedObjects detection, String name, String path) throws IOException {
            // Make image copy with alpha channel because original image was jpg
            img.drawBoundingBoxes(detection);
            Path outputDir = Paths.get(path);
            Files.createDirectories(outputDir);
            Path imagePath = outputDir.resolve(name);
            // OpenJDK can't save jpg with alpha channel
            img.save(Files.newOutputStream(imagePath), "png");
        }
    }



    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 模型
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_coco_object_detection_djl\\traffic.zip";

        // 标签
        String labelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_coco_object_detection_djl\\label_file.txt";

        // 图片
        String imagePath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_coco_object_detection_djl\\ped_vec.jpeg";



        Path imageFile = new File(imagePath).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);
        // 转为图片用于标注
        BufferedImage backup = (BufferedImage)image.getWrappedImage();

        // 加载模型和类别
        CocoDetection cocoDetection = new CocoDetection(modelPath,labelPath);

        // 推理
        DetectedObjects detections = cocoDetection.predict(image);

        List<DetectedObjects.DetectedObject> items = detections.items();
        for (DetectedObjects.DetectedObject item : items) {

            System.out.println("类别:"+item.getClassName()+",得分:"+item.getProbability()+",坐标:"+item.getBoundingBox());
            // 标注
            Graphics2D g2d = backup.createGraphics();
            g2d.setColor(Color.RED);
            g2d.setStroke(new BasicStroke(2));
            // 边框
            g2d.drawRect(
                    Double.valueOf(item.getBoundingBox().getBounds().getX() * backup.getWidth()).intValue() ,
                    Double.valueOf(item.getBoundingBox().getBounds().getY()  * backup.getHeight()).intValue() ,
                    Double.valueOf(item.getBoundingBox().getBounds().getWidth()  * backup.getWidth()).intValue() ,
                    Double.valueOf(item.getBoundingBox().getBounds().getHeight() * backup.getHeight()).intValue()
            );            // 类别
            g2d.drawString(
                    item.getClassName(),
                    Double.valueOf(item.getBoundingBox().getBounds().getX()  * backup.getWidth()).intValue() ,
                    Double.valueOf(item.getBoundingBox().getBounds().getY() * backup.getHeight()).intValue()
            );
            // 释放资源
            g2d.dispose();

        }


        // 弹窗显示
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
