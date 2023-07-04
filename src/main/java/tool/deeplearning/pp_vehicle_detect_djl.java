package tool.deeplearning;


import ai.djl.Model;
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
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import javax.swing.*;

/**
*   @desc : 车辆检测 , pp开源模型 , djl 推理
 *
*   @auth : tyf
*   @date : 2022-06-15  10:00:03
*/
public class pp_vehicle_detect_djl {



    public static class VehicleDetect {

        public VehicleDetect() {}

        public Criteria<Image, DetectedObjects> criteria(String modelPath) {

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .optEngine("PaddlePaddle")
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optModelName("inference")
                            .optTranslator(new VehicleTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }

        private final class VehicleTranslator implements Translator<Image, DetectedObjects> {
            private int width;
            private int height;
            private List<String> className;

            VehicleTranslator() {}

            @Override
            public void prepare(TranslatorContext ctx) throws IOException {
                Model model = ctx.getModel();
                try (InputStream is = model.getArtifact("label_file.txt").openStream()) {
                    className = Utils.readLines(is, true);
                    //            classes.add(0, "blank");
                    //            classes.add("");
                }
            }

            @Override
            public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
                return processImageOutput(list);
            }

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
                array = NDImageUtils.resize(array, 608, 608);
                if (!array.getDataType().equals(DataType.FLOAT32)) {
                    array = array.toType(DataType.FLOAT32, false);
                }
                array = array.div(255f);
                NDArray mean =
                        ctx.getNDManager().create(new float[] {0.485f, 0.456f, 0.406f}, new Shape(1, 1, 3));
                NDArray std =
                        ctx.getNDManager().create(new float[] {0.229f, 0.224f, 0.225f}, new Shape(1, 1, 3));
                array = array.sub(mean);
                array = array.div(std);

                array = array.transpose(2, 0, 1); // HWC -> CHW RGB
                array = array.expandDims(0);
                width = input.getWidth();
                height = input.getHeight();
                NDArray imageSize = ctx.getNDManager().create(new int[] {height, width});
                imageSize = imageSize.toType(DataType.INT32, false);

                imageSize = imageSize.expandDims(0);

                return new NDList(array, imageSize);
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
                    float[] array = result.get(i).toFloatArray();
                    //        [  0.          0.9627503 172.78745    22.62915   420.2703    919.949    ]
                    //        [  0.          0.8364255 497.77234   161.08307   594.4088    480.63745  ]
                    //        [  0.          0.7247823  94.354065  177.53668   169.24417   429.2456   ]
                    //        [  0.          0.5549363  18.81821   209.29712   116.40645   471.8595   ]
                    names.add(className.get((int) array[0]));
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
                    boxes.add(
                            new Rectangle(
                                    array[2] / width,
                                    array[3] / height,
                                    (array[4] - array[2]) / width,
                                    (array[5] - array[3]) / height));
                }
                return new DetectedObjects(names, prob, boxes);
            }
        }
    }


    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // 模型  =
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_vehicle_detect_djl\\vehicle.zip";

        // 测试图片
        String picPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_vehicle_detect_djl\\vehicle.png";


        Path imageFile = new File(picPath).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);

        Criteria<Image, DetectedObjects> criteria = new VehicleDetect().criteria(modelPath);

        try (ZooModel model = ModelZoo.loadModel(criteria);
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detections = predictor.predict(image);

            List<DetectedObjects.DetectedObject> items = detections.items();
            List<String> names = new ArrayList<>();
            List<Double> prob = new ArrayList<>();
            List<BoundingBox> rect = new ArrayList<>();
            for (DetectedObjects.DetectedObject item : items) {
                if (item.getProbability() < 0.55) {
                    continue;
                }
                names.add(item.getClassName());
                prob.add(item.getProbability());
                rect.add(item.getBoundingBox());
            }

            detections = new DetectedObjects(names, prob, rect);


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



            System.out.println("识别结果:");
            System.out.println(detections);

        }
    }


}
