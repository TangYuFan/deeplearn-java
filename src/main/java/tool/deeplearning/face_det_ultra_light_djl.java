package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *   @desc : Ultra-Light-Fast 模型人脸检测（5关键点）, djl 推理
 *
 *          模型pytorch权重:
 *          https://resources.djl.ai/test-models/pytorch/ultranet.zip
 *
 *   @auth : tyf
 *   @date : 2022-06-13  11:00:44
 */
public class face_det_ultra_light_djl {

    public static class FaceDetectionTranslator implements Translator<Image, DetectedObjects> {

        private double confThresh;
        private double nmsThresh;
        private int topK;
        private double[] variance;
        private int[][] scales;
        private int[] steps;
        private int width;
        private int height;

        public FaceDetectionTranslator(
                double confThresh,
                double nmsThresh,
                double[] variance,
                int topK,
                int[][] scales,
                int[] steps) {
            this.confThresh = confThresh;
            this.nmsThresh = nmsThresh;
            this.variance = variance;
            this.topK = topK;
            this.scales = scales;
            this.steps = steps;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            width = input.getWidth();
            height = input.getHeight();
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            array = array.transpose(2, 0, 1).flip(0); // HWC -> CHW RGB -> BGR
            // The network by default takes float32
            if (!array.getDataType().equals(DataType.FLOAT32)) {
                array = array.toType(DataType.FLOAT32, false);
            }
            NDArray mean =
                    ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(3, 1, 1));
            array = array.sub(mean);
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            NDManager manager = ctx.getNDManager();
            double scaleXY = variance[0];
            double scaleWH = variance[1];

            NDArray prob = list.get(1).get(":, 1:");
            prob =
                    NDArrays.stack(
                            new NDList(
                                    prob.argMax(1).toType(DataType.FLOAT32, false),
                                    prob.max(new int[] {1})));

            NDArray boxRecover = boxRecover(manager, width, height, scales, steps);
            NDArray boundingBoxes = list.get(0);
            NDArray bbWH = boundingBoxes.get(":, 2:").mul(scaleWH).exp().mul(boxRecover.get(":, 2:"));
            NDArray bbXY =
                    boundingBoxes
                            .get(":, :2")
                            .mul(scaleXY)
                            .mul(boxRecover.get(":, 2:"))
                            .add(boxRecover.get(":, :2"))
                            .sub(bbWH.mul(0.5f));

            boundingBoxes = NDArrays.concat(new NDList(bbXY, bbWH), 1);

            NDArray landms = list.get(2);
            landms = decodeLandm(landms, boxRecover, scaleXY);

            // filter the result below the threshold
            NDArray cutOff = prob.get(1).gt(confThresh);
            boundingBoxes = boundingBoxes.transpose().booleanMask(cutOff, 1).transpose();
            landms = landms.transpose().booleanMask(cutOff, 1).transpose();
            prob = prob.booleanMask(cutOff, 1);

            // start categorical filtering
            long[] order = prob.get(1).argSort().get(":" + topK).toLongArray();
            prob = prob.transpose();
            List<String> retNames = new ArrayList<>();
            List<Double> retProbs = new ArrayList<>();
            List<BoundingBox> retBB = new ArrayList<>();

            Map<Integer, List<BoundingBox>> recorder = new ConcurrentHashMap<>();

            for (int i = order.length - 1; i >= 0; i--) {
                long currMaxLoc = order[i];
                float[] classProb = prob.get(currMaxLoc).toFloatArray();
                int classId = (int) classProb[0];
                double probability = classProb[1];

                double[] boxArr = boundingBoxes.get(currMaxLoc).toDoubleArray();
                double[] landmsArr = landms.get(currMaxLoc).toDoubleArray();
                Rectangle rect = new Rectangle(boxArr[0], boxArr[1], boxArr[2], boxArr[3]);
                List<BoundingBox> boxes = recorder.getOrDefault(classId, new ArrayList<>());
                boolean belowIoU = true;
                for (BoundingBox box : boxes) {
                    if (box.getIoU(rect) > nmsThresh) {
                        belowIoU = false;
                        break;
                    }
                }
                if (belowIoU) {
                    List<Point> keyPoints = new ArrayList<>();
                    for (int j = 0; j < 5; j++) { // 5 face landmarks
                        double x = landmsArr[j * 2];
                        double y = landmsArr[j * 2 + 1];
                        keyPoints.add(new Point(x * width, y * height));
                    }
                    Landmark landmark =
                            new Landmark(boxArr[0], boxArr[1], boxArr[2], boxArr[3], keyPoints);

                    boxes.add(landmark);
                    recorder.put(classId, boxes);
                    String className = "Face"; // classes.get(classId)
                    retNames.add(className);
                    retProbs.add(probability);
                    retBB.add(landmark);
                }
            }

            return new DetectedObjects(retNames, retProbs, retBB);
        }

        private NDArray boxRecover(
                NDManager manager, int width, int height, int[][] scales, int[] steps) {
            int[][] aspectRatio = new int[steps.length][2];
            for (int i = 0; i < steps.length; i++) {
                int wRatio = (int) Math.ceil((float) width / steps[i]);
                int hRatio = (int) Math.ceil((float) height / steps[i]);
                aspectRatio[i] = new int[] {hRatio, wRatio};
            }

            List<double[]> defaultBoxes = new ArrayList<>();

            for (int idx = 0; idx < steps.length; idx++) {
                int[] scale = scales[idx];
                for (int h = 0; h < aspectRatio[idx][0]; h++) {
                    for (int w = 0; w < aspectRatio[idx][1]; w++) {
                        for (int i : scale) {
                            double skx = i * 1.0 / width;
                            double sky = i * 1.0 / height;
                            double cx = (w + 0.5) * steps[idx] / width;
                            double cy = (h + 0.5) * steps[idx] / height;
                            defaultBoxes.add(new double[] {cx, cy, skx, sky});
                        }
                    }
                }
            }

            double[][] boxes = new double[defaultBoxes.size()][defaultBoxes.get(0).length];
            for (int i = 0; i < defaultBoxes.size(); i++) {
                boxes[i] = defaultBoxes.get(i);
            }
            return manager.create(boxes).clip(0.0, 1.0);
        }

        // decode face landmarks, 5 points per face
        private NDArray decodeLandm(NDArray pre, NDArray priors, double scaleXY) {
            NDArray point1 =
                    pre.get(":, :2").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
            NDArray point2 =
                    pre.get(":, 2:4").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
            NDArray point3 =
                    pre.get(":, 4:6").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
            NDArray point4 =
                    pre.get(":, 6:8").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
            NDArray point5 =
                    pre.get(":, 8:10").mul(scaleXY).mul(priors.get(":, 2:")).add(priors.get(":, :2"));
            return NDArrays.concat(new NDList(point1, point2, point3, point4, point5), 1);
        }
    }

    public static class LightFaceDetection {

        String model;
        Image out;

        private LightFaceDetection(String model) {
            this.model = model;
        }

        public DetectedObjects predict(String pic) throws IOException, ModelException, TranslateException {
            Path facePath = new File(pic).toPath();
            Image img = ImageFactory.getInstance().fromFile(facePath);

            double confThresh = 0.85f;
            double nmsThresh = 0.45f;
            double[] variance = {0.1f, 0.2f};
            int topK = 5000;
            int[][] scales = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
            int[] steps = {8, 16, 32, 64};

            FaceDetectionTranslator translator = new FaceDetectionTranslator(confThresh, nmsThresh, variance, topK, scales, steps);

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, DetectedObjects.class)
                            .optModelPath(new File(model).toPath())
                            .optTranslator(translator)
                            .optProgress(new ProgressBar())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .build();

            try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
                try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                    DetectedObjects detection = predictor.predict(img);

                    // 保存标注结果
                    saveBoundingBoxImage(img, detection);
                    return detection;
                }
            }
        }

        private void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {

            img.drawBoundingBoxes(detection);
            this.out = img;
        }


        private void show(){


            // 弹窗显示
            BufferedImage out = (BufferedImage)this.out.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(out.getWidth(), out.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(out)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);

        }
    }

    public static void main(String[] args) throws Exception{

        // 模型
        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_det_ultra_light_djl\\ultranet.pt";

        // 人脸图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_det_ultra_light_djl\\largest_selfie.jpg";

        // 预测
        LightFaceDetection detection = new LightFaceDetection(model);
        detection.predict(pic);

        // 显示
        detection.show();



    }

}
