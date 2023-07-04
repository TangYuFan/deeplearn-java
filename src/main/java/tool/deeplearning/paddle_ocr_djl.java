package tool.deeplearning;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.zoo.cv.objectdetection.BoundFinder;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.TranslateException;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

/**
 *  @Desc: paddle ocr 字符识别  djl 推理
 *
 *
 *  @Date: 2022-06-12 11:17:19
 *  @auth: TYF
 */
public class paddle_ocr_djl {


    /**
     *  @Desc: det
     *  @Date: 2022-06-12 11:21:07
     *  @auth: TYF
     */
    public static class PpWordDetectionTranslator implements NoBatchifyTranslator<Image, DetectedObjects> {

        private final int maxLength;

        /**
         * Creates the {@link PpWordDetectionTranslator} instance.
         *
         * @param arguments the arguments for the translator
         */
        public PpWordDetectionTranslator(Map<String, ?> arguments) {
            maxLength = ArgumentsUtil.intValue(arguments, "maxLength", 960);
        }

        /** {@inheritDoc} */
        @Override
        public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
            NDArray result = list.singletonOrThrow();
            result = result.squeeze().mul(255f).toType(DataType.UINT8, true).neq(0);
            boolean[] flattened = result.toBooleanArray();
            Shape shape = result.getShape();
            int w = (int) shape.get(0);
            int h = (int) shape.get(1);
            boolean[][] grid = new boolean[w][h];
            IntStream.range(0, flattened.length)
                    .parallel()
                    .forEach(i -> grid[i / h][i % h] = flattened[i]);
            List<BoundingBox> boxes = new BoundFinder(grid).getBoxes();
            List<String> names = new ArrayList<>();
            List<Double> probs = new ArrayList<>();
            int boxSize = boxes.size();
            for (int i = 0; i < boxSize; i++) {
                names.add("word");
                probs.add(1.0);
            }
            return new DetectedObjects(names, probs, boxes);
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray img = input.toNDArray(ctx.getNDManager());
            int h = input.getHeight();
            int w = input.getWidth();
            int[] hw = scale(h, w, maxLength);

            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img);
            img =
                    NDImageUtils.normalize(
                            img,
                            new float[] {0.485f, 0.456f, 0.406f},
                            new float[] {0.229f, 0.224f, 0.225f});
            img = img.expandDims(0);
            return new NDList(img);
        }

        private int[] scale(int h, int w, int max) {
            int localMax = Math.max(h, w);
            float scale = 1.0f;
            if (max < localMax) {
                scale = max * 1.0f / localMax;
            }
            // paddle model only take 32-based size
            return resize32(h * scale, w * scale);
        }

        private int[] resize32(double h, double w) {
            double min = Math.min(h, w);
            if (min < 32) {
                h = 32.0 / min * h;
                w = 32.0 / min * w;
            }
            int h32 = (int) h / 32;
            int w32 = (int) w / 32;
            return new int[] {h32 * 32, w32 * 32};
        }
    }


    /**
     *  @Desc: rec
     *  @Date: 2022-06-12 11:21:11
     *  @auth: TYF
     */
    public static class PpWordRecognitionTranslator implements NoBatchifyTranslator<Image, String> {

        private List<String> table;
        String KEYS;
        public PpWordRecognitionTranslator(String KEYS) {
            this.KEYS = KEYS;
        }
        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            // "ppocr_keys_v1.txt"
            try (InputStream is = ctx.getModel().getArtifact(KEYS).openStream()) {
                table = Utils.readLines(is, true);
                table.add(0, "blank");
                table.add("");
            }

        }
        @Override
        public String processOutput(TranslatorContext ctx, NDList list) throws Exception {
            StringBuilder sb = new StringBuilder();
            NDArray tokens = list.singletonOrThrow();
            long[] indices = tokens.get(new long[]{0L}).argMax(1).toLongArray();
            for (int i = 0; i < indices.length; i++) {
                if (i!=indices.length-1&&indices[i]==indices[i+1]&&indices[i]!=0) indices[i]=0;
            }
            int lastIdx = 0;
            for(int i = 0; i < indices.length; ++i) {
                if (indices[i] > 0L && (i <= 0 || indices[i] != (long)lastIdx)) {
                    sb.append((String)this.table.get((int)indices[i]));
                }
            }
            return sb.toString();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray img = input.toNDArray(ctx.getNDManager());
            int[] hw = this.resize32((double)input.getWidth());
            img = NDImageUtils.resize(img, hw[1], hw[0]);
            img = NDImageUtils.toTensor(img).sub(0.5F).div(0.5F);
            img = img.expandDims(0);
            NDList ndArrays = new NDList(new NDArray[]{img});
            return ndArrays;

        }

        private int[] resize32(double w) {
            int width = (int)Math.max(48.0, w) / 48 * 48;
            return new int[]{48, width};
        }
    }



    public static class PytorchOCRUtil {

        static String DET_PATH;
        static String REC_PATH;
        static String KEYS;

        ZooModel<Image, DetectedObjects> detectionModel;
        ZooModel<Image, String> recognitionModel;

        // 保存所有识别结果
        List<String> res1 = new ArrayList<>();
        // 保存所有边框
        List<int[]> res2 = new ArrayList<>();
        // 保存输入图片
        Image imgbackup;

        public PytorchOCRUtil(String DET_PATH,String REC_PATH,String KEYS) throws ModelNotFoundException, MalformedModelException, IOException {

            this.DET_PATH = DET_PATH;
            this.REC_PATH = REC_PATH;
            this.KEYS = KEYS;

            System.out.println(this.DET_PATH);
            System.out.println(this.REC_PATH);
            System.out.println(this.KEYS);

            /**
             * DET模型构建
             */
            Criteria<Image, DetectedObjects> criteria_det = Criteria.builder()
                    .setTypes(Image.class, DetectedObjects.class)
                    .optModelPath(new File(DET_PATH).toPath())
                    .optTranslator(new PpWordDetectionTranslator(new ConcurrentHashMap<String, String>()))
                    .build();

            /**
             * REC模型构建
             */
            Criteria<Image, String> criteria_rec = Criteria.builder()
                    .setTypes(Image.class, String.class)
                    .optModelPath(new File(REC_PATH).toPath())
                    .optTranslator(new PpWordRecognitionTranslator(KEYS))
                    .optProgress(new ProgressBar()).build();

            // 模型加载
            detectionModel = criteria_det.loadModel();
            recognitionModel = criteria_rec.loadModel();
        }

        public void ocr(String path) throws IOException, TranslateException {
            /**
             * 两个Predictor生成
             */
            Predictor<Image, DetectedObjects> detector = detectionModel.newPredictor();
            Predictor<Image, String> recognizer = recognitionModel.newPredictor();

            /**
             * 加载图片
             */
            Image img = ImageFactory.getInstance().fromFile(Paths.get(path));
            this.imgbackup = img.duplicate();

            /**
             * 文字区域检测
             */
            DetectedObjects detectedObj = detector.predict(img);
            Image newImage = img.duplicate();
            newImage.drawBoundingBoxes(detectedObj);
            newImage.getWrappedImage();

            /**
             * 获取分割出来的文字区域列表,并识别返回文本
             */
            List<DetectedObjects.DetectedObject> boxes = detectedObj.items();


            System.out.println("文本区域个数:"+boxes.size());
            for (int i = 0; i < boxes.size(); i++) {
                Image subImage = getSubImage(img, boxes.get(i).getBoundingBox());
                subImage.getWrappedImage();
                String predict = recognizer.predict(subImage);
                res1.add(predict);
            }
        }

        // 截取图片区域
        public Image getSubImage(Image img, BoundingBox box) {
            Rectangle rect = box.getBounds();
            double[] extended = extendRect(rect.getX(), rect.getY(), rect.getWidth(), rect.getHeight());
            int width = img.getWidth();
            int height = img.getHeight();
            int[] recovered = {
                    (int) (extended[0] * width),
                    (int) (extended[1] * height),
                    (int) (extended[2] * width),
                    (int) (extended[3] * height)
            };
            res2.add(recovered);
            // 在原始图片进行标注
            return img.getSubImage(recovered[0], recovered[1], recovered[2], recovered[3]);
        }

        public static double[] extendRect(double xmin, double ymin, double width, double height) {
            double centerx = xmin + width / 2;
            double centery = ymin + height / 2;
            if (width > height) {
                width += height * 1.6;
                height *= 2.6;
            } else {
                height += width * 1.6;
                width *= 2.6;
            }
            double newX = centerx - width / 2 < 0 ? 0 : centerx - width / 2;
            double newY = centery - height / 2 < 0 ? 0 : centery - height / 2;
            double newWidth = newX + width > 1 ? 1 - newX : width;
            double newHeight = newY + height > 1 ? 1 - newY : height;
            return new double[] {newX, newY, newWidth, newHeight};
        }
        public static Image rotateImg(Image image) {
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray rotated = NDImageUtils.rotate90(image.toNDArray(manager), 1);
                return ImageFactory.getInstance().fromNDArray(rotated);
            }
        }


        // 显示
        public void show(){


            // 图片 imgbackup
            BufferedImage out = (BufferedImage)imgbackup.getWrappedImage();

            // 边框 res2
            for (int i = 0; i < res2.size(); i++) {

                // xywh
                int[] box = res2.get(i);

                Graphics graphics = out.getGraphics();
                graphics.setColor(Color.RED);
                graphics.drawRect(box[0], box[1], box[2], box[3]);

                // 添加文本区域编号
                graphics.setFont(new Font("Arial", Font.BOLD, 14));
                graphics.drawString(String.valueOf(i+1),box[0], box[1]-5);

            }

            // 文字 res1
            for (int i = 0; i < res1.size(); i++) {

                String text = res1.get(i);
                System.out.println("文本区域编号"+(i+1)+",文本:"+text);
            }


            // 弹窗显示
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


        // 检测
        String DET_PATH = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\paddle_ocr_djl\\ch_ptocr_det_infer.pt";

        // 识别
        String REC_PATH = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\paddle_ocr_djl\\ch_ptocr_rec_infer.pt";


        // 字库
        String KEYS = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\paddle_ocr_djl\\ppocr_keys_v1.txt";


        // 待识别图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\paddle_ocr_djl\\baidu.png";

        // 识别
        PytorchOCRUtil ocrUtil = new PytorchOCRUtil(DET_PATH,REC_PATH,KEYS);
        ocrUtil.ocr(pic);


        // 显示结果
        ocrUtil.show();

    }






}
