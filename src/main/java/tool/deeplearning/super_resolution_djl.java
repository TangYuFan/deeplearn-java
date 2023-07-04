package tool.deeplearning;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import javax.swing.*;


/**
*   @desc : 超分辨率 esrgan-tf2 模型 ,  djl 推理
 *
 *         这个是 djl 官方例子,模型下载链接:
 *         https://github.com/mymagicpower/AIAS/releases/download/apps/esrgan-tf2_1.zip
 *
*   @auth : tyf
*   @date : 2022-06-15  11:05:10
*/
public class super_resolution_djl {



    public static class SuperResolution {
        String modelPath;
        public SuperResolution(String modelPath) {
            this.modelPath = modelPath;
        }

        public Image predict(Image img)
                throws IOException, ModelException, TranslateException {
            Criteria<Image, Image> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, Image.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optOption("Tags", "serve")
                            .optEngine("TensorFlow") // Use TensorFlow engine
                            .optOption("SignatureDefKey", "serving_default")
                            .optTranslator(new SuperResolutionTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<Image, Image> model = criteria.loadModel();
                 Predictor<Image, Image> enhancer = model.newPredictor()) {
                return enhancer.predict(img);
            }
        }

        public List<Image> predict(List<Image> inputImages)
                throws IOException, ModelException, TranslateException {
            Criteria<Image, Image> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, Image.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optOption("Tags", "serve")
                            .optEngine("TensorFlow") // Use TensorFlow engine
                            .optOption("SignatureDefKey", "serving_default")
                            .optTranslator(new SuperResolutionTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<Image, Image> model = criteria.loadModel();
                 Predictor<Image, Image> enhancer = model.newPredictor()) {
                return enhancer.batchPredict(inputImages);
            }
        }
    }


    public static class SuperResolutionTranslator implements Translator<Image, Image> {

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDManager manager = ctx.getNDManager();
            return new NDList(input.toNDArray(manager).toType(DataType.FLOAT32, false));
        }

        @Override
        public Image processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDArray output = list.get(0).clip(0, 255).toType(DataType.UINT8, false);
            return ImageFactory.getInstance().fromNDArray(output.squeeze());
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }



    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 模型
        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\super_resolution_djl\\esrgan-tf2_1.zip";


        // 图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\super_resolution_djl\\srgan.png";



        Path imageFile = new File(pic).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);

        // 推理
        SuperResolution enhancer = new SuperResolution(model);
        Image img = enhancer.predict(image);

        // 弹窗显示
        BufferedImage out = (BufferedImage)img.getWrappedImage();
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
