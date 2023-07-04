package tool.deeplearning;


import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
*   @desc : meta-ai sam  对所有对象进行分割 , 使用djl进行推理
 *
 *
*   @auth : tyf
*   @date : 2022-06-12  10:34:15
*/
public class metaai_sam_test_djl {


    /**
    *   @desc : 定义记录行
    *   @auth : tyf
    *   @date : 2022-06-12  10:44:45
    */
    public static class SamRawOutput{

        NDArray iouPred;
        NDArray lowResLogits;
        NDArray mask;
        public SamRawOutput(NDArray iouPred, NDArray lowResLogits, NDArray mask) {
            this.lowResLogits = lowResLogits;
            this.iouPred = iouPred;
            this.mask = mask;
        }
        public void close() {
            iouPred.close();
            lowResLogits.close();
            mask.close();
        }
    }


    /**
    *   @desc : 定义模型预处理和后处理
    *   @auth : tyf
    *   @date : 2022-06-12  10:44:56
    */
    public static class SamTranslator implements Translator<Image, SamRawOutput> {

        private final Builder builder;

        public SamTranslator(Builder builder) {
            this.builder = builder;
        }

        public static Builder builder() {
            return new Builder();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            // Convert image to NDArray
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            // Apply build transforms
            for (Transform transform : builder.transforms) {
                array = transform.transform(array);
            }
            return new NDList(array);
        }

        @Override
        public SamRawOutput processOutput(TranslatorContext ctx, NDList list) {
            // Note: this causes a memory leak if the NDArrays are not closed
            list.detach();
            NDArray iouPred = list.get(0);
            NDArray lowResLogits = list.get(1);
            NDArray mask = list.get(2);
            return new SamRawOutput(iouPred, lowResLogits, mask);
        }

        public static class Builder {
            public List<Transform> transforms;

            public Builder() {
                this.transforms = new java.util.ArrayList<Transform>();
            }

            public Builder addTransform(Transform transform) {
                this.transforms.add(transform);
                return this;
            }

            public SamTranslator build() {
                return new SamTranslator(this);
            }
        }
    }


    /**
    *   @desc : 模型处理器
    *   @auth : tyf
    *   @date : 2022-06-12  10:45:08
    */
    public static class Sam {
        private final Predictor<Image, SamRawOutput> predictor;

        public Sam(String path) {
            Translator<Image, SamRawOutput> translator = SamTranslator.builder()
                    .addTransform(new Resize(1024))
                    .addTransform(new ToTensor())
                    // Normalize with mean and std of ImageNet / 255
                    .addTransform(new Normalize(new float[]{0.485f, 0.456f, 0.406f}, new float[]{0.229f, 0.224f, 0.225f}))
                    .build();
            Criteria<Image, SamRawOutput> criteria = Criteria.builder()
                    .setTypes(Image.class, SamRawOutput.class)
                    .optModelPath(new File(path).toPath())
                    .optTranslatorFactory(new SemanticSegmentationTranslatorFactory())
                    .optEngine("PyTorch")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .optDevice(Device.cpu())
                    .build();
            ZooModel<Image, SamRawOutput> model;
            try {
                model = criteria.loadModel();
            } catch (IOException | ModelNotFoundException | MalformedModelException e) {
                throw new RuntimeException(e);
            }
            this.predictor = model.newPredictor();
        }

        public SamRawOutput predict(Image image) {
            try {
                return predictor.predict(image);
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
    }


    public static void main(String[] args) throws Exception{



        // 模型地址
        String path = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\metaai_sam_test_djl\\sam_vit_b.pt";

        // 预测图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\metaai_sam_test_djl\\truck.jpg";

        // 预测器
        Sam sam = new Sam(path);


        try {
            ImageFactory factory = ImageFactory.getInstance();
            Image img = factory.fromFile(new File(pic).toPath());

            // 对所有对象进行分割
            SamRawOutput output = sam.predict(img);

            // 输出的mask
            NDArray mask = output.mask;

            Image imgOut = img.duplicate().resize(1024, 1024, true);
            int height = (int) mask.getShape().get(0);
            int width = (int) mask.getShape().get(1);
            int[] pixels = new int[width * height];

            // We convert the mask to an image to visualize it
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int red;
                    int green;
                    int blue;
                    int opacity;
                    // 根据mask阈值判断
                    if (mask.getBoolean(h, w)) {
                        red = 0;
                        green = 0;
                        blue = 255;
                        opacity = 120;
                    } else {
                        red = 0;
                        green = 0;
                        blue = 0;
                        opacity = 0;
                    }
                    int color = opacity << 24 | red << 16 | green << 8 | blue;
                    pixels[h * width + w] = color; // black
                }
            }

            Image maskImage = factory.fromPixels(pixels, width, height);

            // 写入最终的结果
            imgOut.drawImage(maskImage, true);


            // 弹窗显示
            BufferedImage out = (BufferedImage)imgOut.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(out.getWidth(), out.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(out)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);


            output.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }


}
