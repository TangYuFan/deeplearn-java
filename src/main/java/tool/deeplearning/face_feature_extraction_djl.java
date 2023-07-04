package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

/**
*   @desc : 人脸特征提取 512 维, 相似度对比, djl 推理
 *
 *      使用的模型:
 *      https://resources.djl.ai/test-models/pytorch/face_feature.zip
 *
*   @auth : tyf
*   @date : 2022-06-13  11:17:46
*/
public class face_feature_extraction_djl {


    public static class FeatureExtraction {

        String model;
        private FeatureExtraction(String model) {
            this.model = model;
        }

        public float[] predict(String pic) throws IOException, ModelException, TranslateException {

            Path imageFile = new File(pic).toPath();
            Image img = ImageFactory.getInstance().fromFile(imageFile);

            img.getWrappedImage();
            Criteria<Image, float[]> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, float[].class)
                            .optModelPath(new File(model).toPath())
                            .optModelName("face_feature") // specify model file prefix
                            .optTranslator(new FaceFeatureTranslator())
                            .optProgress(new ProgressBar())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .build();

            try (ZooModel<Image, float[]> model = criteria.loadModel()) {
                Predictor<Image, float[]> predictor = model.newPredictor();
                return predictor.predict(img);
            }
        }

        private final class FaceFeatureTranslator implements Translator<Image, float[]> {

            FaceFeatureTranslator() {}

            /** {@inheritDoc} */
            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
                Pipeline pipeline = new Pipeline();
                pipeline
                        // .add(new Resize(160))
                        .add(new ToTensor())
                        .add(
                                new Normalize(
                                        new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                        new float[] {
                                                128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                        }));

                return pipeline.transform(new NDList(array));
            }

            /** {@inheritDoc} */
            @Override
            public float[] processOutput(TranslatorContext ctx, NDList list) {
                NDList result = new NDList();
                long numOutputs = list.singletonOrThrow().getShape().get(0);
                for (int i = 0; i < numOutputs; i++) {
                    result.add(list.singletonOrThrow().get(i));
                }
                float[][] embeddings =
                        result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
                float[] feature = new float[embeddings.length];
                for (int i = 0; i < embeddings.length; i++) {
                    feature[i] = embeddings[i][0];
                }
                return feature;
            }
        }
    }


    public static float calculSimilar(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 模型
        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_feature_extraction_djl\\face_feature.pt";

        // 人脸图片
        String pic1 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_feature_extraction_djl\\kana1.jpg";

        // 人脸图片
        String pic2 = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\face_feature_extraction_djl\\kana2.jpg";


        // 模型
        FeatureExtraction featureExtraction = new FeatureExtraction(model);


        // 对两张图片进行特征提取
        float[] feature1 = featureExtraction.predict(pic1);
        float[] feature2 = featureExtraction.predict(pic2);

        System.out.println(feature1.length);
        System.out.println(feature2.length);

        System.out.println("特征1:"+Arrays.toString(feature1));
        System.out.println("特征2:"+Arrays.toString(feature2));


        // 对比相似度
        float similar = calculSimilar(feature1,feature2);

        System.out.println("相似度:"+similar);

    }

}
