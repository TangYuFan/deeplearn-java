package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

/**
*   @desc : clip 模型图片文本理解（openai） djl 部署, 文本和图片进行比较,计算图文相关性
*   @auth : tyf
*   @date : 2022-06-13  09:45:02
*/
public class clip_image_text_compare_djl {


    public static class ClipModel implements AutoCloseable {

        private ZooModel<NDList, NDList> clip;
        private Predictor<Image, float[]> imageFeatureExtractor;
        private Predictor<String, float[]> textFeatureExtractor;
        private Predictor<Pair<Image, String>, float[]> imgTextComparator;

        public ClipModel(String model,String tokenizer) throws ModelException, IOException {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelPath(new File(model).toPath())
                            .optTranslator(new NoopTranslator())
                            .optEngine("PyTorch")
                            .build();
            clip = criteria.loadModel();
            imageFeatureExtractor = clip.newPredictor(new ImageTranslator());
            textFeatureExtractor = clip.newPredictor(new TextTranslator(tokenizer));
            imgTextComparator = clip.newPredictor(new ImageTextTranslator(tokenizer));
        }

        public float[] extractTextFeatures(String inputs) throws TranslateException {
            return textFeatureExtractor.predict(inputs);
        }

        public float[] extractImageFeatures(Image inputs) throws TranslateException {
            return imageFeatureExtractor.predict(inputs);
        }

        public float[] compareTextAndImage(Image image, String text) throws TranslateException {
            return imgTextComparator.predict(new Pair<>(image, text));
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            imageFeatureExtractor.close();
            textFeatureExtractor.close();
            imgTextComparator.close();
            clip.close();
        }
    }

    public static class ImageTranslator implements NoBatchifyTranslator<Image, float[]> {

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray array = list.singletonOrThrow();
            return array.toFloatArray();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);

            float percent = 224f / Math.min(input.getWidth(), input.getHeight());
            int resizedWidth = Math.round(input.getWidth() * percent);
            int resizedHeight = Math.round(input.getHeight() * percent);

            array =
                    NDImageUtils.resize(
                            array, resizedWidth, resizedHeight, Image.Interpolation.BICUBIC);
            array = NDImageUtils.centerCrop(array, 224, 224);
            array = NDImageUtils.toTensor(array);
            NDArray placeholder = ctx.getNDManager().create("");
            placeholder.setName("module_method:get_image_features");
            return new NDList(array.expandDims(0), placeholder);
        }
    }



    public static class TextTranslator implements NoBatchifyTranslator<String, float[]> {

        HuggingFaceTokenizer tokenizer;

        public TextTranslator(String toknizer) throws IOException {
//            tokenizer = HuggingFaceTokenizer.newInstance("openai/clip-vit-base-patch32");
            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(toknizer));
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding encoding = tokenizer.encode(input);
            NDArray attention = ctx.getNDManager().create(encoding.getAttentionMask());
            NDArray inputIds = ctx.getNDManager().create(encoding.getIds());
            NDArray placeholder = ctx.getNDManager().create("");
            placeholder.setName("module_method:get_text_features");
            return new NDList(inputIds.expandDims(0), attention.expandDims(0), placeholder);
        }
    }


    public static class ImageTextTranslator implements NoBatchifyTranslator<Pair<Image, String>, float[]> {

        private ImageTranslator imgTranslator;
        private TextTranslator txtTranslator;

        public ImageTextTranslator(String tokenizer) throws IOException {
            this.imgTranslator = new ImageTranslator();
            this.txtTranslator = new TextTranslator(tokenizer);
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
            NDArray logitsPerImage = list.get(0);
            return logitsPerImage.toFloatArray();
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Pair<Image, String> input) throws Exception {
            NDList imageInput = imgTranslator.processInput(ctx, input.getKey());
            NDList textInput = txtTranslator.processInput(ctx, input.getValue());
            return new NDList(textInput.get(0), imageInput.get(0), textInput.get(1));
        }
    }




    public static void main(String[] args) throws Exception{

        // 模型
        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\clip_image_text_compare_djl\\clip.pt";


        // 图片
        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\clip_image_text_compare_djl\\2cat.jpg";

        // tokenizer
        String tokenizer = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\clip_image_text_compare_djl\\tokenizer.json";

        // 加载模型
        ClipModel clipModel = new ClipModel(model,tokenizer);

        // 加载图片
        Image imgobj = ImageFactory.getInstance().fromUrl(new File(img).toURL());

        // 文本
        String text1 = "A photo of cats";
        String text2 = "A photo of dogs";

        // 对比结果
        float[] logit1 = clipModel.compareTextAndImage(imgobj, text1);
        float[] logit2 = clipModel.compareTextAndImage(imgobj, text2);

        // 相关性总和
        double total = Arrays.stream(new double[] {logit1[0], logit2[0]}).map(Math::exp).sum();

        // 文本1相关性
        double re1 = Math.exp(logit1[0]) / total;
        // 文本2相关性
        double re2 = Math.exp(logit2[0]) / total;

        System.out.println("文本:"+text1+",相关性:"+re1);
        System.out.println("文本:"+text2+",相关性:"+re2);

    }

}
