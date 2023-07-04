package tool.deeplearning;


import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;


/**
*   @desc : 动物图片分类 resnet50/mobilenet_v2 ,djl 推理
 *
 *          参考连接:
 *          https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification
 *          https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_animals&en_category=ImageClassification
 *
 *          模型下载:
 *          - Link 1: https://github.com/mymagicpower/AIAS/releases/download/apps/animals.zip
 *          - Link 2: https://github.com/mymagicpower/AIAS/releases/download/apps/mobilenet_animals.zip
 *
*   @auth : tyf
*   @date : 2022-06-15  10:39:29
*/
public class pp_animals_classification {



    public static class AnimalTranslator implements Translator<Image, Classifications> {
        List<String> classes;

        String labels;
        public AnimalTranslator(String labels) {
            this.labels = labels;
        }

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {

            // 加载类别
            classes = Utils.readLines(new File(labels).toPath(), true);
        }

        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray prob = list.singletonOrThrow();
            return new Classifications(this.classes, prob);
        }

        public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);

            float percent = 256f / Math.min(input.getWidth(), input.getHeight());
            int resizedWidth = Math.round(input.getWidth() * percent);
            int resizedHeight = Math.round(input.getHeight() * percent);
//        img = img.resize((resizedWidth, resizedHeight), Image.LANCZOS)

            array = NDImageUtils.resize(array,resizedWidth,resizedHeight);
            array = NDImageUtils.centerCrop(array,224,224);

            // The network by default takes float32
            if (!array.getDataType().equals(DataType.FLOAT32)) {
                array = array.toType(DataType.FLOAT32, false);
            }

            array = array.transpose(2, 0, 1).div(255f);  // HWC -> CHW RGB

            NDArray mean =
                    ctx.getNDManager().create(new float[] {0.485f, 0.456f, 0.406f}, new Shape(3, 1, 1));
            NDArray std =
                    ctx.getNDManager().create(new float[] {0.229f, 0.224f, 0.225f}, new Shape(3, 1, 1));

            array = array.sub(mean);
            array = array.div(std);

            array = array.expandDims(0);

            return new NDList(array);
        }

        public Batchifier getBatchifier() {
            return null;
        }
    }


    public static class AnimalsClassification {

        private static final Logger logger = LoggerFactory.getLogger(AnimalsClassification.class);

        private AnimalsClassification() {}

        public static Classifications predict(String modelPath,String label,Image img)
                throws IOException, ModelException, TranslateException {
            Classifications classifications = AnimalsClassification.classfier(img,modelPath,label);
            List<Classifications.Classification> items = classifications.items();
            double sum = 0;
            double max = 0;
            double[] probArr = new double[items.size()];

            List<String> names = new ArrayList<>();
            List<Double> probs = new ArrayList<>();

            for (int i = 0; i < items.size(); i++) {
                Classifications.Classification item = items.get(i);
                double prob = item.getProbability();
                probArr[i] = prob;
                if (prob > max) max = prob;
            }

            for (int i = 0; i < items.size(); i++) {
                probArr[i] = Math.exp(probArr[i] - max);
                sum = sum + probArr[i];
            }

            for (int i = 0; i < items.size(); i++) {
                Classifications.Classification item = items.get(i);
                names.add(item.getClassName());
                probs.add(probArr[i]);
            }

            return new Classifications(names, probs);
        }

        public static Classifications classfier(Image img,String modelPath,String label)
                throws IOException, ModelException, TranslateException {

            Criteria<Image, Classifications> criteria =
                    Criteria.builder()
                            .optEngine("PaddlePaddle")
                            .setTypes(Image.class, Classifications.class)
                            .optModelPath(new File(modelPath).toPath())
                            .optModelName("inference")
                            .optTranslator(new AnimalTranslator(label))
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel rotateModel = ModelZoo.loadModel(criteria)) {
                try (Predictor<Image, Classifications> classifier = rotateModel.newPredictor()) {
                    Classifications classifications = classifier.predict(img);
                    return classifications;
                }
            }
        }
    }


    public static void main(String[] args) throws IOException, ModelException, TranslateException {


        // 模型  mobilenet_animals.zip   animals.zip
        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_animals_classification\\mobilenet_animals.zip";


        // 类别
        String label = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_animals_classification\\label_list.txt";


        // 图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\pp_animals_classification\\tiger.jpeg";

        Path imageFile = new File(pic).toPath();
        Image image = ImageFactory.getInstance().fromFile(imageFile);
        Classifications classifications = AnimalsClassification.predict(model,label,image);
        Classifications.Classification bestItem = classifications.best();
        System.out.println(bestItem.getClassName() + " : " + bestItem.getProbability());
        //    List<Classifications.Classification> items = classifications.items();
        //    List<String> names = new ArrayList<>();
        //    List<Double> probs = new ArrayList<>();
        //    for (int i = 0; i < items.size(); i++) {
        //      Classifications.Classification item = items.get(i);
        //      names.add(item.getClassName());
        //      probs.add(item.getProbability());
        //    }
        System.out.println("分类结果：");
        System.out.println(classifications);

    }


}
