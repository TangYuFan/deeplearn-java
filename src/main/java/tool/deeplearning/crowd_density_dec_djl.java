package tool.deeplearning;

import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.zoo.Criteria;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.io.IOException;


/**
*   @desc : 人群密度检测（PaddlePaddle-CrowdNet）,人计数,人流密度图  , djl 推理
 *
 *          百度飞将开源模型:
 *          https://aistudio.baidu.com/aistudio/projectdetail/519178
 *
*   @auth : tyf
*   @date : 2022-06-14  17:43:03
*/
public class crowd_density_dec_djl {



    // 人群密度检测
    public static class CrowdDetect {

        public CrowdDetect() {}
        public Criteria<Image, NDList> criteria(String model) {
            Criteria<Image, NDList> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, NDList.class)
                            .optModelPath(new File(model).toPath())
                            .optEngine("PaddlePaddle")
                            .optTranslator(new CrowdTranslator())
                            .optProgress(new ProgressBar())
                            .build();

            return criteria;
        }

        private final class CrowdTranslator implements Translator<Image, NDList> {
            CrowdTranslator() {}
            @Override
            public NDList processOutput(TranslatorContext ctx, NDList list) {
                list.detach();
                return list;
            }
            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
                array = NDImageUtils.resize(array, 640, 480);
                array = array.div(255f);
                array = array.transpose().reshape(1, 3, 640, 480);
                if (!array.getDataType().equals(DataType.FLOAT32)) {
                    array = array.toType(DataType.FLOAT32, false);
                }
                return new NDList(array);
            }

            @Override
            public Batchifier getBatchifier() {
                return null;
            }
        }
    }



    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // 模型
        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\crowd_density_dec_djl\\crowdnet.zip";

        // 测试图片
        String picPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\crowd_density_dec_djl\\crowd1.jpg";



        // 推理
        Criteria<Image, NDList> criteria = new CrowdDetect().criteria(modelPath);

        try (ZooModel model = ModelZoo.loadModel(criteria);

             Predictor<Image, NDList> predictor = model.newPredictor()) {

            Path imageFile = new File(picPath).toPath();
            Image image = ImageFactory.getInstance().fromFile(imageFile);

            NDList list = predictor.predict(image);

            //quantity 为人数
            float q = list.get(1).toFloatArray()[0];
            int quantity = (int)(Math.abs(q) + 0.5);
            System.out.println("人数 quantity: " + quantity);

            // density 为密度图
            NDArray densityArray = list.get(0);
            System.out.println("密度图 density: " + densityArray);

        }
    }

}
