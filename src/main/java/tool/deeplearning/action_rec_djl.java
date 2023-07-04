package tool.deeplearning;


import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

import java.io.File;
import java.nio.file.Path;

/**
*   @desc : MXNet图片分类：人体动作识别 , djl 推理
*   @auth : tyf
*   @date : 2022-06-13  12:57:21
*/
public class action_rec_djl {


    public static class ActionRecognition {

        private ActionRecognition() {}

        public static Classifications predict(String path) throws Exception {
            Path imageFile = new File(path).toPath();
            Image img = ImageFactory.getInstance().fromFile(imageFile);

            Criteria<Image, Classifications> criteria =
                    Criteria.builder()
                            .optApplication(Application.CV.ACTION_RECOGNITION)
                            .setTypes(Image.class, Classifications.class)
                            .optFilter("backbone", "inceptionv3")
                            .optFilter("dataset", "ucf101")
                            .optEngine("MXNet")
                            .optProgress(new ProgressBar())
                            .build();

            try (ZooModel<Image, Classifications> inception = criteria.loadModel()) {
                try (Predictor<Image, Classifications> action = inception.newPredictor()) {
                    return action.predict(img);
                }
            }
        }
    }


    public static void main(String[] args) throws Exception{

        // 图片
        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\action_rec_djl\\action_discus_throw.png";

        // 模型使用mxnet引擎
        // <dependency>
        //			<groupId>ai.djl.mxnet</groupId>
        //			<artifactId>mxnet-model-zoo</artifactId>
        //			<version>0.22.1</version>
        //		</dependency>

        // 会自动下载文件
        // https://mlrepo.djl.ai/model/cv/action_recognition/ai/djl/mxnet/action_recognition/classes.txt
        // https://mlrepo.djl.ai/model/cv/action_recognition/ai/djl/mxnet/action_recognition/0.0.1/inceptionv3_ucf101-symbol.json

        Classifications classification = ActionRecognition.predict(img);
        System.out.println("分类结果:");
        System.out.println(classification);
    }


}





