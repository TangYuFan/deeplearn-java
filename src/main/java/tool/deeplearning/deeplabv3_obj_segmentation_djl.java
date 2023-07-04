package tool.deeplearning;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;


/**
*   @desc : deeplabv3 实例分割 ,  djl 推理
 *
 *      使用的模型:
 *      https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip
 *
 *
*   @auth : tyf
*   @date : 2022-06-13  20:03:22
*/
public class deeplabv3_obj_segmentation_djl {

    public static class SemanticSegmentation {

        private static final Logger logger = LoggerFactory.getLogger(SemanticSegmentation.class);

        // 模型路径
        String model;

        // 分割得到的对象
        Image out;

        private SemanticSegmentation(String model) {
            this.model = model;
        }


        public void predict(String imgPath) throws IOException, ModelException, TranslateException {
            Path imageFile = new File(imgPath).toPath();
            ImageFactory factory = ImageFactory.getInstance();
            Image img = factory.fromFile(imageFile);

            Criteria<Image, CategoryMask> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, CategoryMask.class)
                            .optModelPath(new File(model).toPath())
                            .optTranslatorFactory(new SemanticSegmentationTranslatorFactory())
                            .optEngine("PyTorch")
                            .optProgress(new ProgressBar())
                            .build();


            try (ZooModel<Image, CategoryMask> model = criteria.loadModel();

                 Predictor<Image, CategoryMask> predictor = model.newPredictor()) {
                CategoryMask mask = predictor.predict(img);


                // 指定透明度,绘制所有对象的mask
                this.out = img.duplicate();
                mask.drawMask(out, 180, 0);


            }
        }


        public void show(){

            BufferedImage pic = (BufferedImage) this.out.getWrappedImage();
            JFrame frame = new JFrame("Image");
            frame.setSize(pic.getWidth(), pic.getHeight());
            JPanel panel = new JPanel();
            panel.add(new JLabel(new ImageIcon(pic)));
            frame.getContentPane().add(panel);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);

        }

    }


    public static void main(String[] args) throws IOException, ModelException, TranslateException {



        String model = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\deeplabv3_obj_segmentation_djl\\deeplabv3.pt";


        String img = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\deeplabv3_obj_segmentation_djl\\dog_bike_car.jpg";


        // 推理
        SemanticSegmentation segmentation = new SemanticSegmentation(model);
        segmentation.predict(img);

        // 显示
        segmentation.show();

    }



}
