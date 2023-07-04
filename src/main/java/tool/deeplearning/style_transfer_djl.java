package tool.deeplearning;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.StyleTransferTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;



/**
*   @desc : 动画风格迁移  djl 部署
*   @auth : tyf
*   @date : 2022-06-12  18:56:06
*/
public class style_transfer_djl {


    public static Image transfer(String modelPath,String imagePath) throws Exception{

        Image image = ImageFactory.getInstance().fromFile(Paths.get(imagePath));

        Criteria<Image, Image> criteria = Criteria.builder()
                        .optApplication(Application.CV.IMAGE_GENERATION)
                        .setTypes(Image.class, Image.class)
                        .optModelPath(new File(modelPath).toPath())
                        .optProgress(new ProgressBar())
                        .optTranslatorFactory(new StyleTransferTranslatorFactory())
                        .optEngine("PyTorch")
                        .build();

        try (ZooModel<Image, Image> model = criteria.loadModel();
             Predictor<Image, Image> styler = model.newPredictor()) {
            return styler.predict(image);
        }
    }

    public static void save(Image image, String name, String path) throws Exception{
        Path outputPath = Paths.get(path);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(name + ".png");
        image.save(Files.newOutputStream(imagePath), "png");
    }


    public static void main(String[] args) throws Exception{

        String imagePath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\style_transfer_djl\\img.jpg";

        String modelPath = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\style_transfer_djl\\style_cezanne.pt";


        Image output = transfer(modelPath,imagePath);

        // 弹窗显示
        BufferedImage out = (BufferedImage)output.getWrappedImage();
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
