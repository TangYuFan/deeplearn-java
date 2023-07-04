package tool.deeplearning;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.modality.cv.translator.InstanceSegmentationTranslator;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.*;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/**
*   @desc : maskrcnn 实例分割 , djl 推理
*   @auth : tyf
*   @date : 2022-06-13  14:22:26
*/
public class mask_rcnn_resnet18_v1b_coco_seg_djl {


    // 实例分割
    public static class InstanceSegmentation {
        Image out;
        private InstanceSegmentation() {
        }

        public DetectedObjects predict(String pic) throws IOException, ModelException, TranslateException {
            Path imageFile = new File(pic).toPath();
            Image img = ImageFactory.getInstance().fromFile(imageFile);

            Criteria<Image, DetectedObjects> criteria =
                    Criteria.builder()
                            .optApplication(Application.CV.INSTANCE_SEGMENTATION)
                            .setTypes(Image.class, DetectedObjects.class)
                            .optEngine("MXNet")
                            .optFilter("backbone", "resnet18")
                            .optFilter("flavor", "v1b")
                            .optFilter("dataset", "coco")
                            .optProgress(new ProgressBar())
                            .build();

            ZooModel<Image, DetectedObjects> model = criteria.loadModel();

            // 打印模型查看使用的 translater
            System.out.println(model);

            Predictor<Image, DetectedObjects> predictor = model.newPredictor();
            DetectedObjects detection = predictor.predict(img);
            saveBoundingBoxImage(img, detection);
            return detection;
        }

        private void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
            img.drawBoundingBoxes(detection);
            this.out = img;
        }


        private void show(){
            BufferedImage pic = (BufferedImage)out.getWrappedImage();
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

        // 会自动下载,这里从本地引入
        // 使用的 mxnet 引擎模型包含两个文件 params和json
        // https://mlrepo.djl.ai/model/cv/instance_segmentation/ai/djl/mxnet/mask_rcnn/0.0.1/mask_rcnn_resnet18_v1b_coco-symbol.json
        // https://mlrepo.djl.ai/model/cv/instance_segmentation/ai/djl/mxnet/mask_rcnn/0.0.1/mask_rcnn_resnet18_v1b_coco-0000.params.gz
        // https://mlrepo.djl.ai/model/cv/instance_segmentation/ai/djl/mxnet/mask_rcnn/classes.txt

        // 模型下载路径
        // C:\Users\tyf\.djl.ai\cache\repo\model\cv\instance_segmentation\ai\djl\mxnet\mask_rcnn\resnet18\v1b\coco\0.0.1

        // 模型=,mxnet模型包含两个文件
        // mask_rcnn_resnet18_v1b_coco-0000.params
        // mask_rcnn_resnet18_v1b_coco-symbol.json

        // 类别
        // classes.txt


        // 图片
        String pic = new File("").getCanonicalPath()+
                "\\model\\deeplearning\\mask_rcnn_resnet18_v1b_coco_seg_djl\\segmentation.jpg";


        // 实例分割,结果里面有类别和box,mask
        InstanceSegmentation det = new InstanceSegmentation();
        DetectedObjects detection = det.predict(pic);
        System.out.println("分割结果:"+detection);


        // 显示
        det.show();


    }


}
