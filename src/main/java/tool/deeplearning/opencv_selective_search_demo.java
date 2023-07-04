package tool.deeplearning;


import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_ximgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_ximgproc.SelectiveSearchSegmentation;

import javax.swing.*;

/**
*   @desc : opencv 实现 rcnn 的 SelectiveSearch 算法
*   @auth : tyf
*   @date : 2022-05-05  18:01:14
*/
public class opencv_selective_search_demo {

    public static void main(String[] args) {


        // 加载图片
        Mat image = opencv_imgcodecs.imread("C:\\work\\workspace\\duijieTool\\model\\yolo\\muzo.png");

        // ss 算法对象
        SelectiveSearchSegmentation sss = opencv_ximgproc.createSelectiveSearchSegmentation();
        sss.setBaseImage(image);
        sss.switchToSelectiveSearchFast();

        // 获取目标框
        RectVector rects = new RectVector();
        sss.process(rects);
        Rect[] rectArray = rects.get();

        System.out.println("检测完毕,");

        // 标注
        for (Rect rect : rectArray) {
            opencv_imgproc.rectangle(image, rect.tl(), rect.br(), new Scalar(1));
        }

        //  弹窗显示
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame canvas = new CanvasFrame("Image", 1);
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.showImage(converter.convert(image));

    }

}
