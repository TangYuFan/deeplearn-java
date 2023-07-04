package tool.util;

import tool.deeplearning.*;

import javax.swing.*;
import java.awt.*;
import java.io.File;

/**
*   @desc : 人像转为素描、动漫、等
*   @auth : tyf
*   @date : 2022-05-08  12:01:41
*/
public class FaceGanTransfer {


    public static void main(String[] args) throws Exception{

        // 原始图片
        String pic = new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\face.JPG";

        // 动漫画
        anime_gan.init1(new File("").getCanonicalPath()+"\\model\\deeplearning\\anime_gan\\face_paint_512_v2_0.onnx");
        anime_gan.ImageObj img1 = new anime_gan.ImageObj(pic);

        // 素描画
        u2_net.init(new File("").getCanonicalPath()+"\\model\\deeplearning\\u2_net\\u2net_portrait.onnx");
        u2_net.ImageObj img2 = new u2_net.ImageObj(pic);

        // 素描画
        informative_drawings.init(new File("").getCanonicalPath()+"\\model\\deeplearning\\informative_drawings\\opensketch_style_512x512.onnx");
        informative_drawings.ImageObj img3 = new informative_drawings.ImageObj(pic);

        // 卡通画
        style_gan_cartoon.init(new File("").getCanonicalPath()+"\\model\\deeplearning\\style_gan_cartoon\\minivision_female_photo2cartoon.onnx");
        style_gan_cartoon.ImageObj img4 = new style_gan_cartoon.ImageObj(pic);

        // ai修复
        gfp_gan_v1.init(new File("").getCanonicalPath()+"\\model\\deeplearning\\gfp_gan_v1\\GFPGANv1.3.onnx");
        gfp_gan_v1.ImageObj img5 = new gfp_gan_v1.ImageObj(pic);

        // 显示
        // 一行两列
        JPanel content = new JPanel(new GridLayout(1,6,5,5));

        // display the image in a window
        ImageIcon src = new ImageIcon(img1.in_img);
        JLabel le_src = new JLabel(src);

        ImageIcon dsr1 = new ImageIcon(img1.out_img);
        JLabel le_dsr1 = new JLabel(dsr1);

        ImageIcon dsr2 = new ImageIcon(img2.out_img);
        JLabel le_dsr2 = new JLabel(dsr2);

        ImageIcon dsr3 = new ImageIcon(img3.out_img);
        JLabel le_dsr3 = new JLabel(dsr3);

        ImageIcon dsr4 = new ImageIcon(img4.out_img);
        JLabel le_dsr4 = new JLabel(dsr4);

        ImageIcon dsr5 = new ImageIcon(img5.out_img);
        JLabel le_dsr5 = new JLabel(dsr5);

        content.add(le_src);
        content.add(le_dsr1);
        content.add(le_dsr2);
        content.add(le_dsr3);
        content.add(le_dsr4);
        content.add(le_dsr5);

        JFrame frame = new JFrame();
        frame.add(content);
        frame.pack();
        frame.setVisible(true);




    }


}
