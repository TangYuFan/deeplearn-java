package tool.deeplearning;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.FloatBuffer;
import java.util.*;


/**
*   @desc : meta-ai sam , 使用抠图点进行分割
 *
 *
*   @auth : tyf
*   @date : 2022-04-25  09:34:40
*/
public class metaai_sam_test_onnx {


    // 模型1
    public static OrtEnvironment env1;
    public static OrtSession session1;

    // 模型2
    public static OrtEnvironment env2;
    public static OrtSession session2;


    // 模型1
    public static void init1(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env1 = OrtEnvironment.getEnvironment();
        session1 = env1.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型1输入-----------");
        session1.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型1输出-----------");
        session1.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session1.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }

    // 模型2
    public static void init2(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env2 = OrtEnvironment.getEnvironment();
        session2 = env2.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型2输入-----------");
        session2.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型2输出-----------");
        session2.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
//        session2.getMetadata().getCustomMetadata().entrySet().forEach(n->{
//            System.out.println("元数据:"+n.getKey()+","+n.getValue());
//        });

    }


    public static class ImageObj{

        // 原始图片
        Mat src;
        // 模型1输入
        Mat dst_3_1024_1024;
        // 原始图片
        BufferedImage image_3_1024_1024;
        // 模型1输出
        float[][][] image_embeddings;
        // 提示的坐标点
        ArrayList<float[]> points;
        // 保存分割得到的mask
        float[][] info;
        public ImageObj(String image) {
            this.src = this.readImg(image);
            this.dst_3_1024_1024 = this.resizeWithoutPadding(src,1024,1024);
            this.image_3_1024_1024 = mat2BufferedImage(this.dst_3_1024_1024);
        }

        public void setPoints(ArrayList<float[]> pp) {
            // 设置抠图点提示
            this.points = new ArrayList<>();
            pp.stream().forEach(n->{
                float[] xyl = n;
                // 坐标转到 1024*1024上
                float x = xyl[0] * 1024f / Float.valueOf(src.width());
                float y = xyl[1] * 1024f / Float.valueOf(src.height());
                float l = n[2];
                points.add(new float[]{x,y,l});
            });
        }

        public Mat readImg(String path){
            Mat img = Imgcodecs.imread(path);
            return img;
        }
        public Mat resizeWithoutPadding(Mat src,int inputWidth,int inputHeight){
            // 调整图像大小
            Mat resizedImage = new Mat();
            Size size = new Size(inputWidth, inputHeight);
            Imgproc.resize(src, resizedImage, size, 0, 0, Imgproc.INTER_AREA);
            return resizedImage;
        }
        // 3维转1维
        public float[] chw2chw(float[][][] chw,int c,int h,int w){
            float[] res = new float[ c * h * w ];

            int index = 0;
            for(int i=0;i<c;i++){
                for(int j=0;j<h;j++){
                    for(int k=0;k<w;k++){
                        float d = chw[i][j][k];
                        res[index] = d;
                        index++;
                    }
                }
            }

            return res;
        }

        // 推理1
        public void infenence1() throws Exception{

            // 输入(c,h,w)的图像，减均值除以方差，对图像进行缩放，按照长边缩放成1024，
            // 短边不够就pad,得到(c,1024,1024)的图像，经过image encoder，
            // 得到对图像16倍下采样的feature，大小为(256,64,64)。

            // BGR -> RGB
            Imgproc.cvtColor(dst_3_1024_1024, dst_3_1024_1024, Imgproc.COLOR_BGR2RGB);
            dst_3_1024_1024.convertTo(dst_3_1024_1024, CvType.CV_32FC1);

            // 转为 whc
            float[] whc = new float[ 3 * 1024 * 1024];
            dst_3_1024_1024.get(0, 0, whc);

            // 转为 cwh 并归一化TODO
            float[] chw = new float[whc.length];
            int j = 0;
            for (int ch = 0; ch < 3; ++ch) {
                for (int i = ch; i < whc.length; i += 3) {
                    chw[j] = whc[i];
                    j++;
                }
            }

            // 计算均值
            float mean = 0.0f;
            float std = 0.0f;
            for (int i = 0; i < chw.length; i++) {
                mean += chw[i];
            }
            mean /= chw.length;
            for (int i = 0; i < chw.length; i++) {
                std += Math.pow(chw[i] - mean, 2);
            }
            std = (float) Math.sqrt(std / chw.length);

            // 再将所有元素减去均值并除标准差
            for (int i = 0; i < chw.length; i++) {
                chw[i] = (chw[i] - mean) / std;
            }

            // 推理
            OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(chw), new long[]{1,3,1024,1024});
            // 推理
            OrtSession.Result res = session1.run(Collections.singletonMap("x", tensor));

            // 输出 256 * 64 * 64
            float[][][] image_embeddings  = ((float[][][][])(res.get(0)).getValue())[0];
            this.image_embeddings = image_embeddings;

        }

        // 推理2
        public void infenence2() throws Exception{


            // image_embeddings 图片编码 [1, 256, 64, 64]
            float[] chw = this.chw2chw(this.image_embeddings,256,64,64);
            OnnxTensor _image_embeddings = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,256, 64, 64});

            // point_coords 抠图点 [1, -1, 2]
            float[] pc = new float[points.size()*2];
            // point_labels 类别 [1, -1] 每个抠图点需要设置属于前景还是背景
            float[] pc_label = new float[points.size()];

            for(int i=0;i<points.size();i++){
                // 需要从原始坐标转换到 1024*1024 坐标系中
                float[] xyl = points.get(i);
                pc[i*2] = xyl[0];
                pc[i*2+1] = xyl[1];
                pc_label[i] = xyl[2];
            }

            // 提示点
            OnnxTensor _point_coords = OnnxTensor.createTensor(env2, FloatBuffer.wrap(pc), new long[]{1,points.size(),2});
            OnnxTensor _point_labels = OnnxTensor.createTensor(env2, FloatBuffer.wrap(pc_label), new long[]{1,points.size()});

            //  orig_im_size 原始图像尺寸 [2]
            OnnxTensor _orig_im_size = OnnxTensor.createTensor(env2, FloatBuffer.wrap(new float[]{1024,1024}), new long[]{2});


            // has_mask_input 输入是否包含mask [1]
            OnnxTensor _has_mask_input = OnnxTensor.createTensor(env2, FloatBuffer.wrap(new float[]{0}), new long[]{1});

            // mask_input 提示mask [1, 1, 256, 256] 因为设置了不包含mask这里生成一个固定长度无用的数组即可
            // 这里模型的输出可以作为下一次预测的输入,以提高提示准确性
            float[] ar_256_156 = new float[256*256];
            for(int i=0;i<256*156;i++){
                ar_256_156[i] = 0;
            }
            OnnxTensor _mask_input = OnnxTensor.createTensor(env2, FloatBuffer.wrap(ar_256_156), new long[]{1,1,256,256});

            // 封装参数
            Map<String,OnnxTensor> in = new HashMap<>();
            in.put("image_embeddings",_image_embeddings);
            in.put("point_coords", _point_coords);
            in.put("point_labels",_point_labels);
            in.put("has_mask_input",_has_mask_input);
            in.put("orig_im_size",_orig_im_size);
            in.put("mask_input",_mask_input);


            // 推理
            OrtSession.Result res = session2.run(in);

            // ---------模型2输出-----------
            // masks -> [-1, -1, -1, -1] -> FLOAT
            // iou_predictions -> [-1, 1] -> FLOAT
            // low_res_masks -> [-1, 1, -1, -1] -> FLOAT
            float[][][] masks  = ((float[][][][])(res.get(0)).getValue())[0];
            float[][] iou_predictions  = ((float[][])(res.get(1)).getValue());
            float[][][][] low_res_masks  = ((float[][][][])(res.get(2)).getValue());

            // 遍历每个 mask 分数最高的排最前面,那么只获取第零个即可
            int count = masks.length;

            for(int i=0;i < count;i++){
                // 这里输出的每个mask都是模型输入时 orig_im_size 指定的宽高 1024*1024
                float[][] info = masks[i];
                this.info = info;
                // 分数最高的排最前面,那么只获取第零个即可
                break;

            }

        }

        // Mat 转 BufferedImage
        public BufferedImage mat2BufferedImage(Mat mat){
            BufferedImage bufferedImage = null;
            try {
                // 将Mat对象转换为字节数组
                MatOfByte matOfByte = new MatOfByte();
                Imgcodecs.imencode(".jpg", mat, matOfByte);
                // 创建Java的ByteArrayInputStream对象
                byte[] byteArray = matOfByte.toArray();
                ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArray);
                // 使用ImageIO读取ByteArrayInputStream并将其转换为BufferedImage对象
                bufferedImage = ImageIO.read(byteArrayInputStream);
            } catch (Exception e) {
                e.printStackTrace();
            }
            return bufferedImage;
        }

        // 图片缩放
        public BufferedImage resize(BufferedImage img, int newWidth, int newHeight) {
            Image scaledImage = img.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
            BufferedImage scaledBufferedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = scaledBufferedImage.createGraphics();
            g2d.drawImage(scaledImage, 0, 0, null);
            g2d.dispose();
            return scaledBufferedImage;
        }

        public void show(){


            // 显示子图片
            int sub_w = info.length;
            int sub_h = info[0].length;

            // 原始图像 image_3_1024_1024 上进行mask标注
            for(int j=0;j<sub_w;j++){
                for(int k=0;k<sub_h;k++){
                    float da = info[j][k];
                    da = da + 1;
                    if(da>0.5){
                        // 修改颜色为绿色
                        image_3_1024_1024.setRGB(k,j, Color.GREEN.getRGB());
                    }
                }
            }

            // 将抠图点标注出来
            Color color = Color.RED;
            Graphics graphics = image_3_1024_1024.getGraphics();
            graphics.setColor(color);
            this.points.stream().forEach(n->{
                float x = n[0];
                float y = n[1];
                int radius = 5;  // 圆的半径
                graphics.fillOval(Float.valueOf(x - radius).intValue(), Float.valueOf(y - radius).intValue(), radius * 2, radius * 2);
            });

            graphics.drawImage(image_3_1024_1024, 0, 0, null);

            // 1024 * 1024 还原到原始大小
            BufferedImage showImg = resize(image_3_1024_1024,src.width(),src.height());

            // 弹窗显示
            JFrame frame = new JFrame();
            frame.setTitle("Meta-ai: SAM");
            JPanel content = new JPanel();
            content.add(new JLabel(new ImageIcon(showImg)));
            frame.add(content);
            frame.pack();
            frame.setVisible(true);


        }

    }


    public static void main(String[] args) throws Exception{



        // ---------模型1输入-----------
        // x -> [-1, 3, 1024, 1024] -> FLOAT
        // ---------模型1输出-----------
        // image_embeddings -> [-1, 256, -1, -1] -> FLOAT
        init1(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\metaai_sam_test_onnx\\encoder-vit_b.quant.onnx");


        // ---------模型2输入-----------
        // image_embeddings -> [1, 256, 64, 64] -> FLOAT
        // point_coords -> [1, -1, 2] -> FLOAT
        // point_labels -> [1, -1] -> FLOAT
        // mask_input -> [1, 1, 256, 256] -> FLOAT
        // has_mask_input -> [1] -> FLOAT
        // orig_im_size -> [2] -> FLOAT
        // ---------模型2输出-----------
        // masks -> [-1, -1, -1, -1] -> FLOAT
        // iou_predictions -> [-1, 1] -> FLOAT
        // low_res_masks -> [-1, 1, -1, -1] -> FLOAT
        init2(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\metaai_sam_test_onnx\\decoder-vit_b.quant.onnx");


        // 图片
        ImageObj imageObj = new ImageObj(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\metaai_sam_test_onnx\\truck.jpg");


        // 提示,这里使用抠图点进行提示,可以设置多个提示点以及属于前景还是背景 x y label
        // 一个物体可能属于父物体,mask会输出物体以及父物体的掩膜取第一个即可
        // 多个点同时指定在一个物体上,提高子物体更加精细化的分割
        ArrayList<float[]> points = new ArrayList<>();
//        points.add(new float[]{514,357,1});// 车窗户
//        points.add(new float[]{555,377,1});// 车窗户
//        points.add(new float[]{556,387,1});// 车窗户
//        points.add(new float[]{1063,590,1});// 车门
//        points.add(new float[]{1303,538,1});// 车门
//        points.add(new float[]{1159,468,1});// 车门
        points.add(new float[]{990,997,1});// 地面
        points.add(new float[]{990,997,1});// 地面
        points.add(new float[]{1597,982,1});// 地面
        imageObj.setPoints(points);

        // 推理
        imageObj.infenence1();
        imageObj.infenence2();

        // 显示
        imageObj.show();


    }



}
