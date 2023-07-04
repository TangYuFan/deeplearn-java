package tool.deeplearning;


import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.opencv.core.Core;

import java.io.File;
import java.util.Arrays;

/**
*   @desc : PicoDet目标检测
 *
 *          参考代码：
 *          https://github.com/hpc203/picodet-onnxruntime
 *
 *          不过在PicoDet官方代码仓库里提供了10个.onnx文件， 我逐个运行之后，发现只有4个.onnx文件是onnxruntime库能正常读取的
 *
 *          coco.names: 类别信息
 *          picodet_xxx: 不同尺寸的模型
 *
*   @auth : tyf
*   @date : 2022-05-23  16:45:05
*/
public class picodet_object_dec {

    // 模型1
    public static OrtEnvironment env;
    public static OrtSession session;

    // 环境初始化
    public static void init(String weight) throws Exception{
        // opencv 库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        env = OrtEnvironment.getEnvironment();
        session = env.createSession(weight, new OrtSession.SessionOptions());

        // 打印模型信息,获取输入输出的shape以及类型：
        System.out.println("---------模型输入-----------");
        session.getInputInfo().entrySet().stream().forEach(n->{
            String inputName = n.getKey();
            NodeInfo inputInfo = n.getValue();
            long[] shape = ((TensorInfo)inputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)inputInfo.getInfo()).type.toString();
            System.out.println(inputName+" -> "+ Arrays.toString(shape)+" -> "+javaType);
        });
        System.out.println("---------模型输出-----------");
        session.getOutputInfo().entrySet().stream().forEach(n->{
            String outputName = n.getKey();
            NodeInfo outputInfo = n.getValue();
            long[] shape = ((TensorInfo)outputInfo.getInfo()).getShape();
            String javaType = ((TensorInfo)outputInfo.getInfo()).type.toString();
            System.out.println(outputName+" -> "+Arrays.toString(shape)+" -> "+javaType);
        });
        session.getMetadata().getCustomMetadata().entrySet().forEach(n->{
            System.out.println("元数据:"+n.getKey()+","+n.getValue());
        });

    }


    public static void main(String[] args) throws Exception{


        // ---------模型输入-----------
        // image -> [1, 3, 320, 320] -> FLOAT
        // ---------模型输出-----------
        // save_infer_model/scale_0.tmp_1 -> [1, 1600, 80] -> FLOAT
        // save_infer_model/scale_1.tmp_1 -> [1, 400, 80] -> FLOAT
        // save_infer_model/scale_2.tmp_1 -> [1, 100, 80] -> FLOAT
        // save_infer_model/scale_3.tmp_1 -> [1, 25, 80] -> FLOAT
        // save_infer_model/scale_4.tmp_1 -> [1, 1600, 32] -> FLOAT
        // save_infer_model/scale_5.tmp_1 -> [1, 400, 32] -> FLOAT
        // save_infer_model/scale_6.tmp_1 -> [1, 100, 32] -> FLOAT
        // save_infer_model/scale_7.tmp_1 -> [1, 25, 32] -> FLOAT
        init(new File("").getCanonicalPath()+
                "\\model\\deeplearning\\picodet_object_dec\\picodet_m_320_coco.onnx");



    }


}
